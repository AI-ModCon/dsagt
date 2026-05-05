"""
Claude Code agent setup.

Install: ``npm i -g @anthropic-ai/claude-code``.
Generates: ``.mcp.json``, ``CLAUDE.md``, ``.dsagt_env``.

Post-proxy: the user brings ``ANTHROPIC_API_KEY`` (and optionally
``ANTHROPIC_MODEL``, ``ANTHROPIC_BASE_URL``) themselves and Claude Code
talks directly to its provider.  We only inject OTel telemetry env vars
so the agent's LLM-call traces land in the project's MLflow.

OTel support: **full** for native MLflow visibility (verified).  Tool
args land on the ``claude_code.tool_result`` event when
``OTEL_LOG_TOOL_DETAILS=1`` and the assistant's response (with
tool_use blocks) lands on the ``api_response_body`` event when
``OTEL_LOG_RAW_API_BODIES=1``; both are gated.
``CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`` enables the trace hierarchy.
All four flags are set in ``_CLAUDE_TELEMETRY_ENV`` below.  Cited from
https://code.claude.com/docs/en/monitoring-usage.md and verified
end-to-end by the smoke-test harness once ``session.id`` was added to
``OTEL_RESOURCE_ATTRIBUTES``.

Memory extraction: **does NOT work without --enable-proxy**, even
though traces are visible in the MLflow UI.  Claude Code emits its
conversation via OTel *log events* (``api_response_body``,
``tool_result``), which is a different shape from the LiteLLM-autolog
``mlflow.spanInputs`` / ``mlflow.spanOutputs`` shape that
``memory.drain_session_traces`` reads.  Users who want both visibility
*and* extraction should run ``dsagt start --enable-proxy``.  See
``agents/__init__.py`` module docstring for the design rationale.

Truncation caveats: ``tool_input`` truncates per-value at 512 chars and
total at ~4 KB; ``api_response_body`` is capped at 60 KB.  Large diffs
or huge tool outputs may be clipped.

Cache-marker injection: Claude Code handles Anthropic prompt caching
natively against the Anthropic API, so the (deleted) proxy's
``_inject_cache_breakpoints`` is unnecessary here.  Users on a custom
``ANTHROPIC_BASE_URL`` that proxies to a non-Anthropic provider lose
caching either way.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_env_block,
    _mcp_server_args,
    _run_simple_script,
)


# Env vars Claude Code reads to gate full LLM-call telemetry.  Without
# these, the OTel spans only carry counts/cost/duration — tool_use
# payloads (which memory extraction needs) are absent.  See
# https://code.claude.com/docs/en/monitoring-usage.md.
_CLAUDE_TELEMETRY_ENV: dict[str, str] = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
    "OTEL_LOG_TOOL_DETAILS": "1",
    "OTEL_LOG_RAW_API_BODIES": "1",
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
}


class ClaudeSetup(AgentSetup):
    name = "claude"
    base_command = ["claude"]
    static_marker = "CLAUDE.md"
    install_hint = "Install with `npm i -g @anthropic-ai/claude-code`."
    # Anthropic-protocol native; cross-protocol routing requires the proxy.
    credential_env_vars = (
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
    )
    otel_payload_support = "full"
    telemetry_env = _CLAUDE_TELEMETRY_ENV
    credential_hints = (
        ("ANTHROPIC_API_KEY", "your Anthropic API key (skip if subscription-authed)"),
        ("ANTHROPIC_BASE_URL", "optional gateway / proxy URL"),
        ("ANTHROPIC_MODEL", "optional model override"),
    )

    def env_overrides(self, config: dict) -> dict[str, str]:
        """Phase-2 proxy-mode hook: pin ``ANTHROPIC_MODEL`` to the
        upstream-served name so claude doesn't fall back to its
        built-in default.  Only fires when ``config["proxy"]["port"]``
        is set (gated by ``agents/__init__.py:agent_env``).

        ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY`` are set by
        :meth:`proxy_env_overrides` (base default) — point at the
        localhost proxy with the sentinel key.  Claude posts
        ``/v1/messages`` to the proxy regardless of upstream protocol;
        the proxy translates.
        """
        model = (config.get("llm") or {}).get("model")
        if model and not str(model).startswith("${"):
            return {"ANTHROPIC_MODEL": model}
        return {}

    def vscode_hint(self, project_dir: Path) -> list[str]:
        return [f"Open {project_dir} in VS Code and start the Claude extension."]

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / "CLAUDE.md", instructions, _DSAGT_MARKER,
            )
            if action:
                actions.append(action)
        return actions

    def write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        """Write ``.mcp.json``.  Claude Code reads it from cwd at launch.

        The env block in ``.mcp.json`` carries DSAGT/MLflow/embedding
        routing for the MCP-server children — claude inherits parent
        env into them, but baking it into the JSON is robust against
        shells that don't have those vars set.
        """
        del env, pdir
        actions: list[str] = []
        env_block = _mcp_env_block(config)

        mcp_config: dict = {"mcpServers": {}}
        for server in ("registry", "knowledge"):
            entry: dict = {"command": "uv", "args": _mcp_server_args(server)}
            if env_block:
                entry["env"] = env_block
            mcp_config["mcpServers"][f"dsagt-{server}"] = entry

        mcp_path = working_dir / ".mcp.json"
        mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
        actions.append(f"Wrote {mcp_path}")
        return actions

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``claude -p`` call with the entire script as one prompt.

        ``--verbose`` streams tool-call progress instead of buffering
        everything until the agent finishes.

        ``--max-thinking-tokens 4096`` caps per-turn extended thinking
        — claude code's default is much higher and a multi-task smoke
        prompt can spend tens of seconds per turn just thinking.  4096
        is enough headroom for the bounded reasoning each smoke task
        needs.
        """
        del config, max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--verbose",
            "--max-thinking-tokens", "4096",
            "-p", text,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
