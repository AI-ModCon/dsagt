"""
Claude Code agent setup.

Install: ``npm i -g @anthropic-ai/claude-code``.
Generates: ``CLAUDE.md`` (instructions) and ``.mcp.json`` (MCP config).

The user brings ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_MODEL``,
``ANTHROPIC_BASE_URL``) themselves and Claude Code talks directly to its
provider.  DSAGT sets **no** telemetry env on the agent — agent-side traces
are recovered post-hoc from Claude's on-disk transcript by DSAGT's own
serverless pipeline (MCP-server heartbeat → ``ClaudeReader`` →
``ClaudeTranslator`` → ``MLflowSink``), uniformly with every other agent; not
by forcing native OTel emission or wiring MLflow's autolog hook.

Cache-marker injection: Claude Code handles Anthropic prompt caching
natively against the Anthropic API.  Users on a custom
``ANTHROPIC_BASE_URL`` that proxies to a non-Anthropic provider lose
caching.
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


class ClaudeSetup(AgentSetup):
    name = "claude"
    base_command = ["claude"]
    static_marker = "CLAUDE.md"
    native_skills_dir = ".claude/skills"
    install_hint = "Install with `npm i -g @anthropic-ai/claude-code`."
    # Anthropic-protocol native.
    credential_env_vars = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
    )
    otel_payload_support = "full"
    credential_hints = (
        ("ANTHROPIC_API_KEY", "your Anthropic API key (skip if subscription-authed)"),
        ("ANTHROPIC_BASE_URL", "optional gateway / proxy URL"),
        ("ANTHROPIC_MODEL", "optional model override"),
    )

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [
            working_dir / "CLAUDE.md",
            working_dir / ".mcp.json",
            working_dir / ".claude",
        ]

    def vscode_hint(self, project_dir: Path) -> list[str]:
        return [f"Open {project_dir} in VS Code and start the Claude extension."]

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / "CLAUDE.md",
                instructions,
                _DSAGT_MARKER,
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
        """Write ``.mcp.json``.

        The env block carries DSAGT/MLflow/embedding routing for the MCP-server
        children — claude inherits parent env into them, but baking it into the
        JSON is robust against shells that don't have those vars set.

        No trace wiring here: DSAGT's own serverless pipeline (the MCP-server
        heartbeat → ``ClaudeReader`` → ``ClaudeTranslator`` → ``MLflowSink``)
        produces Claude's traces, uniformly with every other agent — so we do
        NOT also wire MLflow's ``autolog claude`` Stop hook (which would
        double-log the same turns, and only Claude can use it serverlessly).
        """
        del env, pdir
        actions: list[str] = []
        env_block = _mcp_env_block(config)

        entry: dict = {"command": "uv", "args": _mcp_server_args()}
        if env_block:
            entry["env"] = env_block
        mcp_config: dict = {"mcpServers": {"dsagt": entry}}

        mcp_path = working_dir / ".mcp.json"
        mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
        actions.append(f"Wrote {mcp_path}")

        # Skills are mirrored into .claude/skills/ centrally via
        # AgentSetup.setup_skills (driven by native_skills_dir) in
        # dynamic_agent_record — see base.py.  Picked up on the next Claude
        # start, which is fine: this runs at init/start, before launch.
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
            "--max-thinking-tokens",
            "4096",
            "-p",
            text,
        ]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
