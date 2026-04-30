"""
Claude Code agent setup.

Install: ``npm i -g @anthropic-ai/claude-code``.
Generates: ``.mcp.json``, ``CLAUDE.md``, ``.dsagt_env``.
Proxy routing: ``ANTHROPIC_BASE_URL``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _PROXY_FORWARDED_SENTINEL,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_env_block,
    _mcp_server_args,
    _run_simple_script,
    _write_env_file,
)


class ClaudeSetup(AgentSetup):
    name = "claude"
    base_command = ["claude"]
    static_marker = "CLAUDE.md"
    install_hint = "Install with `npm i -g @anthropic-ai/claude-code`."

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
        proxy_port: int,
    ) -> list[str]:
        """Claude Code DOES inherit parent process env into MCP server children,
        so the explicit env block in ``.mcp.json`` is partly redundant — but we
        include it anyway to keep behavior identical when the agent is launched
        outside our ``dsagt start`` flow (e.g. someone runs ``claude`` directly
        in the project dir after sourcing ``.dsagt_env``).
        """
        del env  # MCP env block is config-derived, not env-derived
        actions: list[str] = []
        env_block = _mcp_env_block(config, proxy_port)

        mcp_config: dict = {"mcpServers": {}}
        for server in ("registry", "knowledge"):
            entry: dict = {"command": "uv", "args": _mcp_server_args(server)}
            if env_block:
                entry["env"] = env_block
            mcp_config["mcpServers"][f"dsagt-{server}"] = entry

        mcp_path = working_dir / ".mcp.json"
        mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
        actions.append(f"Wrote {mcp_path}")

        env_path = working_dir / ".dsagt_env"
        _write_env_file(env_path, {
            "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
        })
        actions.append(f"Wrote {env_path}")
        return actions

    def env_overrides(self, config: dict, proxy_port: int) -> dict[str, str]:
        # Pin the model.  Without this Claude Code falls back to its built-in
        # default (currently claude-sonnet-4-6) which won't exist on project
        # gateways like PNNL's ai-incubator-api and the proxy will 400 every
        # request.
        return {
            "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
            "ANTHROPIC_MODEL": config["llm"]["model"],
            "ANTHROPIC_API_KEY": _PROXY_FORWARDED_SENTINEL,
        }

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``claude -p`` call with the entire script as one prompt.

        Mirrors goose/cline/roo: hand the agent the whole script in one
        invocation and let its internal tool-call loop drive multi-turn
        behavior.  ``max_turns`` is not applicable in single-shot mode
        (Claude Code has no flag exposing its internal turn cap); the smoke
        wrapper's wall-clock timeout is the safety net.
        """
        del config, max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        cmd = ["claude", "--dangerously-skip-permissions", "-p", text]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
