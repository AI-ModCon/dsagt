"""
Claude Code agent setup.

Install: ``npm i -g @anthropic-ai/claude-code``.
Generates: ``CLAUDE.md`` (instructions) and ``.mcp.json`` (MCP config).

The user sets ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_MODEL``,
``ANTHROPIC_BASE_URL``) in the shell and Claude Code talks directly to its
provider.  Agent-side traces are recovered from Claude's on-disk transcript
by the periodic pass (``ClaudeReader``, ``ClaudeTranslator``,
``MLflowSink``), the same way as for every other agent, so the agent's
environment carries no telemetry setting.

Prompt caching: Claude Code handles Anthropic prompt caching natively
against the Anthropic API.  Users on a custom ``ANTHROPIC_BASE_URL`` that
proxies to a non-Anthropic provider lose caching.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    AgentSetup,
    _write_dsagt_block,
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

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [
            working_dir / "CLAUDE.md",
            working_dir / ".mcp.json",
            working_dir / ".claude",
        ]

    def vscode_hint(self, project_dir: Path) -> list[str]:
        return [f"Open {project_dir} in VS Code and start the Claude extension."]

    def write_static(self, working_dir: Path, *, auto_assess: bool = True) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions(auto_assess)
        if instructions:
            action = _write_dsagt_block(working_dir / "CLAUDE.md", instructions)
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
        children.  Claude passes its parent env to them, and writing the block
        into the JSON as well covers a shell where those vars are unset.

        The periodic pass (``ClaudeReader``, ``ClaudeTranslator``,
        ``MLflowSink``) produces Claude's traces, the same way as for every
        other agent, so the file carries no trace setting; MLflow's ``autolog
        claude`` Stop hook would log the same turns a second time.
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

        # Skills are mirrored into .claude/skills/ by AgentSetup.setup_skills
        # (driven by native_skills_dir) in dynamic_agent_record.  Claude reads
        # them on its next start; this runs at init/start, before launch.
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

        ``--verbose`` streams tool-call progress as it happens.

        ``--max-thinking-tokens 4096`` caps per-turn extended thinking.
        Claude Code's default is much higher, and a multi-task smoke prompt
        can spend tens of seconds per turn on thinking alone.  4096 is
        enough for the bounded reasoning each smoke task needs.
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
