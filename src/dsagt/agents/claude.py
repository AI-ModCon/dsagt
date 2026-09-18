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
import sys
from pathlib import Path

from .base import (
    AgentSetup,
    _write_dsagt_block,
    _load_master_instructions,
    _mcp_env_block,
    _mcp_server_args,
    _run_simple_script,
)

_BASH_GUARD_SCRIPT = "dsagt-bash-guard"


def _bash_guard_command() -> str:
    """The guard's absolute path, beside the interpreter running dsagt.

    Claude Code spawns a hook from a shell that has no dsagt environment
    active under pipx or ``uv tool install``, where ``uv run`` finds nothing
    and a hook that fails to start lets the call through.
    """
    return str(Path(sys.executable).parent / _BASH_GUARD_SCRIPT)


def _write_bash_guard_hook(working_dir: Path) -> list[str]:
    """Write the bash guard into the project's Claude Code settings.

    ``.claude/settings.json`` is shared with the user's own settings, so the
    file is read and only the dsagt hook entry is set: an entry whose
    command names the guard is replaced (a reinstall moves the script), and
    a user's other hooks are kept.  The guard refuses a bare ``python`` call
    from the Bash tool with the recorded form to use (``dsagt-bash-guard``).
    """
    settings_path = working_dir / ".claude" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text() or "{}")
    hooks = settings.setdefault("hooks", {})
    entry = {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": _bash_guard_command()}],
    }
    kept = [
        e
        for e in hooks.get("PreToolUse", [])
        if not any(
            _BASH_GUARD_SCRIPT in h.get("command", "") for h in e.get("hooks", [])
        )
    ]
    before = json.dumps(hooks.get("PreToolUse", []), sort_keys=True)
    hooks["PreToolUse"] = [*kept, entry]
    if json.dumps(hooks["PreToolUse"], sort_keys=True) == before:
        return []
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return [f"Wrote the bash guard hook into {settings_path}"]


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
        actions.extend(_write_bash_guard_hook(working_dir))

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
