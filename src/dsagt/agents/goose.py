"""
Goose agent setup.

Install: see https://github.com/block/goose.
Generates: ``goose.yaml``, ``.goosehints``.

BYOA: Goose talks directly to the user's provider via its own
``~/.config/goose/config.yaml`` or ``GOOSE_PROVIDER`` / ``GOOSE_MODEL`` env;
DSAGT sets no telemetry env.

Telemetry / episodic memory: goose's turns come from its on-disk session
database through the periodic pass (``traces.GooseReader``), the same way as
every agent's, so traces and episodic memory work for goose without any hook
in goose itself.  The core capabilities (KB retrieval, registered tools,
skills, tool-execution provenance via ``dsagt-run``) are agent-agnostic.  If goose gains a hook
mechanism the options can be enabled.

Gateway note: goose's openai/anthropic providers read ``OPENAI_HOST`` /
``ANTHROPIC_HOST`` for the base URL (a goose-specific naming convention from
its Rust client), NOT the standard ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``
everything else uses.  Without HOST set, goose ignores BASE_URL and hits the
provider's default endpoint — silently, for users on a lab gateway.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import (
    AgentSetup,
    _write_dsagt_block,
    _load_master_instructions,
    _mcp_server_args,
    _run_simple_script,
)


class GooseSetup(AgentSetup):
    name = "goose"
    base_command = ["goose", "session"]
    static_marker = ".goosehints"
    native_skills_dir = ".agents/skills"  # cross-agent standard goose discovers
    install_hint = "See https://github.com/block/goose for installation."

    def write_static(self, working_dir: Path, *, auto_assess: bool = True) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions(auto_assess)
        if instructions:
            action = _write_dsagt_block(working_dir / ".goosehints", instructions)
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
        """Write ``goose.yaml``.  Goose inherits parent env into MCP children,
        so extension entries don't need an explicit env list."""
        del config, env, pdir
        actions: list[str] = []

        args = _mcp_server_args()
        goose_config: dict = {
            "extensions": {
                "dsagt": {
                    "enabled": True,
                    "name": "dsagt",
                    "type": "stdio",
                    "cmd": "uv " + " ".join(args),
                    "timeout": 300,
                }
            }
        }

        goose_path = working_dir / "goose.yaml"
        goose_path.write_text(
            yaml.dump(goose_config, default_flow_style=False, sort_keys=False)
        )
        actions.append(f"Wrote {goose_path}")
        return actions

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        # Base default is just the static marker (.goosehints), but
        # write_dynamic also writes goose.yaml — list both so switching
        # agents removes the MCP-extension config too.
        return [
            working_dir / ".goosehints",
            working_dir / "goose.yaml",
        ]

    def interactive_command(self, config: dict) -> list[str]:
        """Goose reads ``~/.config/goose/config.yaml`` for extensions, not a
        project-local file — so the dsagt MCP server is passed via
        ``--with-extension`` on the session command to attach for this project.
        """
        del config
        cmd = list(self.base_command)
        cmd.extend(["--with-extension", "uv run dsagt-server"])
        return cmd

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``goose run`` call — goose's instructions file IS multi-turn."""
        del config
        env["GOOSE_MODE"] = "auto"
        cmd = [
            "goose",
            "run",
            "--instructions",
            str(script_path),
            "--max-turns",
            str(max_turns),
        ]
        cmd.extend(["--with-extension", "uv run dsagt-server"])
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
