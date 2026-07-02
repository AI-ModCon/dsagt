"""
Goose agent setup.

Install: see https://github.com/block/goose.
Generates: ``goose.yaml``, ``.goosehints``.

BYOA: Goose talks directly to the user's provider via its own
``~/.config/goose/config.yaml`` or ``GOOSE_PROVIDER`` / ``GOOSE_MODEL`` env;
DSAGT sets no telemetry env.

Telemetry / episodic memory: Goose has **no Stop/turn hook and no MLflow
autolog integration**, so DSAGT does not offer the mlflow-autolog or
episodic-memory options for goose — those are gated on agents that have both
(claude / codex / opencode).  Goose stays fully supported for the core,
agent-agnostic capabilities (KB retrieval, registered tools, skills,
tool-execution provenance via ``dsagt-run``).  If goose gains a hook
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
    _append_or_write,
    _DSAGT_MARKER,
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
    # Goose's openai/anthropic providers read ``OPENAI_HOST`` / ``ANTHROPIC_HOST``
    # for the base URL (goose-specific naming), plus its own GOOSE_PROVIDER /
    # GOOSE_MODEL routing selectors.
    credential_env_vars = (
        "GOOSE_PROVIDER",
        "GOOSE_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_HOST",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_HOST",
    )
    credential_hints = (
        (
            "GOOSE_PROVIDER",
            "anthropic, openai, etc. (skip if global ~/.config/goose configured)",
        ),
        ("GOOSE_MODEL", "the model name your provider serves"),
        ("ANTHROPIC_API_KEY", "if GOOSE_PROVIDER=anthropic"),
        ("ANTHROPIC_HOST", "if GOOSE_PROVIDER=anthropic and on a gateway"),
        ("OPENAI_API_KEY", "if GOOSE_PROVIDER=openai"),
        (
            "OPENAI_HOST",
            "if GOOSE_PROVIDER=openai and on a gateway "
            "(NOT OPENAI_BASE_URL — goose ignores that)",
        ),
    )

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / ".goosehints",
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
