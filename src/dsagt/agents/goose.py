"""
Goose agent setup.

Install: see https://github.com/block/goose.
Generates: ``goose.yaml``, ``.goosehints``, ``.dsagt_env``.
Proxy routing: ``OPENAI_HOST``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import (
    AgentSetup,
    _PROXY_FORWARDED_SENTINEL,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_server_args,
    _run_simple_script,
    _write_env_file,
)


class GooseSetup(AgentSetup):
    name = "goose"
    base_command = ["goose", "session"]
    static_marker = ".goosehints"
    install_hint = "See https://github.com/block/goose for installation."

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        instructions = _load_master_instructions()
        if instructions:
            action = _append_or_write(
                working_dir / ".goosehints", instructions, _DSAGT_MARKER,
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
        """Goose inherits parent env into MCP children, so the per-extension
        block doesn't need an explicit env list — the proxy URL flows in via
        ``OPENAI_HOST`` from the agent's environment.
        """
        del env
        actions: list[str] = []
        model = config["llm"]["model"]

        goose_config = {
            "GOOSE_PROVIDER": "openai",
            "GOOSE_MODEL": model,
            "extensions": {},
        }
        for server in ("registry", "knowledge"):
            args = _mcp_server_args(server)
            goose_config["extensions"][server] = {
                "enabled": True,
                "name": server,
                "type": "stdio",
                "cmd": "uv " + " ".join(args),
                "timeout": 300,
            }

        goose_path = working_dir / "goose.yaml"
        goose_path.write_text(yaml.dump(goose_config, default_flow_style=False, sort_keys=False))
        actions.append(f"Wrote {goose_path}")

        env_path = working_dir / ".dsagt_env"
        _write_env_file(env_path, {
            "OPENAI_HOST": f"http://localhost:{proxy_port}",
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
        })
        actions.append(f"Wrote {env_path}")
        return actions

    def env_overrides(self, config: dict, proxy_port: int) -> dict[str, str]:
        # Override any global ~/.config/goose/config.yaml so the project's
        # configured model is what actually runs.  Without these, a user's
        # global GOOSE_MODEL (e.g. a model their upstream doesn't offer)
        # silently wins over dsagt_config.yaml.
        return {
            "OPENAI_HOST": f"http://localhost:{proxy_port}",
            "GOOSE_PROVIDER": "openai",
            "GOOSE_MODEL": config["llm"]["model"],
            "OPENAI_API_KEY": _PROXY_FORWARDED_SENTINEL,
        }

    def interactive_command(self, config: dict) -> list[str]:
        """Goose only reads ``~/.config/goose/config.yaml`` for extensions, not
        a project-local file — so the MCP servers are passed via
        ``--with-extension`` flags on the session command to guarantee they
        attach for this project.
        """
        del config
        cmd = list(self.base_command)
        for server in ("registry", "knowledge"):
            cmd.extend(["--with-extension", f"uv run dsagt-{server}-server"])
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
        cmd = ["goose", "run", "--instructions", str(script_path),
               "--max-turns", str(max_turns)]
        for server in ("registry", "knowledge"):
            cmd.extend(["--with-extension", f"uv run dsagt-{server}-server"])
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
