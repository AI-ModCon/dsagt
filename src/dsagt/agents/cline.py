"""
Cline agent setup.

Install: ``npm i -g cline``.
Generates: ``.clinerules/dsagt_instructions.md``, ``.dsagt_env``;
``cline auth`` and ``cline mcp add`` are run per-launch in
:meth:`ClineSetup.write_dynamic` and write to ``$CLINE_DIR/data/``.
Proxy routing: ``cline auth -p openai -b http://localhost:<proxy_port>``
(Cline ignores ``ANTHROPIC_BASE_URL`` and only allows ``--baseurl`` on the
openai provider, so cline talks to our proxy via OpenAI-format
``/chat/completions`` — the same path goose uses).

MCP config: hand-writing ``cline_mcp_settings.json`` is silently ignored;
the only path cline loads is via ``cline mcp add``, which writes a
stripped schema (no ``env`` block).  Per-server env has to come from
process-env inheritance — ``DSAGT_PROJECT_DIR`` / ``LLM_API_KEY`` /
``OPENAI_BASE_URL`` / ``EMBEDDING_MODEL`` are set in :func:`agent_env`
and flow cline → MCP-server-subprocess.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .base import (
    AgentSetup,
    _PROXY_FORWARDED_SENTINEL,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_subprocess_env,
    _run_simple_script,
    _write_env_file,
)

logger = logging.getLogger(__name__)


class ClineSetup(AgentSetup):
    name = "cline"
    base_command = ["cline"]
    static_marker = ".clinerules/dsagt_instructions.md"
    install_hint = "Install with `npm i -g cline`."

    def write_static(self, working_dir: Path) -> list[str]:
        actions: list[str] = []
        (working_dir / ".cline-data").mkdir(parents=True, exist_ok=True)
        instructions = _load_master_instructions()
        if instructions:
            rules_dir = working_dir / ".clinerules"
            rules_dir.mkdir(exist_ok=True)
            action = _append_or_write(
                rules_dir / "dsagt_instructions.md",
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
        proxy_port: int,
    ) -> list[str]:
        """Three side-effects, all of which need the launch-time env dict:

        1. ``cline auth -p openai -b URL`` writes provider config (proxy
           URL + sentinel key + model) into ``$CLINE_DIR/globalState.json``.
           Cline ignores ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` env
           vars; the auth subcommand is the only path.  Provider is
           ``openai`` because ``cline auth -p anthropic -b URL`` errors
           out: "Base URL is only supported for OpenAI and OpenAI-
           compatible providers".  Cline-as-openai sends
           ``/chat/completions`` to our proxy — same path goose uses;
           proxy translates upstream regardless.

        2. ``cline mcp add`` per server registers the dsagt MCP servers
           into cline's loaded state.  Hand-writing
           ``cline_mcp_settings.json`` is silently ignored — only the
           subcommand path works.

        3. Patch the JSON cline wrote with our env block.  Cline doesn't
           inherit parent process env into MCP server subprocesses (unlike
           claude and goose), so the dsagt servers' env vars
           (``MLFLOW_TRACKING_URI``, ``EMBEDDING_*``, ``DSAGT_SESSION_ID``,
           etc.) must live in the JSON.  An earlier attempt added
           ``disabled`` and ``alwaysAllow`` keys here too and cline
           silently rejected the whole config — adding only ``env`` keeps
           the schema close enough to what ``cline mcp add`` wrote that
           the rest of the entry still parses.

        Plus the ``.dsagt_env`` shell file (used by manual ``source``
        workflows).
        """
        del proxy_port  # used via env["CLINE_DIR"] / auth_cmd build
        actions: list[str] = []
        cline_dir = env.get("CLINE_DIR") or str(working_dir / ".cline-data")
        Path(cline_dir).mkdir(parents=True, exist_ok=True)
        sentinel = _PROXY_FORWARDED_SENTINEL
        model = config["llm"]["model"]
        proxy_port = config["proxy"]["port"]

        auth_cmd = [
            "cline", "auth",
            "-p", "openai",
            "-k", sentinel,
            "-m", model,
            "-b", f"http://localhost:{proxy_port}",
        ]
        logger.info("Running: cline auth (CLINE_DIR=%s)", cline_dir)
        try:
            result = subprocess.run(
                auth_cmd, env=env, cwd=str(working_dir),
                capture_output=True, text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "cline command not found. Install with `npm i -g cline`."
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"cline auth failed (exit {result.returncode}): {detail}"
            )
        actions.append(f"Configured cline auth at {cline_dir}")

        for server in ("registry", "knowledge"):
            add_cmd = [
                "cline", "mcp", "add",
                "--config", cline_dir,
                f"dsagt-{server}",
                "--",
                "uv", "run", f"dsagt-{server}-server",
            ]
            result = subprocess.run(
                add_cmd, env=env, cwd=str(working_dir),
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"cline mcp add dsagt-{server} failed "
                    f"(exit {result.returncode}): {detail}"
                )

        mcp_path = Path(cline_dir) / "data" / "settings" / "cline_mcp_settings.json"
        if not mcp_path.exists():
            raise RuntimeError(
                f"cline mcp add succeeded but {mcp_path} was not created — "
                "cline may have changed its config layout."
            )
        settings = json.loads(mcp_path.read_text())
        mcp_env = _mcp_subprocess_env(env)
        for entry in settings.get("mcpServers", {}).values():
            entry["env"] = mcp_env
        mcp_path.write_text(json.dumps(settings, indent=2) + "\n")
        actions.append(
            f"Registered MCP servers and patched {len(mcp_env)} env vars into {mcp_path}"
        )

        env_path = working_dir / ".dsagt_env"
        _write_env_file(env_path, {
            "CLINE_DIR": str(cline_dir),
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
        })
        actions.append(f"Wrote {env_path}")
        return actions

    def env_overrides(self, config: dict, proxy_port: int) -> dict[str, str]:
        # Cline ignores ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY.  Provider
        # config is stored in CLINE_DIR/globalState.json (written by
        # `cline auth`, run from write_dynamic).  CLINE_DIR is project-
        # scoped so MCP config + auth stay isolated per project.
        del proxy_port
        return {
            "CLINE_DIR": str(Path(config["project_dir"]) / ".cline-data"),
        }

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Single ``cline -y`` call with the entire script as one prompt.

        Cline's ``--continue`` resumes a task but rejects a new prompt
        ("Use --continue without a prompt"), so we can't loop discrete
        turns the way claude does.  Instead, hand cline the whole script
        in one task — same shape as goose's ``--instructions FILE``.
        Multi-turn behavior happens inside cline's autonomous tool-call
        loop, not across processes.

        ``-v`` shows model reasoning + tool calls inline; without it,
        cline emits only the task id and final summary, which is opaque
        during smoke tests and makes MCP/tool-call failures invisible.
        """
        del config, max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        cmd = ["cline", "-v", "-y", text]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
