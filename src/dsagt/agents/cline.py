"""
Cline agent setup.

Install: ``npm i -g cline``.
Generates: ``.clinerules/dsagt_instructions.md`` and
``.cline-data/cline_mcp_settings.json`` (per-project MCP config, loaded
via the ``CLINE_MCP_SETTINGS_PATH`` env var at ``dsagt start``).
Provider auth is cline's own (subscription login, ``cline auth``, or the
VS Code extension), configured by the user before dsagt; dsagt never
writes cline auth state, since doing so can overwrite an existing
provider integration.

Batch and smoke-test runs are unsupported, verified against cline 3.0.34.
Batch itself runs (``cline "prompt"`` runs act-mode with auto-approve,
honoring the user's pre-configured auth), and headless cline spawns
registered MCP servers (cwd is the session dir, with the full shell env,
verified with a spawn probe).  It never bridges their tools into the
model's toolset: a session asked to enumerate every tool it has lists
only cline built-ins and team tools, zero MCP entries.  With no dsagt
tools reachable there is nothing for a smoke run to exercise, so
``cline.run_script`` raises ``RuntimeError`` and the smoke test
short-circuits in ``tests/smoke_test/run.sh``.  Re-probe on cline
upgrades (the TUI and VS Code paths do bridge MCP): once a headless
session can call an MCP tool, batch support is a small run_script.

Auth and config scoping, verified 3.0.34: cline stores provider
credentials in ``providers.json`` beside its MCP settings, and both
follow a ``--config`` / ``CLINE_DIR`` redirect, so whole-directory
isolation loses a subscription/OAuth login (immediate Unauthorized).
``CLINE_MCP_SETTINGS_PATH`` relocates only the MCP settings file
(``providers.json`` resolves independently via its own
``CLINE_PROVIDER_SETTINGS_PATH`` / data-dir default).  dsagt sets it in
:meth:`ClineSetup.runtime_env`, which gives per-project dsagt MCP config
with the user's global auth and leaves the global settings unchanged
(verified: per-project server spawns, subscription answered, global
mcpServers unchanged).

Interactive use is unaffected: ``dsagt init --agent cline`` writes the
project state, and users running cline via VS Code can drive the project
manually.

OTel support: none (verified).  Cline includes ``@opentelemetry/*``
packages but installs only a ``MeterProvider`` and ``LoggerProvider``,
never a ``TracerProvider``; zero spans are ever created
(``OpenTelemetryClientProvider.ts``).  Its
``captureConversationTurnEvent`` records only ``ulid``, ``provider``,
``model``, ``source``, and token counts, with no messages and no tool
calls.  Activation also uses ``CLINE_OTEL_*``-prefixed env vars and
ignores the standard ``OTEL_EXPORTER_OTLP_ENDPOINT``.  Memory extraction
receives no agent-conversation traces; tool execution and KB
observability still work via dsagt-run and MCP-server spans.

MCP config: cline 3.x loads whatever file ``CLINE_MCP_SETTINGS_PATH``
names, including a file written by dsagt, with the schema
``{"mcpServers": {<name>: {"transport": {type, command, args, env}}}}``.
The env block is written to ``transport.env`` so dsagt MCP-server
children get ``MLFLOW_TRACKING_URI``, ``DSAGT_PROJECT_DIR``,
``EMBEDDING_*``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import (
    AgentSetup,
    _write_dsagt_block,
    _load_master_instructions,
    _mcp_env_block,
)

logger = logging.getLogger(__name__)


class ClineSetup(AgentSetup):
    name = "cline"
    base_command = ["cline"]
    static_marker = ".clinerules/dsagt_instructions.md"
    # Cline skills are opt-in (Settings > Features > Enable Skills); mirroring
    # is harmless if unused, and search_skills covers the disabled case.
    native_skills_dir = ".cline/skills"
    install_hint = "Install with `npm i -g cline`."
    # Cline owns its provider auth (subscription login, `cline auth`, or the
    # VS Code extension's settings): the user configures cline before
    # pointing dsagt at it, and dsagt never writes auth state, because
    # running `cline auth` from env vars can overwrite an existing provider
    # integration (e.g. an OpenAI subscription login).

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [
            working_dir / ".clinerules",
            working_dir / ".cline-data",
            working_dir / ".cline",
        ]

    def write_static(self, working_dir: Path, *, auto_assess: bool = True) -> list[str]:
        actions: list[str] = []
        (working_dir / ".cline-data").mkdir(parents=True, exist_ok=True)
        instructions = _load_master_instructions(auto_assess)
        if instructions:
            rules_dir = working_dir / ".clinerules"
            rules_dir.mkdir(exist_ok=True)
            action = _write_dsagt_block(
                rules_dir / "dsagt_instructions.md", instructions
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
        """Write ``<project>/.cline-data/cline_mcp_settings.json``.

        Cline 3.x resolves its MCP settings file via the
        ``CLINE_MCP_SETTINGS_PATH`` env var (set by :meth:`runtime_env`),
        independent of ``providers.json`` (auth), so a per-project file
        gives project-scoped dsagt MCP config while the user's global auth
        and global settings stay unchanged (verified 3.0.34: per-project
        server spawns, subscription auth honored, global mcpServers list
        unchanged).  The env block is written to ``transport.env`` per
        cline's ``McpStdioTransportConfig`` schema.

        Provider auth is cline's own (subscription login, ``cline auth``, or
        the VS Code extension) and must be configured before using dsagt;
        dsagt never writes cline auth state (see the class comment).

        Idempotent; preserves any non-dsagt entries the user added to the
        per-project file.  Init runs without a cline binary.
        """
        del env, pdir
        settings_path = working_dir / ".cline-data" / "cline_mcp_settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        settings: dict = {"mcpServers": {}}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
                settings.setdefault("mcpServers", {})
            except (json.JSONDecodeError, OSError):
                settings = {"mcpServers": {}}

        mcp_env = _mcp_env_block(config)
        transport: dict = {
            "type": "stdio",
            "command": "uv",
            "args": ["run", "dsagt-server"],
        }
        if mcp_env:
            transport["env"] = mcp_env
        settings["mcpServers"]["dsagt"] = {"transport": transport}
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        return [
            f"Wrote {settings_path} ({len(mcp_env)} MCP env vars; loaded via "
            "CLINE_MCP_SETTINGS_PATH at dsagt start)"
        ]

    def runtime_env(self, config: dict) -> dict[str, str]:
        """Point cline at the per-project MCP settings.

        ``CLINE_MCP_SETTINGS_PATH`` relocates only the MCP settings file;
        ``providers.json`` (auth) resolves independently, so the user's
        subscription/OAuth login keeps working and the global settings
        never receive a dsagt entry.
        """
        env = super().runtime_env(config)
        env["CLINE_MCP_SETTINGS_PATH"] = str(
            Path(config["project_dir"]) / ".cline-data" / "cline_mcp_settings.json"
        )
        return env

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Raise: cline's headless CLI loads no MCP servers (module docstring).

        Cline 3.x batch runs (act mode, pre-configured auth), but a
        scripted session has no dsagt tools to exercise.
        """
        del config, env, working_dir, script_path, max_turns
        raise RuntimeError(
            "cline batch mode is not supported: cline's headless CLI "
            "(verified 3.0.34) does not load MCP servers, so a scripted "
            "session has no dsagt tools.  See agents/cline.py module "
            "docstring; re-probe on cline upgrades."
        )
