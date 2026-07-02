"""
Cline agent setup.

Install: ``npm i -g cline``.
Generates: ``.clinerules/dsagt_instructions.md``;
``cline mcp add`` runs per-init in :meth:`ClineSetup.write_dynamic` and
writes to ``$CLINE_DIR/data/``.  Provider auth is cline's own
(subscription login / ``cline auth`` / the VS Code extension), configured
by the user before dsagt — dsagt never writes cline auth state, since
doing so can clobber an existing provider integration.

**Batch / smoke-test status: NOT SUPPORTED.** Cline's bundled
``lib.mjs`` ships a hardcoded anthropic-provider model whitelist
(``claude-haiku-4-5-20251001``, ``claude-sonnet-4-5-20250929``, etc. —
all without the ``-v1-project`` suffix lab gateways like PNNL require).
Even when ``cline auth -m claude-haiku-4-5-20251001-v1-project`` stores
our model name correctly in ``globalState.json``, at request time
cline's anthropic provider validates against the whitelist, sees the
unrecognized suffixed name, and silently substitutes its hardcoded
default (``claude-sonnet-4-5-20250929``) — which the gateway then 401s
because PNNL only serves the suffixed variant.

The openai-native path has the same problem (whitelisted ``gpt-5.5``,
``gpt-4o`` etc.; PNNL serves ``-project``-suffixed variants).  Bedrock
is locked behind ``cline auth``'s interactive setup flow, not the CLI.
So ``cline.run_script`` raises ``RuntimeError`` and the smoke test
short-circuits in ``tests/smoke_test/run.sh``.

Interactive use is unaffected — ``dsagt init --agent cline`` still
writes the project state, and users running cline via VS Code (where
the UI exposes ``awsBedrockEndpoint`` etc.) can drive the project
manually.

OTel support: **none** (verified).
Cline ships ``@opentelemetry/*`` packages but installs only a
``MeterProvider`` + ``LoggerProvider`` — never a ``TracerProvider``;
zero spans are ever created (``OpenTelemetryClientProvider.ts``).  Its
``captureConversationTurnEvent`` records only ``ulid``, ``provider``,
``model``, ``source``, token counts — no messages, no tool calls.
Activation also uses non-standard ``CLINE_OTEL_*``-prefixed env vars
that ignore our standard ``OTEL_EXPORTER_OTLP_ENDPOINT``.  Memory
extraction will see no agent-conversation traces; tool execution and KB
observability still work via dsagt-run / MCP-server spans.

MCP config: hand-writing ``cline_mcp_settings.json`` is silently ignored;
the only path cline loads is via ``cline mcp add``, which writes a
stripped schema (no ``env`` block).  We patch in the env block ourselves
after ``cline mcp add`` runs so the dsagt MCP-server children inherit
``MLFLOW_TRACKING_URI``, ``EMBEDDING_*``, ``OTEL_*`` etc.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .base import (
    AgentSetup,
    _append_or_write,
    _DSAGT_MARKER,
    _load_master_instructions,
    _mcp_env_block,
)

logger = logging.getLogger(__name__)


class ClineSetup(AgentSetup):
    name = "cline"
    base_command = ["cline"]
    static_marker = ".clinerules/dsagt_instructions.md"
    # Cline skills are opt-in (Settings → Features → Enable Skills); mirroring
    # is harmless if unused, and search_skills covers the disabled case.
    native_skills_dir = ".cline/skills"
    install_hint = "Install with `npm i -g cline`."
    # Cline owns its provider auth (subscription login, `cline auth`, or the
    # VS Code extension's settings) — BYOA means the user configures cline
    # BEFORE pointing dsagt at it, and dsagt never touches auth state:
    # running `cline auth` from scavenged env vars can clobber an existing
    # provider integration (e.g. an OpenAI subscription login).
    credential_env_vars = ()
    credential_hints = ()

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        return [
            working_dir / ".clinerules",
            working_dir / ".cline-data",
            working_dir / ".cline",
        ]

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
    ) -> list[str]:
        """Two side-effects, both idempotent:

        1. ``cline mcp add`` per server (skipped if entry already exists
           — cline's mcp subcommand has no remove, only add).
        2. Patch the JSON cline wrote with our env block (cline doesn't
           inherit parent env into MCP children, so MLFLOW_TRACKING_URI,
           DSAGT_PROJECT_DIR, EMBEDDING_* must live in the JSON).

        Provider auth is cline's own (subscription login / ``cline auth`` /
        the VS Code extension) and must be configured before using dsagt —
        dsagt never writes cline auth state (see the class comment).

        Requires the ``cline`` binary to be installed at init time.
        """
        del env, pdir
        actions: list[str] = []
        cline_dir = str(working_dir / ".cline-data")
        Path(cline_dir).mkdir(parents=True, exist_ok=True)

        # Cline's mcp subcommand only has ``add`` (no ``remove``), and ``add``
        # errors if the server already exists.  Detect pre-existing entries
        # and skip the add so this method is idempotent (called by both
        # ``dsagt init`` and ``_cmd_start`` → ``dynamic_agent_record``).
        # The env-block patch below runs unconditionally to keep on-disk
        # routing in sync if mlflow port / paths ever changed.
        mcp_path = Path(cline_dir) / "data" / "settings" / "cline_mcp_settings.json"
        existing: set[str] = set()
        if mcp_path.exists():
            try:
                existing = set(
                    json.loads(mcp_path.read_text()).get("mcpServers", {}).keys()
                )
            except (json.JSONDecodeError, OSError):
                existing = set()

        if "dsagt" not in existing:
            # ``--config`` is a *global* option on cline ≥3.x — it must
            # precede the subcommand (``mcp add`` rejects it as unknown).
            # ``--yes`` skips the interactive add wizard cline 3.x opens.
            add_cmd = [
                "cline",
                "--config",
                cline_dir,
                "mcp",
                "add",
                "--yes",
                "dsagt",
                "--",
                "uv",
                "run",
                "dsagt-server",
            ]
            result = subprocess.run(
                add_cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"cline mcp add dsagt failed "
                    f"(exit {result.returncode}): {detail}"
                )

        mcp_path = Path(cline_dir) / "data" / "settings" / "cline_mcp_settings.json"
        if not mcp_path.exists():
            raise RuntimeError(
                f"cline mcp add succeeded but {mcp_path} was not created — "
                "cline may have changed its config layout."
            )
        settings = json.loads(mcp_path.read_text())
        mcp_env = _mcp_env_block(config)
        for entry in settings.get("mcpServers", {}).values():
            entry["env"] = mcp_env
        mcp_path.write_text(json.dumps(settings, indent=2) + "\n")
        actions.append(
            f"Registered MCP servers and patched {len(mcp_env)} env vars into {mcp_path}"
        )
        return actions

    def runtime_env(self, config: dict) -> dict[str, str]:
        """BYOA infrastructure: per-project ``CLINE_DIR`` (state dir).

        Isolates per-project MCP / auth state from the global
        ``~/.cline-data``.
        """
        env = super().runtime_env(config)
        env["CLINE_DIR"] = str(Path(config["project_dir"]) / ".cline-data")
        return env

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Not supported for cline — see module docstring.

        Cline's bundled anthropic provider hardcodes a model-ID whitelist
        that doesn't include lab-gateway aliases (``-v1-project`` etc.),
        so the request silently substitutes a default cline thinks it
        knows and the gateway 401s.
        """
        del config, env, working_dir, script_path, max_turns
        raise RuntimeError(
            "cline batch mode is not supported — cline's anthropic "
            "provider rewrites unrecognized model names (lab-gateway "
            "aliases like ``-v1-project``) to its hardcoded default, "
            "which the gateway then rejects.  See agents/cline.py module "
            "docstring."
        )
