"""
Cline agent setup.

Install: ``npm i -g cline``.
Generates: ``.clinerules/dsagt_instructions.md``;
``cline auth`` + ``cline mcp add`` run per-init in
:meth:`ClineSetup.write_dynamic` and write to ``$CLINE_DIR/data/``.

**BYOA Phase 1 status: PUNTED for batch / smoke-test.** Cline's bundled
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

Phase 2 reactivates batch mode: ``dsagt-proxy`` aliases the cline-
substituted name back to the upstream's served name (same trick old_code
used for roo).  Until then, ``cline.run_script`` raises ``RuntimeError``
and the smoke test short-circuits in ``tests/smoke_test/run.sh``.

Interactive use is unaffected — ``dsagt init --agent cline`` still
writes the project state, and users running cline via VS Code (where
the UI exposes ``awsBedrockEndpoint`` etc.) can drive the project
manually.

OTel support: **none** (verified, ``otel_payload_support = "none"``).
Cline ships ``@opentelemetry/*`` packages but installs only a
``MeterProvider`` + ``LoggerProvider`` — never a ``TracerProvider``;
zero spans are ever created (``OpenTelemetryClientProvider.ts``).  Its
``captureConversationTurnEvent`` records only ``ulid``, ``provider``,
``model``, ``source``, token counts — no messages, no tool calls.
Activation also uses non-standard ``CLINE_OTEL_*``-prefixed env vars
that ignore our standard ``OTEL_EXPORTER_OTLP_ENDPOINT``.  Memory
extraction will see no agent-conversation traces; tool execution and KB
observability still work via dsagt-run / MCP-server spans.

Workaround: ``dsagt start --enable-proxy`` makes every cline LLM call
visible in MLflow — each agent turn becomes an inspectable trace
(messages + assistant response + tool_use blocks) you can audit live or
replay.  The dsagt-proxy interposes on cline's LLM calls, forwards them
to the user's upstream, and emits an OTel trace on cline's behalf.
Memory extraction is a downstream consequence — but the primary value
of the flag is being able to see what cline is doing at all.  Adds one
subprocess per project session.  See ``commands/proxy_server.py``.

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
    _run_simple_script,
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
    otel_payload_support = "none"
    # Cline's CLI nominally supports openai-native + anthropic, but cline
    # auth's ``-b/--baseurl`` flag is openai-only and the openai-native
    # path needs a non-standard model env var.  Anthropic is the only
    # path with consistent env conventions: ``ANTHROPIC_BASE_URL`` is read
    # by cline's anthropic SDK at runtime, covering api.anthropic.com and
    # gateway endpoints alike.  Match roo's policy.
    credential_env_vars = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
    )
    # Cline emits no OTel — agent-side telemetry only via --proxy_traces.
    telemetry_env = {}
    credential_hints = (
        (
            "ANTHROPIC_API_KEY",
            "your provider API key (works for openai-shape "
            "gateways too — cline's anthropic SDK reaches them via "
            "ANTHROPIC_BASE_URL)",
        ),
        (
            "ANTHROPIC_BASE_URL",
            "gateway / proxy URL "
            "(cline auth's -b flag is openai-only; the anthropic SDK reads "
            "this env var at runtime)",
        ),
        (
            "ANTHROPIC_MODEL",
            "model name your gateway serves "
            "(e.g. claude-haiku-4-5-20251001-v1-project)",
        ),
    )

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
        """Three side-effects, all idempotent:

        1. ``cline auth`` — populate per-project auth state from
           ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_MODEL`` in the user's
           shell.  Cline's per-project state dir is empty until ``cline
           auth`` writes to it; the user's global cline auth doesn't
           propagate into a fresh ``--config <project>/.cline-data``.
        2. ``cline mcp add`` per server (skipped if entry already exists
           — cline's mcp subcommand has no remove, only add).
        3. Patch the JSON cline wrote with our env block (cline doesn't
           inherit parent env into MCP children, so MLFLOW_TRACKING_URI,
           DSAGT_PROJECT_DIR, EMBEDDING_* must live in the JSON).

        Requires the ``cline`` binary to be installed at init time.
        """
        del pdir
        actions: list[str] = []
        cline_dir = str(working_dir / ".cline-data")
        Path(cline_dir).mkdir(parents=True, exist_ok=True)

        # 1. Configure per-project cline auth from shell env.
        api_key = env.get("ANTHROPIC_API_KEY")
        model = env.get("ANTHROPIC_MODEL")
        if not api_key or not model:
            raise RuntimeError(
                "cline batch mode requires ANTHROPIC_API_KEY and "
                "ANTHROPIC_MODEL in the shell env (and ANTHROPIC_BASE_URL "
                "if you're on a gateway).  Cline's anthropic SDK reads "
                "ANTHROPIC_BASE_URL at runtime, so this single config "
                "covers api.anthropic.com and lab gateways alike.  For "
                "openai-shape gateways like PNNL, alias "
                "ANTHROPIC_API_KEY=$OPENAI_API_KEY — most lab gateways "
                "serve both wire protocols on the same key."
            )
        auth_cmd = [
            "cline",
            "auth",
            "--config",
            cline_dir,
            "-p",
            "anthropic",
            "-k",
            api_key,
            "-m",
            model,
        ]
        result = subprocess.run(
            auth_cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"cline auth failed (exit {result.returncode}): {detail}"
            )
        actions.append(f"Configured cline auth at {cline_dir}")

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
            add_cmd = [
                "cline",
                "mcp",
                "add",
                "--config",
                cline_dir,
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

        # ``launch_oneliner`` (used by ``dsagt init`` printout) tells
        # the user to run ``cline -v -y -a --config <pdir>/.cline-data``
        # — the per-project state dir we just populated above.
        return actions

    # --- Phase 2 proxy-mode write_dynamic --------------------------------
    # Differs from BYOA only in the auth call: ``cline auth -p openai -b
    # http://localhost:<proxy_port> -k <sentinel> -m <model>`` instead of
    # ``-p anthropic``.  The MCP-add + JSON-patch steps are identical, so
    # we extract them into a helper.

    def _patch_mcp_servers(
        self,
        config: dict,
        working_dir: Path,
        cline_dir: str,
    ) -> str:
        """Idempotent ``cline mcp add`` + JSON env-block patch.
        Returns the action string for the printout.
        """
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
            add_cmd = [
                "cline",
                "mcp",
                "add",
                "--config",
                cline_dir,
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
        return (
            f"Registered MCP servers and patched {len(mcp_env)} "
            f"env vars into {mcp_path}"
        )

    def proxy_write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        """Phase-2 proxy-mode setup.  Cline auths against the
        localhost dsagt-proxy with the sentinel API key (real upstream
        creds live only in the proxy subprocess).  Provider is forced
        to ``openai`` because cline auth's ``-b/--baseurl`` flag is
        openai-only — the proxy speaks /chat/completions on every
        upstream regardless of what the agent emits, so this is fine.
        """
        del pdir
        from .base import _PROXY_FORWARDED_SENTINEL

        actions: list[str] = []
        cline_dir = str(working_dir / ".cline-data")
        Path(cline_dir).mkdir(parents=True, exist_ok=True)

        proxy_port = (config.get("proxy") or {}).get("port")
        model = (config.get("llm") or {}).get("model")
        if not proxy_port or not model:
            raise RuntimeError(
                "cline proxy-mode write_dynamic requires "
                "config['proxy']['port'] and config['llm']['model']."
            )
        auth_cmd = [
            "cline",
            "auth",
            "--config",
            cline_dir,
            "-p",
            "openai",
            "-k",
            _PROXY_FORWARDED_SENTINEL,
            "-m",
            model,
            "-b",
            f"http://localhost:{proxy_port}",
        ]
        result = subprocess.run(
            auth_cmd,
            env=env,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"cline auth (proxy) failed (exit {result.returncode}): {detail}"
            )
        actions.append(f"Configured cline auth at {cline_dir} (proxy mode)")
        actions.append(self._patch_mcp_servers(config, working_dir, cline_dir))
        return actions

    def runtime_env(self, config: dict) -> dict[str, str]:
        """BYOA infrastructure: per-project ``CLINE_DIR`` (state dir)
        plus inherited telemetry flags.  Isolates per-project MCP /
        auth state from the global ``~/.cline-data``.
        """
        env = super().runtime_env(config)
        env["CLINE_DIR"] = str(Path(config["project_dir"]) / ".cline-data")
        return env

    def launch_oneliner(self, project: str, project_dir: Path) -> str:
        """Cline needs ``--config <pdir>/.cline-data`` so it picks up
        the MCP-server registrations + auth state ``write_dynamic``
        wrote.  Also ``-v -y -a`` for verbose / auto-yes / act-mode.
        """
        del project
        import shlex

        pdir = shlex.quote(str(project_dir))
        cline_dir = shlex.quote(str(project_dir / ".cline-data"))
        return f"cd {pdir} && cline -v -y -a --config {cline_dir}"

    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Punted in BYOA Phase 1 — see module docstring.

        Cline's bundled anthropic provider hardcodes a model-ID whitelist
        that doesn't include lab-gateway aliases (``-v1-project`` etc.),
        so the request silently substitutes a default cline thinks it
        knows and the gateway 401s.  Phase 2 ``dsagt-proxy`` fixes this
        by aliasing names back to the upstream's served IDs.
        """
        del config, env, working_dir, script_path, max_turns
        raise RuntimeError(
            "cline batch mode is punted in Phase 1 BYOA — cline's anthropic "
            "provider rewrites unrecognized model names (lab-gateway "
            "aliases like ``-v1-project``) to its hardcoded default, "
            "which the gateway then rejects.  Phase 2 reactivates this "
            "via dsagt-proxy aliasing.  See agents/cline.py module "
            "docstring."
        )

    def proxy_run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Phase-2 proxy-mode batch run.  Un-punts cline:
        ``proxy_write_dynamic`` already configured cline auth against
        the localhost proxy + registered our MCP servers, so we just
        invoke cline pointing at the per-project state dir.  The proxy
        translates lab-gateway-aliased model names back to upstream-
        served names, sidestepping cline's hardcoded whitelist.
        """
        del max_turns
        text = script_path.read_text().strip()
        if not text:
            return 1
        cline_dir = env.get("CLINE_DIR") or str(
            Path(config["project_dir"]) / ".cline-data"
        )
        # ``-a`` (act mode) — without it cline defaults to its task UI
        # mode whose model fallback is ``gpt-5.5`` regardless of what
        # ``cline auth`` configured for plan/act modes.  Model comes
        # from the per-project auth state ``proxy_write_dynamic`` populated.
        cmd = ["cline", "-v", "-y", "-a", "--config", cline_dir, text]
        return _run_simple_script(cmd, env, working_dir, self.install_hint)
