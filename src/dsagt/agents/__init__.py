"""
Agent platform configuration and launch.

Generates platform-specific config files (MCP server entries, agent
instructions, env vars) from the single ``dsagt_config.yaml``.  Launches
the agent process in the foreground and blocks until it exits.

Post-proxy DSAGT: each agent talks directly to its own provider.  We
inject MLflow + OTel env vars so every agent's native LLM-call telemetry
lands in the project's MLflow store, but model selection / API keys /
provider base URLs are the user's responsibility.

Two orthogonal questions about each agent:

1. **Native visibility.**  Without ``--enable-proxy``, are the agent's
   LLM calls visible in the MLflow UI (audit, drill-in, search)?
2. **Native extraction.**  Without ``--enable-proxy``, does end-of-
   session memory extraction work?

Answer to #2 is uniformly **no for non-proxy paths**, on purpose.  Each
agent that emits OTel does so in its own shape (Claude Code: span
events; Goose: domain spans; LiteLLM autolog: ``mlflow.spanInputs`` /
``mlflow.spanOutputs``).  Memory extraction reads a single canonical
shape (the LiteLLM-autolog shape, emitted only by ``dsagt-proxy``) so
the parser stays small and we avoid per-agent maintenance forever.
Run with ``--enable-proxy`` to get extraction.

Native visibility (``otel_payload_support`` class var, per-agent):

  =========  ============  =============================================
  Agent      Support tier  Native MLflow visibility (no proxy)
  =========  ============  =============================================
  claude     full          yes — every turn lands as a trace with
                           messages, response, tool_use blocks
                           (gated by 4 ``CLAUDE_CODE_*`` /
                           ``OTEL_LOG_*`` flags we set)
  goose      full          yes — ``dispatch_tool_call`` span carries
                           tool + arguments JSON
  codex      partial       limited — Codex OTel spans carry only token
                           counts and tool names; full conversation
                           lives in ``$CODEX_HOME/sessions/rollout-*.jsonl``
                           (side-channel reader not yet wired)
  cline      none          no — Cline emits no OTel spans at all;
                           the agent is a black box from DSAgt's view
  roo        none          no — Roo Code imports zero OTel SDKs;
                           PostHog telemetry is payload-free
  =========  ============  =============================================

Pick the run mode by what you need:

* **Visibility only** (Claude Code / Goose): default ``dsagt start``
  is enough.  Real-time audit + drill-in works in the MLflow UI.
  Memory extraction will produce nothing.
* **Visibility + extraction** (any agent): ``dsagt start
  --enable-proxy``.  Adds one subprocess; routes every LLM call
  through it; lands traces in canonical shape that extraction reads.
* **Visibility for non-emitters** (Cline / Roo / Codex partial):
  ``dsagt start --enable-proxy`` is the only way to see the agent's
  LLM calls at all.  Extraction is a free downstream consequence.

Tool execution provenance (``dsagt-run`` ``tool.execute`` spans) and KB
observability (``kb.*`` / ``registry.*`` spans from MCP servers) always
work via OTLP regardless of run mode — the proxy decision affects only
the agent's own LLM-call traces.

Each agent's quirks live in its own module — see ``base.py`` for the
:class:`AgentSetup` ABC and one of the subclass modules
(``claude.py``, ``goose.py``, ``cline.py``, ``roo.py``, ``codex.py``)
for the platform-specific details and the per-agent investigation
behind its support tier.

Public API exported here:

- :func:`agent_env` — build the env dict for an agent process.
- :func:`agent_command` — argv list for interactive launch.
- :func:`static_agent_record` — write instructions + state dirs.
- :func:`static_agent_files_present` — has the static record been written?
- :func:`dynamic_agent_record` — write runtime-dependent files.
- :func:`launch_agent` — fork the agent and block until exit.
- :func:`agent_otel_support` — query an agent's OTel payload tier.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .base import (
    AgentSetup,
    _build_mcp_servers_dict,
    _mcp_env_block,
    _mcp_server_args,
    _PROXY_FORWARDED_SENTINEL,
)
from .claude import ClaudeSetup
from .cline import ClineSetup
from .codex import CodexSetup, _render_codex_config
from .goose import GooseSetup
from .opencode import OpenCodeSetup
from .roo import RooSetup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent registry (string name → setup class)
# ---------------------------------------------------------------------------

_AGENT_CLASSES: tuple[type[AgentSetup], ...] = (
    ClaudeSetup,
    GooseSetup,
    ClineSetup,
    RooSetup,
    CodexSetup,
    OpenCodeSetup,
)

AGENTS: dict[str, type[AgentSetup]] = {cls.name: cls for cls in _AGENT_CLASSES}


def _setup_for(agent_name: str, proxy_port: int | None = None) -> AgentSetup:
    """Return a fresh :class:`AgentSetup` instance for ``agent_name``.

    *proxy_port*, when set, is forwarded to the constructor so the
    instance binds its proxy-mode methods (``write_dynamic`` →
    ``proxy_write_dynamic``, ``run_script`` → ``proxy_run_script``).
    Callers don't have to know about the dispatch — they always call
    ``setup.write_dynamic(...)`` / ``setup.run_script(...)``.

    Raises ``KeyError`` with a helpful message if the agent isn't registered.
    """
    cls = AGENTS.get(agent_name)
    if cls is None:
        raise KeyError(f"Unknown agent {agent_name!r}.  Registered: {sorted(AGENTS)}.")
    return cls(proxy_port=proxy_port)


def _proxy_port_from_config(config: dict) -> int | None:
    """Extract the active proxy port from a config dict, if any."""
    return (config.get("proxy") or {}).get("port")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def agent_env(config: dict) -> dict:
    """Build the environment dict an agent process needs to inherit.

    Layers, in order:
      1. ``os.environ`` (user's shell env).
      2. DSAGT-wide vars (``DSAGT_PROJECT``, ``DSAGT_PROJECT_DIR``,
         ``DSAGT_AGENT``, ``DSAGT_SESSION_ID``).
      3. MLflow + OTel telemetry env (``MLFLOW_TRACKING_URI``,
         ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
         ``OTEL_RESOURCE_ATTRIBUTES``) so the agent's native OTel SDK and
         the MCP servers' ``init_tracing`` both land traces in the same
         MLflow experiment.
      4. Per-agent overrides via :meth:`AgentSetup.env_overrides` —
         each setup class owns its quirks: claude code's verbosity
         flags, goose's ``GOOSE_PROVIDER``/``GOOSE_MODEL`` selectors,
         translation of ``llm.{api_key,base_url}`` into the env-var
         names that agent's runtime reads, codex/cline state-dir env, etc.
      5. Proxy overrides (when ``--enable-proxy`` set) — last so they
         win over per-agent provider env.
    """
    pdir = config["project_dir"]
    agent_name = config["agent"]
    setup = _setup_for(agent_name)

    env = dict(os.environ)
    env["DSAGT_PROJECT"] = config["project"]
    env["DSAGT_PROJECT_DIR"] = pdir
    env["DSAGT_AGENT"] = agent_name
    if config.get("session_id"):
        env["DSAGT_SESSION_ID"] = config["session_id"]

    mlflow_port = config.get("mlflow", {}).get("port")
    proxy_port_for_otel = (config.get("proxy") or {}).get("port")
    if mlflow_port:
        mlflow_url = f"http://localhost:{mlflow_port}"
        env["MLFLOW_TRACKING_URI"] = mlflow_url
        # Claude in BYOA mode (no proxy) uses ``mlflow autolog claude``
        # for agent-side traces — its Stop hook produces richer
        # transcript-based traces than native OTel.  Skip OTel routing
        # so we don't get duplicate (and inferior) trace shapes.  MCP
        # servers and dsagt-run still get tracing because their
        # ``init_tracing`` reads MLflow URL from cwd's ``dsagt_config.yaml``,
        # not these env vars.
        skip_otel_for_claude_byoa = agent_name == "claude" and not proxy_port_for_otel
        if not skip_otel_for_claude_byoa:
            # OTel endpoint for the agent's native telemetry SDK.  The
            # x-mlflow-experiment-id header is mandatory for MLflow's OTLP
            # receiver — resolve to the experiment's numeric id once at
            # startup; if MLflow can't be reached yet we still write the env
            # vars so init_tracing can resolve later.
            env["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{mlflow_url}/v1/traces"
            experiment_id = _resolve_experiment_id(mlflow_url, config["project"])
            if experiment_id:
                env["OTEL_EXPORTER_OTLP_HEADERS"] = (
                    f"x-mlflow-experiment-id={experiment_id}"
                )
            # Resource attributes flow onto every span emitted by the agent's
            # OTel SDK.  ``session.id`` is what MLflow's OTLP receiver
            # promotes to ``mlflow.trace.session`` trace_metadata — required
            # so end-of-session memory extraction can find this run's traces.
            resource_attrs = [f"service.name={agent_name}"]
            if config.get("session_id"):
                resource_attrs.append(f"session.id={config['session_id']}")
            env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)

    pre_runtime_env = dict(env)
    # BYOA: only dsagt-owned env (telemetry capture flags, per-project
    # state dirs).  Provider credentials live in the user's shell;
    # ``env_overrides`` is reserved for Phase 2 proxy mode.
    env.update(setup.runtime_env(config))

    # Transparency: surface what credential env vars the agent will
    # actually pick up from the user's shell, so a missing or wrong
    # value isn't silently consumed.
    proxy_port = (config.get("proxy") or {}).get("port")
    if not proxy_port:
        _warn_on_preconfigured_creds(setup, env, pre_runtime_env)

    # Opt-in proxy routing (Phase 2).  When ``--enable-proxy`` populated
    # ``config["proxy"]["port"]``, delegate to the agent's proxy hooks.
    # ``proxy_env_overrides`` sets the proxy URL; ``env_overrides`` is
    # the per-agent llm-config translation hook the proxy path will use.
    if proxy_port:
        env.update(setup.env_overrides(config))
        env.update(setup.proxy_env_overrides(proxy_port))

    return env


def _warn_on_preconfigured_creds(
    setup: AgentSetup,
    env: dict,
    pre_overrides_env: dict,
) -> None:
    """Emit a one-line transparency warning when our env_overrides
    didn't populate any of the agent's credential env vars but the
    user's shell did.

    Lists the var NAMES (never values — keys are secrets).  Helps the
    user verify what the agent will actually pick up when the project
    YAML is empty or has unresolved ``${VAR}`` placeholders.

    Skipped when the proxy is enabled (proxy_env_overrides plants
    everything we need) and when the agent has no credential env vars
    declared (IDE-only agents).
    """
    if not setup.credential_env_vars:
        return

    # What did our env_overrides actually inject?
    injected = {
        k
        for k in setup.credential_env_vars
        if env.get(k) and env.get(k) != pre_overrides_env.get(k)
    }
    if injected:
        return  # Project YAML supplied credentials; nothing to warn about.

    # Nothing injected — list the vars present from the user's shell.
    from_shell = [k for k in setup.credential_env_vars if env.get(k)]
    if from_shell:
        logger.warning(
            "%s: project config has no llm credentials — agent will use "
            "preconfigured env vars: %s",
            setup.name,
            ", ".join(from_shell),
        )
    else:
        logger.warning(
            "%s: project config has no llm credentials and none of %s are "
            "set in the shell — agent may fall back to its own auth flow "
            "(claude.ai subscription, codex login, etc.) or fail at first call.",
            setup.name,
            ", ".join(setup.credential_env_vars),
        )


def _resolve_experiment_id(mlflow_url: str, project_name: str) -> str | None:
    """Look up the MLflow experiment id for *project_name*; create if absent.

    Returns None when MLflow isn't reachable yet — agent_env writes the
    rest of the OTel block anyway and ``init_tracing`` retries on the
    server side.  We keep this best-effort because dsagt-launch can race
    against MLflow startup; observability.py's _resolve_experiment_id
    runs again per-process and will succeed once MLflow is up.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_url)
        return str(mlflow.set_experiment(project_name).experiment_id)
    except Exception as e:
        logger.debug("could not resolve experiment id at agent_env time: %s", e)
        return None


def agent_command(config: dict) -> list[str]:
    """Return the shell command to launch the agent interactively."""
    return _setup_for(config["agent"]).interactive_command(config)


def agent_otel_support(agent_name: str) -> str:
    """Return the OTel-payload support tier for *agent_name*.

    See the module docstring matrix for the meaning of each tier.
    Consumers (smoke-test, ``dsagt info``, future warning systems) use
    this to decide whether to hard-fail or soft-warn when an agent's
    LLM-call traces don't appear in MLflow.
    """
    return _setup_for(agent_name).otel_payload_support


def static_agent_record(
    config: dict,
    agent: str,
    working_dir: str | Path,
) -> list[str]:
    """Write the agent's static project files: instructions + state dirs.

    Idempotent.  If the dsagt marker is already in the instructions file,
    the write is skipped — preserves any user edits made between init
    and start.
    """
    del config  # reserved
    return _setup_for(agent).write_static(Path(working_dir))


def static_agent_files_present(agent: str, working_dir: str | Path) -> bool:
    """True if the agent's marker file exists in the working dir."""
    return (Path(working_dir) / _setup_for(agent).static_marker).exists()


def dynamic_agent_record(
    config: dict,
    env: dict,
    working_dir: str | Path,
) -> list[str]:
    """Write the agent's runtime-dependent files: MCP config + ``.dsagt_env``
    + ``dsagt-launch.sh``.

    Caller must have already:
      - Resolved the agent and stored it in ``config["agent"]``
      - Started MLflow and updated ``config["mlflow"]["port"]`` to the
        actually-bound port
      - Built ``env`` via :func:`agent_env`
    """
    from .base import _write_launch_shim

    setup = _setup_for(config["agent"], proxy_port=_proxy_port_from_config(config))
    actions = setup.write_dynamic(
        config,
        env,
        Path(working_dir),
        Path(config["project_dir"]),
    )
    # Launch shim is BYOA-only. Skip when proxy mode is active —
    # ``dsagt start --enable-proxy`` is the only sensible entry point
    # in that mode (proxy URL must be plumbed through agent env).
    if not _proxy_port_from_config(config):
        actions.append(_write_launch_shim(setup, config, Path(working_dir)))
    return actions


def launch_agent(
    config: dict,
    env: dict,
    working_dir: str | Path,
    script_path: str | Path | None = None,
    max_turns: int = 30,
) -> int:
    """Launch the agent in the foreground.  Blocks until it exits."""
    working_dir = Path(working_dir)
    setup = _setup_for(config["agent"], proxy_port=_proxy_port_from_config(config))

    if script_path is not None:
        return setup.run_script(
            config,
            env,
            working_dir,
            Path(script_path),
            max_turns,
        )

    cmd = setup.interactive_command(config)
    logger.info("Launching: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error("Command not found: %s. %s", cmd[0], setup.install_hint)
        return 1
    except KeyboardInterrupt:
        return 0


__all__ = [
    "AGENTS",
    "AgentSetup",
    "agent_command",
    "agent_env",
    "agent_otel_support",
    "dynamic_agent_record",
    "launch_agent",
    "static_agent_files_present",
    "static_agent_record",
    # Re-exported for tests + other in-tree consumers
    "_build_mcp_servers_dict",
    "_mcp_env_block",
    "_mcp_server_args",
    "_PROXY_FORWARDED_SENTINEL",
    "_render_codex_config",
]
