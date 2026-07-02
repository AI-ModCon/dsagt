"""
Agent platform configuration and launch.

Generates platform-specific config files (MCP server entries, agent
instructions, env vars) from the single ``.dsagt/config.yaml``.  Launches
the agent process in the foreground and blocks until it exits.

BYOA: each agent talks directly to its own provider.  DSAGT forces no
telemetry env on the agent — agent LLM-call history is recovered
post-hoc from the agent's on-disk session record, not by native OTel
emission.  We set only ``MLFLOW_TRACKING_URI`` (so the MCP servers and
any MLflow client log to the project's store) and per-project state
dirs.  Model selection / API keys / provider base URLs are the user's
responsibility.

The ``otel_payload_support`` class var records each agent's *potential*
native-OTel fidelity — retained as reference for the Phase 2 trace
pipeline, which reads transcripts rather than OTel:

  =========  ============  =============================================
  Agent      Support tier  Native MLflow visibility
  =========  ============  =============================================
  claude     full          yes — every turn lands as a trace with
                           messages, response, tool_use blocks
                           (gated by 4 ``CLAUDE_CODE_*`` /
                           ``OTEL_LOG_*`` flags we set)
  goose      full          yes — native OTel ``dispatch_tool_call`` span
                           (not consumed; no hook → no DSAGT autolog/memory)
  codex      partial       limited — Codex OTel spans carry only token
                           counts and tool names
  cline      none          no — Cline emits no OTel spans at all;
                           the agent is a black box from DSAgt's view
  opencode   none          no — no OTel wired in by default
  =========  ============  =============================================

Tool execution provenance (``dsagt-run`` ``tool.execute`` spans) and KB
observability (``kb.*`` / ``registry.*`` spans from MCP servers) always
work via OTLP — independent of the agent's own LLM-call traces.

Each agent's quirks live in its own module — see ``base.py`` for the
:class:`AgentSetup` ABC and one of the subclass modules
(``claude.py``, ``goose.py``, ``cline.py``, ``codex.py``, ``opencode.py``)
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
)
from .claude import ClaudeSetup
from .cline import ClineSetup
from .codex import CodexSetup, _render_codex_config
from .goose import GooseSetup
from .opencode import OpenCodeSetup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent registry (string name → setup class)
# ---------------------------------------------------------------------------

_AGENT_CLASSES: tuple[type[AgentSetup], ...] = (
    ClaudeSetup,
    GooseSetup,
    ClineSetup,
    CodexSetup,
    OpenCodeSetup,
)

AGENTS: dict[str, type[AgentSetup]] = {cls.name: cls for cls in _AGENT_CLASSES}


def _setup_for(agent_name: str) -> AgentSetup:
    """Return a fresh :class:`AgentSetup` instance for ``agent_name``.

    Raises ``KeyError`` with a helpful message if the agent isn't registered.
    """
    cls = AGENTS.get(agent_name)
    if cls is None:
        raise KeyError(f"Unknown agent {agent_name!r}.  Registered: {sorted(AGENTS)}.")
    return cls()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def agent_env(config: dict) -> dict:
    """Build the environment dict an agent process needs to inherit.

    Layers, in order:
      1. ``os.environ`` (user's shell env).
      2. DSAGT-wide vars (``DSAGT_PROJECT``, ``DSAGT_PROJECT_DIR``,
         ``DSAGT_AGENT``, ``DSAGT_SESSION_ID``).
      3. ``MLFLOW_TRACKING_URI`` so the MCP servers' ``init_tracing`` and
         any MLflow client running under the agent log to the project's
         store.  No OTel routing env — DSAGT does not force native agent
         telemetry; agent traces are recovered post-hoc from the on-disk
         transcript.
      4. Per-agent dsagt-owned runtime env via
         :meth:`AgentSetup.runtime_env` — per-project state dirs only
         (``CLINE_DIR``, ``CODEX_HOME``).
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

    from dsagt.observability import resolve_tracking_uri

    env["MLFLOW_TRACKING_URI"] = resolve_tracking_uri(config)

    # BYOA: only dsagt-owned env (per-project state dirs).  DSAGT forces
    # no telemetry env on the agent — agent traces are recovered post-hoc
    # from the on-disk transcript.  Provider credentials live in the
    # user's shell.
    env.update(setup.runtime_env(config))

    # Transparency: surface what credential env vars the agent will
    # actually pick up from the user's shell, so a missing or wrong
    # value isn't silently consumed.
    _warn_on_preconfigured_creds(setup, env)

    return env


def _warn_on_preconfigured_creds(setup: AgentSetup, env: dict) -> None:
    """Emit a one-line transparency note listing which of the agent's
    credential env vars are present in the user's shell.

    Lists the var NAMES (never values — keys are secrets).  Helps the
    user verify what the agent will actually pick up.  No-op for agents
    with no credential env vars declared (IDE-only agents).
    """
    if not setup.credential_env_vars:
        return

    from_shell = [k for k in setup.credential_env_vars if env.get(k)]
    if from_shell:
        logger.warning(
            "%s: agent will use preconfigured env vars from the shell: %s",
            setup.name,
            ", ".join(from_shell),
        )
    else:
        logger.warning(
            "%s: none of %s are set in the shell — agent may fall back to "
            "its own auth flow (claude.ai subscription, codex login, etc.) "
            "or fail at first call.",
            setup.name,
            ", ".join(setup.credential_env_vars),
        )


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
    """Write the agent's runtime-dependent files: the per-agent MCP config.

    Caller must have already:
      - Resolved the agent and stored it in ``config["agent"]``
      - Built ``env`` via :func:`agent_env`

    No launch shim is written — ``dsagt init`` collapses to config +
    instructions + MCP config; the user starts the agent directly in the
    project dir or via ``dsagt start``.
    """
    setup = _setup_for(config["agent"])
    actions = setup.write_dynamic(
        config,
        env,
        Path(working_dir),
        Path(config["project_dir"]),
    )
    # Mirror installed skills into the agent's native skills dir (all agents).
    # Central here so each agent only declares native_skills_dir.
    actions += setup.setup_skills(Path(working_dir), config)
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
    setup = _setup_for(config["agent"])

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
    "_render_codex_config",
]
