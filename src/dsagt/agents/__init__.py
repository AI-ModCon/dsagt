"""
Agent platform configuration and launch.

Generates platform-specific config files (MCP server entries, agent
instructions, env vars, proxy routing) from the single ``dsagt_config.yaml``.
Launches the agent process in the foreground and blocks until it exits.

Each agent's quirks live in its own module — see ``base.py`` for the
:class:`AgentSetup` ABC and one of the subclass modules
(``claude.py``, ``goose.py``, ``cline.py``, ``roo.py``, ``codex.py``)
for the platform-specific details.

Public API exported here:

- :func:`agent_env` — build the env dict for an agent process.
- :func:`agent_command` — argv list for interactive launch.
- :func:`static_agent_record` — write instructions + state dirs.
- :func:`static_agent_files_present` — has the static record been written?
- :func:`dynamic_agent_record` — write runtime-dependent files.
- :func:`launch_agent` — fork the agent and block until exit.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .base import (
    AgentSetup,
    _PROXY_FORWARDED_SENTINEL,
    _build_mcp_servers_dict,
    _mcp_env_block,
    _mcp_server_args,
)
from .claude import ClaudeSetup
from .cline import ClineSetup
from .codex import CodexSetup, _render_codex_config
from .goose import GooseSetup
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
)

AGENTS: dict[str, type[AgentSetup]] = {cls.name: cls for cls in _AGENT_CLASSES}


def _setup_for(agent_name: str) -> AgentSetup:
    """Return a fresh :class:`AgentSetup` instance for ``agent_name``.

    Raises ``KeyError`` with a helpful message if the agent isn't registered.
    """
    cls = AGENTS.get(agent_name)
    if cls is None:
        raise KeyError(
            f"Unknown agent {agent_name!r}.  Registered: {sorted(AGENTS)}."
        )
    return cls()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def agent_env(config: dict) -> dict:
    """Build the environment dict an agent process needs to inherit.

    Layers, in order:
      1. ``os.environ`` (user's shell env)
      2. DSAGT-wide vars (``DSAGT_PROJECT``, ``DSAGT_PROJECT_DIR``,
         ``DSAGT_AGENT``)
      3. Per-agent overrides via :meth:`AgentSetup.env_overrides`
         (proxy routing, model pinning, sentinel API keys)
      4. Always-on shared vars (embedding URL through the proxy,
         ``OPENAI_BASE_URL``, ``EMBEDDING_MODEL``, ``MLFLOW_TRACKING_URI``,
         ``DSAGT_SESSION_ID``)
    """
    pdir = config["project_dir"]
    proxy_port = config["proxy"]["port"]
    agent_name = config["agent"]
    setup = _setup_for(agent_name)

    env = dict(os.environ)
    env["DSAGT_PROJECT"] = config["project"]
    env["DSAGT_PROJECT_DIR"] = pdir
    # Proxy's MlflowLogger subclass reads this to stamp dsagt.agent on
    # every agent-turn trace — lets the MLflow UI tell goose/claude/roo/
    # cline sessions apart without inspecting the request shape.
    env["DSAGT_AGENT"] = agent_name

    env.update(setup.env_overrides(config, proxy_port))

    # Embedding routing: every subprocess (agent → MCP server child, or
    # MCP server inherited via parent env on claude/goose) sees the proxy
    # URL for embeddings.  The real EMBEDDING_BASE_URL/API_KEY only live
    # in the dsagt-proxy subprocess's env (inherited from os.environ
    # before agent_env's override), where the proxy uses them to forward
    # upstream.  Sentinel API key on the agent side prevents accidental
    # direct calls to the real provider.
    env["EMBEDDING_BASE_URL"] = f"http://localhost:{proxy_port}"
    env["EMBEDDING_API_KEY"] = _PROXY_FORWARDED_SENTINEL
    env["OPENAI_BASE_URL"] = f"http://localhost:{proxy_port}"
    if config.get("embedding", {}).get("model"):
        env["EMBEDDING_MODEL"] = config["embedding"]["model"]

    # MLflow tracking URI inherited by every subprocess (dsagt-run, MCP
    # servers) so their mlflow.start_span() calls land in the project's
    # store alongside the proxy's LiteLLM-autologged traces.
    mlflow_port = config.get("mlflow", {}).get("port")
    if mlflow_port:
        env["MLFLOW_TRACKING_URI"] = f"http://localhost:{mlflow_port}"

    # Session id inherited by every subprocess so tool.execute / kb.*
    # spans share the same session tag as the proxy's LLM traces.
    if config.get("session_id"):
        env["DSAGT_SESSION_ID"] = config["session_id"]

    return env


def agent_command(config: dict) -> list[str]:
    """Return the shell command to launch the agent interactively."""
    return _setup_for(config["agent"]).interactive_command(config)


def static_agent_record(
    config: dict, agent: str, working_dir: str | Path,
) -> list[str]:
    """Write the agent's static project files: instructions + state dirs.

    Idempotent.  If the dsagt marker is already in the instructions file,
    the write is skipped — preserves any user edits made between init
    and start.

    The ``config`` arg is reserved for future use (e.g. project-specific
    instruction header injection); today it's unused but keeps the
    signature symmetric with :func:`dynamic_agent_record`.
    """
    del config  # reserved
    return _setup_for(agent).write_static(Path(working_dir))


def static_agent_files_present(agent: str, working_dir: str | Path) -> bool:
    """True if the agent's marker file exists in the working dir.

    Used by ``dsagt start`` to decide whether to call
    :func:`static_agent_record` on its way to launch.  Cheap stat check,
    no parsing.
    """
    return (Path(working_dir) / _setup_for(agent).static_marker).exists()


def dynamic_agent_record(
    config: dict, env: dict, working_dir: str | Path,
) -> list[str]:
    """Write the agent's runtime-dependent files: MCP config + ``.dsagt_env``,
    and (for cline) run ``cline auth`` + ``cline mcp add`` subprocesses.

    Caller must have already:
      - Resolved the agent and stored it in ``config["agent"]``
      - Started services and updated ``config["proxy"]["port"]`` /
        ``config["mlflow"]["port"]`` to the actually-bound ports
      - Built ``env`` via :func:`agent_env`
    """
    setup = _setup_for(config["agent"])
    return setup.write_dynamic(
        config,
        env,
        Path(working_dir),
        Path(config["project_dir"]),
        config["proxy"]["port"],
    )


def launch_agent(
    config: dict,
    env: dict,
    working_dir: str | Path,
    script_path: str | Path | None = None,
    max_turns: int = 30,
) -> int:
    """Launch the agent in the foreground.  Blocks until it exits.

    When ``script_path`` is set, dispatches to the agent's
    :meth:`AgentSetup.run_script` (each agent has a different shape for
    "run a script" — see the per-agent docstrings).  Otherwise launches
    the interactive command.

    Returns the agent's exit code.
    """
    working_dir = Path(working_dir)
    setup = _setup_for(config["agent"])

    if script_path is not None:
        return setup.run_script(
            config, env, working_dir, Path(script_path), max_turns,
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
    "dynamic_agent_record",
    "launch_agent",
    "static_agent_files_present",
    "static_agent_record",
    # Re-exported for tests + other in-tree consumers
    "_PROXY_FORWARDED_SENTINEL",
    "_build_mcp_servers_dict",
    "_mcp_env_block",
    "_mcp_server_args",
    "_render_codex_config",
]
