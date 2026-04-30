"""
Agent setup base class + shared helpers.

The :class:`AgentSetup` ABC captures the contract every supported agent
follows; subclasses live in sibling modules (one per agent) and own their
quirks in one place.  See ``src/dsagt/agents/__init__.py`` for the public
``agent_env`` / ``static_agent_record`` / ``dynamic_agent_record`` /
``launch_agent`` API that wires this together.
"""

from __future__ import annotations

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

# Master instructions ship with the package, one directory above this file.
_INSTRUCTIONS_PATH = Path(__file__).parent.parent / "dsagt_instructions.md"

# Sentinel API key planted in agent / MCP-child env so that if anything
# bypasses the proxy and calls api.openai.com / api.anthropic.com directly,
# the upstream returns 401 — failing loudly instead of silently leaking
# requests around our provenance + tracing pipeline.
_PROXY_FORWARDED_SENTINEL = "dsagt-proxy-forwarded-disable-direct-calls"

# Marker string we look for to decide whether an instructions file already
# carries the dsagt body.  Present in the master instructions header and in
# every per-agent format we emit (CLAUDE.md, AGENTS.md, .goosehints,
# .clinerules/dsagt_instructions.md, .roomodes' JSON-embedded instructions).
# Lets ``_append_or_write`` be idempotent so users can edit instructions
# files between init and start without losing edits on the next start.
_DSAGT_MARKER = "DSAgt Pipeline Builder"

# Tools each dsagt MCP server exposes — listed in ``alwaysAllow`` so roo
# and cline auto-approve them without a human-in-the-loop prompt.  Keep in
# sync with ``commands/registry_server.py`` and
# ``commands/knowledge_server.py`` tool registrations; a tool added there
# but not here means roo/cline will hang on its first call.
_DSAGT_MCP_ALWAYS_ALLOW = {
    "registry": [
        "get_registry", "http_request", "install_dependencies", "read_file",
        "reconstruct_pipeline", "run_command", "save_tool_spec",
        "search_registry", "search_skills",
    ],
    "knowledge": [
        "kb_add_vector_db", "kb_append", "kb_dismiss_suggestion",
        "kb_get_memories", "kb_get_suggestions", "kb_ingest", "kb_job_status",
        "kb_list_collections", "kb_remember", "kb_search",
    ],
}


# ---------------------------------------------------------------------------
# Functional helpers (provider-agnostic)
# ---------------------------------------------------------------------------

def _mcp_server_args(server: str) -> list[str]:
    """Build the argv tail for ``uv run dsagt-<server>-server``.

    Both servers read all configuration from ``DSAGT_PROJECT_DIR`` (env)
    and dsagt_config.yaml — no CLI args needed.
    """
    return ["run", f"dsagt-{server}-server"]


def _mcp_env_block(config: dict, proxy_port: int) -> dict:
    """Build the env block for MCP server entries that get embedded in JSON.

    Embedding requests from dsagt-knowledge-server route through our local
    LiteLLM proxy at localhost:<proxy_port>, same as agent LLM calls — the
    proxy holds the real upstream credentials.  MCP children only see the
    sentinel API key and the proxy URL; if anything misroutes around the
    proxy, a direct call to api.openai.com / api.cohere.com / etc. 401s
    the sentinel and fails loudly instead of silently bypassing.
    """
    proxy_url = f"http://localhost:{proxy_port}"
    return {
        "DSAGT_PROJECT_DIR": config["project_dir"],
        "EMBEDDING_BASE_URL": proxy_url,
        "EMBEDDING_API_KEY": _PROXY_FORWARDED_SENTINEL,
        # OPENAI_BASE_URL kept aligned with EMBEDDING_BASE_URL so any
        # LiteLLM internal openai-client fallback also goes through proxy.
        "OPENAI_BASE_URL": proxy_url,
        "EMBEDDING_MODEL": config.get("embedding", {}).get("model", ""),
    }


def _load_master_instructions() -> str | None:
    """Load the master DSAgt instructions, or None if the file is missing."""
    if _INSTRUCTIONS_PATH.exists():
        return _INSTRUCTIONS_PATH.read_text()
    logger.warning("Master instructions not found: %s", _INSTRUCTIONS_PATH)
    return None


def _format_roomodes(instructions: str) -> str:
    """Wrap master instructions as a Roo Code .roomodes JSON file."""
    parts = instructions.split("\n## ", 1)
    role = parts[0].strip()
    custom = ("## " + parts[1]).strip() if len(parts) > 1 else ""

    return json.dumps({
        "customModes": [{
            "slug": "dsagt",
            "name": "DSAgt Pipeline Builder",
            "roleDefinition": role,
            "customInstructions": custom,
            "groups": ["read", "edit", "browser", "command", "mcp"],
            "source": "project",
        }]
    }, indent=2)


def _write_env_file(path: Path, env_vars: dict) -> None:
    """Write a sourceable ``.dsagt_env`` shell file."""
    lines = [f'export {k}="{v}"' for k, v in env_vars.items()]
    path.write_text("\n".join(lines) + "\n")


def _append_or_write(path: Path, content: str, marker: str) -> str | None:
    """Idempotent write for instructions files.

    - File doesn't exist → write content.
    - File exists, marker absent → append content (preserves user prefix).
    - File exists, marker present → no-op (preserves user edits).

    Returns a one-line action description, or None on no-op.
    """
    if path.exists():
        existing = path.read_text()
        if marker in existing:
            return None
        path.write_text(existing + "\n\n" + content)
        return f"Appended DSAgt instructions to {path}"
    path.write_text(content)
    return f"Wrote {path}"


def _mcp_subprocess_env(parent_env: dict) -> dict:
    """Pick the env vars a dsagt MCP server child needs from the launch env.

    Cline, roo, and codex strip parent env when spawning MCP server
    subprocesses (claude and goose inherit, so they don't need this).
    Everything the server needs at startup must be listed explicitly.  The
    ``${LLM_*}``/``${EMBEDDING_*}`` placeholders in dsagt_config.yaml are
    resolved at server startup via ``resolve_env_vars``; if any are missing,
    the placeholder leaks through and the knowledge server's
    ``api_key.startswith("${")`` validation kills the process — manifesting
    agent-side as a silent connection failure.  Pass them all through.
    """
    keys = (
        "DSAGT_PROJECT_DIR", "DSAGT_PROJECT", "DSAGT_SESSION_ID", "DSAGT_AGENT",
        "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL",
        "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
        "OPENAI_BASE_URL",
        "MLFLOW_TRACKING_URI",
    )
    return {k: parent_env[k] for k in keys if k in parent_env}


def _build_mcp_servers_dict(env_block: dict | None) -> dict:
    """Build the standard ``{"mcpServers": {...}}`` dict for dsagt servers.

    Used by agents that load MCP config from a JSON file (roo via
    ``.roo/mcp.json``).  Claude Code uses the same shape via ``.mcp.json``
    but builds it inline in :class:`ClaudeSetup.write_dynamic`.  Cline
    doesn't use this — it requires ``cline mcp add`` to register servers.
    """
    mcp_config: dict = {"mcpServers": {}}
    for server in ("registry", "knowledge"):
        entry: dict = {
            "command": "uv",
            "args": _mcp_server_args(server),
            "disabled": False,
            "alwaysAllow": _DSAGT_MCP_ALWAYS_ALLOW[server],
        }
        if env_block:
            entry["env"] = env_block
        mcp_config["mcpServers"][f"dsagt-{server}"] = entry
    return mcp_config


def _toml_quote(value: str) -> str:
    """TOML-quote a string: escape backslashes and double-quotes only.

    Codex config.toml is regular TOML — basic strings need backslash and
    quote escaping but not control chars (which we don't have in any of
    the values we emit: paths, URLs, model names).  Avoids pulling in a
    TOML writer dep just for a few lines.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_simple_script(
    cmd: list[str],
    env: dict,
    working_dir: Path,
    install_hint: str,
) -> int:
    """Common ``subprocess.run`` wrapper used by every agent's script runner.

    Returns the agent's exit code, 1 on FileNotFoundError (with a helpful
    install hint), 0 on KeyboardInterrupt.
    """
    logger.info("Launching: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, env=env, cwd=str(working_dir)).returncode
    except FileNotFoundError:
        logger.error("Command not found: %s. %s", cmd[0], install_hint)
        return 1
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# AgentSetup ABC
# ---------------------------------------------------------------------------

class AgentSetup(ABC):
    """Per-agent setup contract.

    Each subclass owns one agent's quirks in one file: the marker file the
    static record creates, the runtime config the dynamic record writes,
    the env vars the agent's process needs, and how to launch it in
    interactive vs script mode.

    Class attributes (set by every subclass):

    - ``name``           — agent identifier as used in ``dsagt_config.yaml``
                          (e.g. ``"claude"``, ``"goose"``).
    - ``base_command``   — argv list that launches the agent interactively.
                          Subclasses may override :meth:`interactive_command`
                          to extend it (goose appends ``--with-extension``).
    - ``static_marker``  — filename relative to the working dir that
                          :func:`static_agent_files_present` stats to decide
                          whether the static record has already been written.
    - ``install_hint``   — one-line install instruction shown on
                          FileNotFoundError; surfaces what the user needs to
                          install if the binary isn't on PATH.
    """

    name: ClassVar[str]
    base_command: ClassVar[list[str]]
    static_marker: ClassVar[str]
    install_hint: ClassVar[str] = "Install the agent CLI first."

    @abstractmethod
    def write_static(self, working_dir: Path) -> list[str]:
        """Write the agent's instructions file + any state directories.

        Idempotent: if the dsagt marker is already in the instructions
        file, the write is skipped (preserves user edits).  Returns a
        list of one-line action descriptions.
        """

    @abstractmethod
    def write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
        proxy_port: int,
    ) -> list[str]:
        """Write the agent's runtime-dependent files (MCP config, .dsagt_env).

        Caller must have already populated ``config["proxy"]["port"]`` /
        ``config["mlflow"]["port"]`` with the actually-bound ports and
        built ``env`` via :func:`agent_env`.  Returns a list of one-line
        action descriptions.
        """

    @abstractmethod
    def env_overrides(self, config: dict, proxy_port: int) -> dict[str, str]:
        """Per-agent additions to the base env dict.

        Examples: ANTHROPIC_BASE_URL for claude/roo, OPENAI_HOST for goose,
        CODEX_HOME for codex, sentinel API keys to prevent direct upstream
        calls.  See :func:`agent_env` (in ``__init__.py``) for how this
        layers with the always-on env vars (embedding URL, MLflow URI,
        session id) shared across all agents.
        """

    def interactive_command(self, config: dict) -> list[str]:
        """Return the argv list for interactive launch.

        Default is ``base_command`` unmodified.  Goose overrides to append
        ``--with-extension`` flags for the dsagt MCP servers.
        """
        del config
        return list(self.base_command)

    @abstractmethod
    def run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        """Run the agent in non-interactive batch mode.

        Each agent has a different shape for "run a script"; see the
        per-agent docstrings.  Returns the agent's exit code.
        """
