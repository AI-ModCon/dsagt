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
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

# Master instructions ship with the package, one directory above this file.
_INSTRUCTIONS_PATH = Path(__file__).parent.parent / "dsagt_instructions.md"

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
        "get_registry",
        "http_request",
        "install_dependencies",
        "read_file",
        "reconstruct_pipeline",
        "run_command",
        "save_skill",
        "save_tool_spec",
        "search_registry",
        "search_skills",
    ],
    "knowledge": [
        "kb_add_vector_db",
        "kb_append",
        "kb_dismiss_suggestion",
        "kb_get_memories",
        "kb_get_suggestions",
        "kb_ingest",
        "kb_job_status",
        "kb_list_collections",
        "kb_remember",
        "kb_search",
    ],
}


# Sentinel API key planted in agent / MCP-child env when the optional
# proxy is enabled, so any direct call that bypasses the proxy returns
# 401 fast — fails loudly instead of silently leaking around our
# observability pipeline.
_PROXY_FORWARDED_SENTINEL = "dsagt-proxy-forwarded-disable-direct-calls"


def _real(value) -> str | None:
    """Return *value* trimmed, or ``None`` if blank or an unresolved
    ``${VAR}`` interpolation.  Centralized so each agent's
    ``env_overrides`` filters config values consistently — we never
    propagate a placeholder to a downstream env var.
    """
    s = (value or "").strip() if isinstance(value, str) else ""
    if not s or s.startswith("${"):
        return None
    return s


def _anthropic_env(llm: dict) -> dict[str, str]:
    """Anthropic-protocol env vars from ``llm.*`` config.

    Returns ``{}`` unless ``llm.provider == "anthropic"`` — never
    propagates anthropic vars on top of an openai-shaped upstream,
    where they'd be meaningless or actively misleading.

    Called only by agent setups whose native runtime speaks anthropic
    (claude code), and by multi-protocol agents (goose, cline, roo)
    when the configured provider matches.
    """
    if (_real(llm.get("provider")) or "").lower() != "anthropic":
        return {}
    out: dict[str, str] = {}
    if api_key := _real(llm.get("api_key")):
        out["ANTHROPIC_API_KEY"] = api_key
    if base_url := _real(llm.get("base_url")):
        out["ANTHROPIC_BASE_URL"] = base_url
    if model := _real(llm.get("model")):
        out["ANTHROPIC_MODEL"] = model
    return out


def _openai_env(llm: dict) -> dict[str, str]:
    """OpenAI-protocol env vars from ``llm.*`` config.

    Returns ``{}`` unless ``llm.provider`` is one of the OpenAI-wire
    aliases (``openai``, ``openai_like``, ``azure``).  Only called by
    agents whose native runtime speaks the OpenAI wire protocol (codex,
    LiteLLM-backed paths) or by multi-protocol agents (goose, cline,
    roo) when the configured provider matches.
    """
    provider = (_real(llm.get("provider")) or "").lower()
    if provider not in ("openai", "openai_like", "azure"):
        return {}
    out: dict[str, str] = {}
    if api_key := _real(llm.get("api_key")):
        out["OPENAI_API_KEY"] = api_key
    if base_url := _real(llm.get("base_url")):
        out["OPENAI_BASE_URL"] = base_url
    return out


# ---------------------------------------------------------------------------
# Functional helpers (provider-agnostic)
# ---------------------------------------------------------------------------


def _mcp_server_args(server: str) -> list[str]:
    """Build the argv tail for ``uv run dsagt-<server>-server``.

    Both servers read all configuration from ``DSAGT_PROJECT_DIR`` (env)
    and dsagt_config.yaml — no CLI args needed.
    """
    return ["run", f"dsagt-{server}-server"]


def _mcp_env_block(config: dict) -> dict[str, str]:
    """Env vars the dsagt MCP server children need at startup.

    Project routing (project name, project_dir, mlflow port) lives in
    ``dsagt_config.yaml`` and is read by services via cwd-walk — single
    source of truth, no env duplication.  This block carries only the
    embedding-backend settings, which the embedding client still reads
    from env (refactor TBD).  For ``backend: api`` the user must set
    ``EMBEDDING_API_KEY`` in their shell env (creds never on disk).
    """
    emb = config.get("embedding") or {}
    block: dict[str, str] = {}
    for key, src in (
        ("EMBEDDING_BACKEND", emb.get("backend")),
        ("EMBEDDING_MODEL", emb.get("model")),
        ("EMBEDDING_BASE_URL", emb.get("base_url")),
    ):
        if src:
            block[key] = str(src)
    return block


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

    return json.dumps(
        {
            "customModes": [
                {
                    "slug": "dsagt",
                    "name": "DSAgt Pipeline Builder",
                    "roleDefinition": role,
                    "customInstructions": custom,
                    "groups": ["read", "edit", "browser", "command", "mcp"],
                    "source": "project",
                }
            ]
        },
        indent=2,
    )


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
# Launch shim — dsagt-launch.sh
# ---------------------------------------------------------------------------


def _render_launch_shim(setup: AgentSetup, config: dict) -> str:
    """Render the dsagt-launch.sh shell script for a project.

    The shim is the BYOA "transparency" entry point — users either run it
    directly (``bash dsagt-launch.sh``), or read it and execute the lines
    manually.  It does NOT exec the agent — instead it prints how to
    launch (CLI / VS Code), letting the user pick.

    Steps:
      1. Start MLflow in the background if not already running.
      2. Resolve the experiment-id for this project at run time
         (curl MLflow's REST API).
      3. Export OTel routing env (skipped for Claude — mlflow autolog
         claude's ``.claude/settings.json`` handles agent-side traces).
      4. Export per-agent telemetry verbosity flags.
      5. Print available agent-launch options.
    """
    project = config["project"]
    mlflow_port = (config.get("mlflow") or {}).get("port")
    agent_name = setup.name
    pdir = config.get("project_dir") or "."

    cli_cmd = " ".join(shlex.quote(p) for p in setup.interactive_command({}))
    vscode_lines = setup.vscode_hint(Path(str(pdir))) or []

    # Claude in BYOA gets agent-side traces via mlflow autolog claude's
    # Stop hook, so we omit the OTel routing block (would create
    # duplicate, lower-fidelity traces alongside the rich transcript).
    skip_otel_routing = agent_name == "claude"

    lines: list[str] = []
    lines.append("#!/usr/bin/env bash")
    lines.append("# dsagt-launch.sh — start MLflow, set env, show launch options.")
    lines.append(
        "# Generated by `dsagt init`. Re-running `dsagt init` overwrites this file."
    )
    lines.append("set -euo pipefail")
    lines.append('cd "$(dirname "$0")"')
    lines.append("")
    lines.append("# 1. Start MLflow in the background if not already running.")
    lines.append(f"dsagt mlflow {shlex.quote(project)} --background-only")
    lines.append("")

    if not skip_otel_routing and mlflow_port:
        lines.append("# 2. Resolve experiment id for OTel routing.")
        lines.append(
            f"EXPERIMENT_ID=$(curl -s "
            f'"http://localhost:{mlflow_port}/api/2.0/mlflow/experiments/get-by-name?experiment_name={project}" '
            '| python3 -c \'import json,sys; print(json.load(sys.stdin)["experiment"]["experiment_id"])\')'
        )
        lines.append("")
        lines.append(
            "# 3. OTel routing env (agent's native OTel SDK ships traces to MLflow)."
        )
        lines.append(f'export MLFLOW_TRACKING_URI="http://localhost:{mlflow_port}"')
        lines.append("export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf")
        lines.append(
            f'export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://localhost:{mlflow_port}/v1/traces"'
        )
        lines.append(
            'export OTEL_EXPORTER_OTLP_HEADERS="x-mlflow-experiment-id=$EXPERIMENT_ID"'
        )
        lines.append(f'export OTEL_RESOURCE_ATTRIBUTES="service.name={agent_name}"')
        lines.append("")
    elif mlflow_port:
        lines.append(
            "# Claude uses `mlflow autolog claude` (.claude/settings.json) for"
        )
        lines.append("# agent-side traces — no OTel routing needed.")
        lines.append(f'export MLFLOW_TRACKING_URI="http://localhost:{mlflow_port}"')
        lines.append("")

    if setup.telemetry_env:
        lines.append("# 4. Agent telemetry verbosity flags.")
        for k, v in setup.telemetry_env.items():
            lines.append(f"export {k}={shlex.quote(v)}")
        lines.append("")

    lines.append("# 5. Show how to launch the agent — pick one.")
    lines.append('echo ""')
    lines.append('echo "Environment ready. Launch the agent in one of these ways:"')
    lines.append('echo ""')
    lines.append(f'echo "  CLI:       {cli_cmd}"')
    if vscode_lines:
        for hint in vscode_lines:
            lines.append(f'echo "  VS Code:   {hint}"')
    lines.append('echo ""')
    lines.append('echo "Run any of the above in this shell."')
    lines.append("")
    return "\n".join(lines)


def _write_launch_shim(setup: AgentSetup, config: dict, working_dir: Path) -> str:
    """Write ``dsagt-launch.sh`` to working_dir and chmod it executable.

    Returns a one-line action description for the init output.
    """
    shim_path = working_dir / "dsagt-launch.sh"
    shim_path.write_text(_render_launch_shim(setup, config))
    shim_path.chmod(0o755)
    return f"Wrote {shim_path}"


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

    def __init__(self, proxy_port: int | None = None):
        """Rebind ``write_dynamic`` / ``run_script`` to their proxy-mode
        analogs (``proxy_write_dynamic`` / ``proxy_run_script``) when
        Phase 2 proxy mode is active AND the subclass overrides them.

        Encapsulates the dispatch so callers always say ``setup.write_dynamic(...)``
        without knowing whether the BYOA or proxy implementation runs.

        Why the override check: the base default ``proxy_write_dynamic``
        delegates to ``self.write_dynamic`` for fall-through.  If we
        rebind ``self.write_dynamic = self.proxy_write_dynamic`` when
        the subclass DOESN'T override, the delegation infinite-loops
        (proxy_write_dynamic → self.write_dynamic → proxy_write_dynamic).
        Only rebind when the subclass has its own implementation.

        Net effect: agents whose proxy + BYOA setup is identical (claude,
        goose, roo's write_dynamic) inherit the BYOA path under proxy
        mode too — no override, no rebind, ``setup.write_dynamic`` stays
        as ``write_dynamic``.  Agents that genuinely differ (cline +
        codex + opencode write_dynamic; cline + roo run_script) get
        their proxy_* override bound in.
        """
        self.proxy_port = proxy_port
        if proxy_port:
            cls = type(self)
            if cls.proxy_write_dynamic is not AgentSetup.proxy_write_dynamic:
                self.write_dynamic = self.proxy_write_dynamic  # type: ignore[method-assign]
            if cls.proxy_run_script is not AgentSetup.proxy_run_script:
                self.run_script = self.proxy_run_script  # type: ignore[method-assign]

    #: Env vars the agent's runtime reads for credentials / endpoint
    #: routing — i.e. the vars our ``env_overrides`` translates ``llm.*``
    #: config into.  Used by ``agent_env`` to surface a transparency
    #: warning when the project YAML is empty: we list which of these
    #: are actually present in the user's shell so the user can see
    #: what the agent will pick up.  Empty for IDE-extension agents
    #: that never read env-var credentials.
    credential_env_vars: ClassVar[tuple[str, ...]] = ()

    #: Whether this agent makes its LLM calls visible in MLflow natively
    #: (without DSAgt's optional ``dsagt-proxy``).  Drives whether
    #: ``dsagt info`` / live audit / memory extraction see the agent's
    #: reasoning + tool calls or just see the agent as a black box.
    #:
    #: ``"full"``        — verified end-to-end (Claude Code, Goose).
    #:                     Every agent turn lands in MLflow as a trace
    #:                     with messages + response + tool_use blocks.
    #:                     ``--enable-proxy`` is unnecessary.
    #: ``"partial"``     — agent emits OTel but spans don't carry
    #:                     message content (Codex: only token counts +
    #:                     tool names).  Conversation may be available
    #:                     via a non-OTel side-channel (Codex's
    #:                     ``~/.codex/sessions/*.jsonl``); not yet
    #:                     wired in.  ``--enable-proxy`` recommended.
    #: ``"none"``        — agent emits no OTel traces, or emits only
    #:                     metrics without payloads (Cline, Roo Code).
    #:                     The agent is a black box from DSAgt's
    #:                     perspective; ``--enable-proxy`` is the only
    #:                     way to see what it's doing.  Tool execution
    #:                     + KB observability still work regardless via
    #:                     dsagt-run / MCP-server spans.
    #:
    #: See agents/<name>.py docstrings for the per-agent investigation.
    otel_payload_support: ClassVar[str] = "full"

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
    ) -> list[str]:
        """Write the agent's runtime-dependent files (MCP config, .dsagt_env).

        Caller must have already populated ``config["mlflow"]["port"]`` with
        the actually-bound port and built ``env`` via :func:`agent_env`.
        Returns a list of one-line action descriptions.
        """

    def runtime_env(self, config: dict) -> dict[str, str]:
        """Dsagt-owned env vars the agent process needs at runtime (BYOA).

        Returned dict is layered into ``agent_env`` and the launch shim's
        telemetry block.  Default returns ``telemetry_env`` (claude's
        OTEL_LOG_* / CLAUDE_CODE_* gates, etc.).  Subclasses augment with
        per-project state-dir env (``CLINE_DIR``, ``CODEX_HOME``).

        This is the only method allowed to set agent-affecting env in
        BYOA.  Provider credentials (ANTHROPIC_*, OPENAI_*, GOOSE_*) are
        the user's responsibility — exported in their shell, never
        translated from ``config["llm"]``.
        """
        del config
        return dict(self.telemetry_env)

    def env_overrides(self, config: dict) -> dict[str, str]:
        """Phase-2 proxy-mode hook.  Not called in BYOA.

        Reserved for the proxy/OTel-redirect path that ``observability.py``
        will own when Phase 2 lands — the hook each agent uses to translate
        ``config["llm"].*`` into provider env vars (after the proxy has
        already rewritten the upstream URL).  Default implementation
        returns ``{}``; agents override with their own translation logic
        as needed.
        """
        del config
        return {}

    def proxy_env_overrides(self, proxy_port: int) -> dict[str, str]:
        """Env vars that route the agent's LLM calls through dsagt-proxy.

        Default implementation covers every agent we support today —
        we set the well-known base URLs (``ANTHROPIC_BASE_URL``,
        ``OPENAI_BASE_URL``) for both wire protocols so an agent that
        speaks either still hits the proxy, plus the embedding base
        URL so MCP-server children inherit proxy routing for
        ``litellm.embedding`` calls.  Real upstream credentials live
        only in the proxy subprocess; we plant the sentinel API key in
        every standard slot so a direct call (bypassing the proxy)
        fails loudly with 401 rather than silently leaking around our
        observability pipeline.

        Subclasses may override only if an agent reads non-standard
        env names.  None do today.
        """
        proxy_url = f"http://localhost:{proxy_port}"
        return {
            "ANTHROPIC_BASE_URL": proxy_url,
            "OPENAI_BASE_URL": proxy_url,
            "EMBEDDING_BASE_URL": proxy_url,
            "ANTHROPIC_API_KEY": _PROXY_FORWARDED_SENTINEL,
            "OPENAI_API_KEY": _PROXY_FORWARDED_SENTINEL,
            "EMBEDDING_API_KEY": _PROXY_FORWARDED_SENTINEL,
        }

    def interactive_command(self, config: dict) -> list[str]:
        """Return the argv list for interactive launch.

        Default is ``base_command`` unmodified.  Goose overrides to append
        ``--with-extension`` flags for the dsagt MCP servers.
        """
        del config
        return list(self.base_command)

    #: Telemetry env vars this agent emits via its native OTel SDK.
    #: Set by subclasses with ``otel_payload_support`` of "full" or
    #: "partial".  Empty for agents that don't emit OTel (cline / roo).
    #: Used by ``byoa_env_hints`` to surface what the user should
    #: export in their shell to get full visibility.
    telemetry_env: ClassVar[dict[str, str]] = {}

    #: Per-agent provider-credential hints surfaced by ``byoa_env_hints``.
    #: List of ``(env_var_name, hint)`` tuples shown to the user with a
    #: "skip if already configured" note.
    credential_hints: ClassVar[tuple[tuple[str, str], ...]] = ()

    def byoa_env_hints(
        self, mlflow_port: int, project: str, project_dir: Path
    ) -> list[tuple[str, str]]:
        """Provider-credential env vars the user should set in their shell.

        BYOA design: dsagt-internal env (project routing, MLflow URL,
        OTel endpoint + headers, telemetry verbosity flags) lives in
        the per-project launch shim — not the user's shell.  This
        method returns only the credentials the user owns, with one
        hint string each.
        """
        del mlflow_port, project, project_dir
        return list(self.credential_hints)

    def launch_oneliner(self, project: str, project_dir: Path) -> str:
        """Shell command to launch the agent interactively.

        ``cd <pdir>`` puts both the agent and any ``dsagt-run`` children
        in the project directory; readers find ``dsagt_config.yaml``
        there and use it as the single source of truth for project /
        agent / mlflow routing — no DSAGT_* env vars to manage.
        """
        del project
        cmd = " ".join(shlex.quote(part) for part in self.interactive_command({}))
        return f"cd {shlex.quote(str(project_dir))} && {cmd}"

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

    def vscode_hint(self, project_dir: Path) -> list[str] | None:
        """One-or-two-line hint for users who run this agent as a VS Code
        extension instead of via the CLI.  Returns ``None`` for agents
        without a working VS Code extension (most of them — only claude
        and roo currently have extensions that auto-discover dsagt's
        per-project files from the workspace root).

        ``dsagt init`` prints these lines under "Or with VS Code extension".
        """
        del project_dir
        return None

    # --- Phase 2 proxy-mode analogs ---------------------------------------
    # ``__init__`` rebinds ``write_dynamic`` / ``run_script`` to these when
    # ``proxy_port`` is set, so callers don't branch.  Defaults defer to
    # the BYOA implementation — agents whose proxy + BYOA paths are the
    # same (claude, goose, roo's write_dynamic) inherit this fallthrough.
    # Override in subclasses where proxy mode genuinely differs (cline
    # auth -b proxy, codex [model_providers.dsagt-proxy] block, opencode
    # baseURL override, cline/roo run_script un-punt).

    def proxy_write_dynamic(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        pdir: Path,
    ) -> list[str]:
        return self.write_dynamic(config, env, working_dir, pdir)

    def proxy_run_script(
        self,
        config: dict,
        env: dict,
        working_dir: Path,
        script_path: Path,
        max_turns: int,
    ) -> int:
        return self.run_script(config, env, working_dir, script_path, max_turns)
