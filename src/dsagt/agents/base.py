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
import shutil
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

# Tools the dsagt MCP server exposes — listed in ``alwaysAllow`` so cline
# auto-approves them without a human-in-the-loop prompt.  Keep in
# sync with the ``mcp/*_tools.py`` tool registrations (registry / knowledge /
# memory / skill); a tool added there but not here means cline will hang on
# its first call.  All dsagt MCP tools live behind the single ``dsagt-server``,
# so the always-allow list is one flat union.
_DSAGT_MCP_ALWAYS_ALLOW = [
    "add_skill_source",
    "get_registry",
    "http_request",
    "install_dependencies",
    "install_skill",
    "kb_append",
    "kb_get_memories",
    "kb_ingest",
    "kb_job_status",
    "kb_list_collections",
    "kb_remember",
    "kb_search",
    "list_skill_sources",
    "read_file",
    "reconstruct_pipeline",
    "run_command",
    "save_skill",
    "save_code_spec",
    "search_registry",
    "search_skills",
]


# ---------------------------------------------------------------------------
# Functional helpers (provider-agnostic)
# ---------------------------------------------------------------------------


def _mcp_server_args() -> list[str]:
    """Build the argv tail for ``uv run dsagt-server``.

    The single merged server reads all configuration from the project's
    ``.dsagt/config.yaml`` (located via cwd-walk) — no CLI args needed.
    """
    return ["run", "dsagt-server"]


def _mcp_env_block(config: dict) -> dict[str, str]:
    """Env vars the dsagt MCP server children need at startup.

    Benign routing only (no credentials, no provider redirection): the
    project name + dir, the serverless ``MLFLOW_TRACKING_URI``, and the
    embedding-backend settings.  MCP children run with cwd == project_dir
    and could read most of this from ``.dsagt/config.yaml``, but agents that
    don't inherit the parent's shell env into their MCP children (codex /
    cline) need it baked into the per-agent MCP config.  For
    ``backend: api`` the user still sets ``EMBEDDING_API_KEY`` in their shell
    (creds never on disk).

    No session id here — the MCP server mints it at startup into
    ``.dsagt/state.yaml`` (it owns the session lifecycle now), so there's
    nothing to thread through the env.
    """
    from dsagt.observability import resolve_tracking_uri

    emb = config.get("embedding") or {}
    block: dict[str, str] = {}
    for key, src in (
        ("DSAGT_PROJECT", config.get("project")),
        ("DSAGT_PROJECT_DIR", config.get("project_dir")),
        ("MLFLOW_TRACKING_URI", resolve_tracking_uri(config)),
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


#: Claude Code caps a skill's frontmatter description (combined with
#: when_to_use) at this many characters; longer ones are rejected.  We
#: truncate the *mirrored* copy only, never the project source.
_NATIVE_DESCRIPTION_CAP = 1536

#: Manifest filename inside a native skills dir listing the skill names
#: dsagt placed there, so the mirror can reap its own stale entries on
#: re-run without ever touching user-authored skills.
_SKILL_MANIFEST = ".dsagt-managed.json"


def _truncate_native_description(skill_md: Path) -> None:
    """If the mirrored SKILL.md's description exceeds the native cap, trim it."""
    import yaml

    text = skill_md.read_text()
    if not text.startswith("---"):
        return
    parts = text.split("---", 2)
    if len(parts) < 3:
        return
    try:
        front = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return
    desc = front.get("description")
    if isinstance(desc, str) and len(desc) > _NATIVE_DESCRIPTION_CAP:
        front["description"] = desc[: _NATIVE_DESCRIPTION_CAP - 1].rstrip() + "…"
        new_front = yaml.dump(front, default_flow_style=False, sort_keys=False)
        skill_md.write_text(f"---\n{new_front}---{parts[2]}")


def _mirror_skills_to(target_dir: Path, skill_dirs: list[Path]) -> list[str]:
    """Idempotently mirror *skill_dirs* into *target_dir* (e.g. .claude/skills).

    Copies each skill directory (SKILL.md + scripts/ + references/) under
    ``target_dir/<dir-name>/``.  A manifest tracks the names dsagt owns so a
    later run reaps skills that were removed upstream **without ever
    touching user-authored skills** that dsagt didn't place.  ``skill_dirs``
    should list bundled dirs before project dirs so a project skill wins a
    name collision (copied last).
    """
    actions: list[str] = []
    manifest_path = target_dir / _SKILL_MANIFEST
    previously: list[str] = []
    if manifest_path.exists():
        try:
            previously = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            previously = []

    target_dir.mkdir(parents=True, exist_ok=True)
    managed: list[str] = []
    for src in skill_dirs:
        if not (src / "SKILL.md").exists():
            continue
        name = src.name
        dest = target_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        _truncate_native_description(dest / "SKILL.md")
        if name not in managed:
            managed.append(name)

    # Reap skills dsagt placed previously that are gone from the source set.
    for stale in set(previously) - set(managed):
        stale_dir = target_dir / stale
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir, ignore_errors=True)

    manifest_path.write_text(json.dumps(sorted(managed), indent=2) + "\n")
    if managed:
        actions.append(f"Mirrored {len(managed)} skill(s) into {target_dir}")
    return actions


def _build_mcp_servers_dict(env_block: dict | None) -> dict:
    """Build the standard ``{"mcpServers": {...}}`` dict for the dsagt server.

    Used by agents that load MCP config from a JSON file.  Claude Code
    uses this shape via ``.mcp.json``
    but builds it inline in :class:`ClaudeSetup.write_dynamic`.  Cline
    doesn't use this — it requires ``cline mcp add`` to register the server.
    """
    entry: dict = {
        "command": "uv",
        "args": _mcp_server_args(),
        "disabled": False,
        "alwaysAllow": _DSAGT_MCP_ALWAYS_ALLOW,
    }
    if env_block:
        entry["env"] = env_block
    return {"mcpServers": {"dsagt": entry}}


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

    - ``name``           — agent identifier as used in ``.dsagt/config.yaml``
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

    #: Env vars the agent's runtime reads for provider credentials /
    #: endpoint routing.  Used by ``agent_env`` to surface a transparency
    #: note about what the agent will pick up from the user's shell: we
    #: list which of these are actually present so a missing or wrong
    #: value isn't silently consumed.  Empty for IDE-extension agents
    #: that never read env-var credentials.
    credential_env_vars: ClassVar[tuple[str, ...]] = ()

    #: Directory (relative to the working dir) the agent natively auto-discovers
    #: ``SKILL.md`` skill folders from.  ``setup_skills`` mirrors installed
    #: (bundled + project) skills AND registered codes here so the agent
    #: discovers/auto-invokes them without an MCP round-trip.  Every supported
    #: agent has one — claude ``.claude/skills``, codex/goose/opencode
    #: ``.agents/skills`` (the cross-agent standard), cline ``.cline/skills``.
    #: ``None`` would mean the agent has no native skill discovery
    #: (none currently).
    native_skills_dir: ClassVar[str | None] = None

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
        """Write the agent's runtime-dependent files (the per-agent MCP config).

        Caller must have built ``env`` via :func:`agent_env`.  Returns a
        list of one-line action descriptions.
        """

    def setup_skills(self, working_dir: Path, config: dict) -> list[str]:
        """Mirror installed skills AND registered codes into the agent's
        native skills dir so it auto-discovers/auto-invokes them.

        Codes share the skill-standard envelope (``codes/<name>/SKILL.md``),
        so the same copy serves both: native discovery puts a code's exact
        dsagt-run command in context at invocation time — a second discovery
        path alongside ``search_registry``, aimed at the from-memory
        command-reconstruction failure mode.

        Idempotent — the manifest-tracked :func:`_mirror_skills_to` only
        reaps skills dsagt placed, never user-authored ones.  No-op when the
        agent declares no ``native_skills_dir`` or ``skills.populate_native``
        is disabled.
        """
        if not self.native_skills_dir:
            return []
        if not (config.get("skills") or {}).get("populate_native", True):
            return []
        from dsagt.registry import CodeRegistry, SkillRegistry

        codes = CodeRegistry(runtime_dir=working_dir, kb=None)
        reg = SkillRegistry(runtime_dir=working_dir, kb=None)
        # Later entries win name collisions: codes first, then bundled
        # skills, then project skills — a deliberately installed instruction
        # skill outranks a registered code of the same name.
        src_dirs = (
            codes.code_dirs() + reg._bundled_skill_dirs() + reg._project_skill_dirs()
        )
        target = working_dir
        for part in self.native_skills_dir.split("/"):
            target = target / part
        return _mirror_skills_to(target, src_dirs)

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        """Files/dirs this agent's setup writes, for cleanup when a project
        re-inits onto a *different* agent platform.

        Lists the instruction file, the per-agent MCP-config file(s), and the
        agent's private per-project state dir(s) — NOT the shared
        ``.agents/`` skill-mirror dir (managed by the manifest reaper), and
        never project data (``.dsagt/``, ``kb_index/``, ``trace_archive/``,
        ``skills/``).  Paths may not all exist; the caller filters.

        Default = just the static marker; subclasses extend.
        """
        return [working_dir / self.static_marker]

    def runtime_env(self, config: dict) -> dict[str, str]:
        """Dsagt-owned env vars the agent process needs at runtime (BYOA).

        Default is empty: DSAGT no longer forces any telemetry env on the
        agent (agent traces are recovered post-hoc from the on-disk
        transcript, not by native OTel emission).  Subclasses override
        only to set per-project state-dir env (``CLINE_DIR``,
        ``CODEX_HOME``) that isolates their global config per project.

        Provider credentials (ANTHROPIC_*, OPENAI_*, GOOSE_*) are the
        user's responsibility — exported in their shell, never translated
        from ``config["llm"]``.
        """
        del config
        return {}

    def interactive_command(self, config: dict) -> list[str]:
        """Return the argv list for interactive launch.

        Default is ``base_command`` unmodified.  Goose overrides to append
        ``--with-extension`` flags for the dsagt MCP servers.
        """
        del config
        return list(self.base_command)

    #: Per-agent provider-credential hints surfaced by ``byoa_env_hints``.
    #: List of ``(env_var_name, hint)`` tuples shown to the user with a
    #: "skip if already configured" note.
    credential_hints: ClassVar[tuple[tuple[str, str], ...]] = ()

    def byoa_env_hints(self) -> list[tuple[str, str]]:
        """Provider-credential env vars the user should set in their shell.

        BYOA design: DSAGT writes no agent-affecting env (no OTel routing,
        no telemetry flags).  The user owns provider credentials —
        exported in their shell — and this returns one hint string each so
        ``dsagt init`` can remind them what their agent needs.
        """
        return list(self.credential_hints)

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
        currently have extensions that auto-discover dsagt's
        per-project files from the workspace root).

        ``dsagt init`` prints these lines under "Or with VS Code extension".
        """
        del project_dir
        return None
