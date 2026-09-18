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
import os
import re
import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

# Master instructions ship with the package, one directory above this file.
_INSTRUCTIONS_PATH = Path(__file__).parent.parent / "dsagt_instructions.md"

# The dsagt instructions sit between these two lines in every per-agent
# instructions file (CLAUDE.md, AGENTS.md, .goosehints,
# .clinerules/dsagt_instructions.md).  ``_write_dsagt_block`` replaces the
# text between them on every init and start, so a changed setting or an
# upgraded dsagt reaches the agent, and keeps whatever the user wrote
# outside them.
_BLOCK_BEGIN = "<!-- dsagt:begin -->"
_BLOCK_END = "<!-- dsagt:end -->"

# Tools the dsagt MCP server exposes — listed in ``alwaysAllow`` so cline
# auto-approves them without a human-in-the-loop prompt.  Keep in
# sync with the ``mcp/*_tools.py`` tool registrations (registry / knowledge /
# memory / skill); a tool added there but not here means cline will hang on
# its first call.  All dsagt MCP tools live behind the single ``dsagt-server``,
# so the always-allow list is one flat union.
_DSAGT_MCP_ALWAYS_ALLOW = [
    "add_skill_source",
    "get_registry",
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
    "readiness_reports",
    "reconstruct_pipeline",
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


#: The launching shell's variables copied into the MCP env block: what an
#: activated environment sets (a venv, conda, ``module load``) and what a
#: compiled dependency needs to load.  Codex and cline start the server from
#: the block alone, so without these the server's python lacked the user's
#: packages.
_SHELL_ENV_PASSTHROUGH = (
    "PATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "MODULEPATH",
    "LOADEDMODULES",
    "_LMFILES_",
)

#: A name matching one of these never enters the block, whatever a config
#: lists: the block is written into the project's agent config, which is a
#: file, and dsagt writes no credential into a file.
_CREDENTIAL_NAME = re.compile(r"_KEY$|_TOKEN$|_SECRET$|SECRET|PASSW", re.I)


def _mcp_env_block(
    config: dict, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Env vars the dsagt MCP server children need at startup.

    Two kinds. Routing: the project name + dir, the resolved
    ``MLFLOW_TRACKING_URI``, and the embedding-backend settings, which MCP
    children could read from ``.dsagt/config.yaml`` but which codex and
    cline, whose children see only this block, need baked in.  The launching
    shell's environment: :data:`_SHELL_ENV_PASSTHROUGH` plus the names the
    config lists under ``mcp.env_passthrough`` for site-specific ones, copied
    from *environ* (the process environment by default) at every ``dsagt
    init`` and ``dsagt start``, so the server's python is the user's
    activated one; a bare launch after a changed activation needs one of the
    two.  Credentials are never part of it: a name matching
    :data:`_CREDENTIAL_NAME` is refused with ``ValueError``, and
    ``EMBEDDING_API_KEY`` and the trace store's key come from the shell or
    ``~/.config/dsagt/env`` (``session.load_user_env``).

    No session id here — the MCP server mints it at startup into
    ``.dsagt/state.yaml`` (it owns the session lifecycle now), so there's
    nothing to thread through the env.
    """
    from dsagt.observability import resolve_tracking_uri

    environ = os.environ if environ is None else environ
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
    extra = (config.get("mcp") or {}).get("env_passthrough") or []
    for name in extra:
        if _CREDENTIAL_NAME.search(name):
            raise ValueError(
                f"mcp.env_passthrough names {name!r}, a credential; the MCP env "
                "block is written into the agent config, and a credential is read "
                "from the shell or ~/.config/dsagt/env instead"
            )
    for name in (*_SHELL_ENV_PASSTHROUGH, *extra):
        value = environ.get(name)
        if value:
            block[name] = value
    return block


_READINESS_MARKER = "<!-- readiness-check -->\n"


def _load_master_instructions(auto_assess: bool = True) -> str | None:
    """Load the master DSAgt instructions, or None if the file is missing.

    The per-operation check rule carries a marker line; with *auto_assess*
    the AI-readiness paragraph replaces it, otherwise the line is dropped.
    """
    if not _INSTRUCTIONS_PATH.exists():
        logger.warning("Master instructions not found: %s", _INSTRUCTIONS_PATH)
        return None
    from dsagt.readiness import INSTRUCTIONS_PARAGRAPH

    text = _INSTRUCTIONS_PATH.read_text()
    if _READINESS_MARKER not in text:
        raise RuntimeError(f"{_INSTRUCTIONS_PATH} has no readiness-check marker")
    filler = INSTRUCTIONS_PARAGRAPH + "\n" if auto_assess else ""
    return text.replace(_READINESS_MARKER, filler)


def _write_dsagt_block(path: Path, content: str) -> str | None:
    """Write *content* as the dsagt block of the instructions file at *path*.

    The block is the text from the begin marker line through the end marker
    line.  A missing file becomes the block; a file without the markers gets
    the block appended after its own text; a file with the markers has the
    block replaced.  Text before or after the block is the user's and is
    kept.  Returns a one-line action description, or None when the file
    already holds this block.
    """
    block = f"{_BLOCK_BEGIN}\n{content.rstrip()}\n{_BLOCK_END}\n"
    if not path.exists():
        path.write_text(block)
        return f"Wrote {path}"
    existing = path.read_text()
    begin = existing.find(_BLOCK_BEGIN)
    end = existing.find(_BLOCK_END, begin + 1) if begin != -1 else -1
    if end == -1:
        path.write_text(existing.rstrip("\n") + "\n\n" + block)
        return f"Appended DSAgt instructions to {path}"
    end += len(_BLOCK_END)
    if existing[end : end + 1] == "\n":
        end += 1
    updated = existing[:begin] + block + existing[end:]
    if updated == existing:
        return None
    path.write_text(updated)
    return f"Updated DSAgt instructions in {path}"


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


def _description_fits_native_cap(skill_md: Path) -> bool:
    import yaml

    text = skill_md.read_text()
    parts = text.split("---", 2)
    if not text.startswith("---") or len(parts) < 3:
        return True
    try:
        front = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return True
    desc = front.get("description")
    return not (isinstance(desc, str) and len(desc) > _NATIVE_DESCRIPTION_CAP)


def _remove_mirror_entry(dest: Path) -> None:
    if dest.is_symlink():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)


def _mirror_skills_to(target_dir: Path, skill_dirs: list[Path]) -> list[str]:
    """Idempotently mirror *skill_dirs* into *target_dir* (e.g. .claude/skills).

    Links each skill directory (SKILL.md + scripts/ + references/) at
    ``target_dir/<dir-name>``, a relative symlink, so the agent reads the
    live files and an edit to a skill's script or SKILL.md under ``skills/``
    is what the next invocation sees; Claude Code and Codex both follow the
    link.  A skill whose description exceeds the native cap is copied instead
    and the copy's description truncated, since a link cannot be trimmed.  A
    manifest tracks the names dsagt owns so a later run reaps skills that
    were removed upstream **without ever touching user-authored skills**
    that dsagt didn't place.  ``skill_dirs`` should list bundled dirs before
    project dirs so a project skill wins a name collision (placed last).
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
        _remove_mirror_entry(dest)
        if _description_fits_native_cap(src / "SKILL.md"):
            dest.symlink_to(os.path.relpath(src.resolve(), target_dir.resolve()))
        else:
            shutil.copytree(src, dest)
            _truncate_native_description(dest / "SKILL.md")
        if name not in managed:
            managed.append(name)

    # Reap skills dsagt placed on an earlier run that are gone from the source set.
    for stale in set(previously) - set(managed):
        _remove_mirror_entry(target_dir / stale)

    manifest_path.write_text(json.dumps(sorted(managed), indent=2) + "\n")
    if managed:
        actions.append(f"Mirrored {len(managed)} skill(s) into {target_dir}")
    return actions


def _build_mcp_servers_dict(env_block: dict | None) -> dict:
    """Build the standard ``{"mcpServers": {...}}`` dict for the dsagt server.

    Used by agents that load MCP config from a JSON file.  Claude Code
    uses this shape via ``.mcp.json``
    but builds it inline in :class:`ClaudeSetup.write_dynamic`.  Cline
    registers the server through ``cline mcp add``.
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
    def write_static(self, working_dir: Path, *, auto_assess: bool = True) -> list[str]:
        """Write the agent's instructions file + any state directories.

        Idempotent: if the dsagt marker is already in the instructions
        file, the write is skipped (preserves user edits).  *auto_assess*
        selects whether the instructions carry the AI-readiness check
        paragraph.  Returns a list of one-line action descriptions.
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

        Codes share the skill-standard envelope and the ``skills/`` directory,
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
        from dsagt.registry import SkillRegistry

        src_dirs = SkillRegistry(runtime_dir=working_dir, kb=None).skill_dirs()
        target = working_dir
        for part in self.native_skills_dir.split("/"):
            target = target / part
        return _mirror_skills_to(target, src_dirs)

    def owned_artifacts(self, working_dir: Path) -> list[Path]:
        """Files and dirs this agent's setup writes, for cleanup when a project
        re-inits onto a different agent platform.

        Lists the instruction file, the per-agent MCP-config files, and the
        agent's private per-project state dirs.  The shared ``.agents/``
        skill-mirror dir is managed by the manifest, and project data
        (``.dsagt/``, ``kb_index/``, ``trace_archive/``, ``skills/``) is
        never listed.  Paths may not all exist; the caller filters.

        Default is the static marker alone; subclasses extend.
        """
        return [working_dir / self.static_marker]

    def runtime_env(self, config: dict) -> dict[str, str]:
        """Dsagt-owned env vars the agent process needs at runtime.

        Default is empty: agent traces are recovered from the on-disk
        transcript, so the agent's environment needs no telemetry setting.
        Subclasses override only to set per-project state-dir env
        (``CLINE_MCP_SETTINGS_PATH``, ``CODEX_HOME``) that isolates their global config
        per project.

        LLM-provider credentials (ANTHROPIC_*, OPENAI_*, GOOSE_*) are the
        user's responsibility, exported in their shell; dsagt never reads
        or translates them.  DSAgt's own service credentials (trace store,
        embedding backend) are handled by ``_mcp_env_block``.
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
        extension.  Returns ``None`` unless the agent's extension
        auto-discovers dsagt's per-project files from the workspace root,
        which claude's does.

        ``dsagt init`` prints these lines under "Or with VS Code extension".
        """
        del project_dir
        return None
