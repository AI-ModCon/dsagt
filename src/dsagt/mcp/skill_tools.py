"""MCP tools for skill discovery, install, and catalog sources.

The full skill surface of ``dsagt-server``, consolidated from the two former
servers: register a project skill (``save_skill``), enable + list external
catalog sources (``add_skill_source`` / ``list_skill_sources``), search the
catalog (``search_skills``), and install a catalog skill into the project
(``install_skill``).  The catalog data plane + router live in
:mod:`dsagt.skills`; these handlers are the thin MCP wiring over it.

Installed/created skills are natively auto-discovered by every supported agent,
so ``search_skills`` covers only the not-yet-installed *catalog* tier (plus the
no-embedder keyword fallback).

These definitions + handlers run inside the merged ``dsagt-server`` (see
:mod:`dsagt.mcp.server`); ``create_skill_server`` is retained only as a
test-facing constructor.
"""

import asyncio
import json
import logging
from functools import partial
from pathlib import Path

import mcp.types as types

from dsagt.knowledge import KnowledgeBase
from dsagt.mcp.server import build_dispatch_server
from dsagt.registry import SkillRegistry
from dsagt.agents import refresh_native_skills
from dsagt.skills import register_skill_scripts

logger = logging.getLogger(__name__)


async def _handle_save_skill(
    arguments: dict,
    *,
    skill_registry: SkillRegistry,
) -> str:
    """Register a skill (workflow / agent instructions) for later reuse.

    Writes SKILL.md to ``<project>/skills/<name>/`` and mirrors it into the
    agent's native skills dir immediately, where every supported agent
    auto-discovers it from its next session.  No KB indexing —
    ``search_skills`` covers only the not-yet-installed *catalog* tier, since
    installed skills are already natively discoverable.
    """
    spec = arguments["spec"]
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            return f"Error: spec must be a JSON object (or string-encoded JSON object): {e}"
    name = spec.get("name") if isinstance(spec, dict) else None
    if name:
        installed = skill_registry.skills_dir / name / "PROVENANCE.txt"
        if installed.exists():
            return (
                f"'{name}' was installed from a catalog and was not changed; "
                f"{installed.parent.name}/PROVENANCE.txt credits its source, and "
                "overwriting it would lose what the credit refers to. Save under "
                "another name, or call delete_skill first to replace it."
            )
    body = arguments.get("body")
    reference_files = arguments.get("reference_files")
    if isinstance(reference_files, str):
        try:
            reference_files = json.loads(reference_files)
        except json.JSONDecodeError as e:
            return f"Error: reference_files must be a JSON object: {e}"
    try:
        action = skill_registry.save_skill(
            spec, body=body, reference_files=reference_files
        )
    except (KeyError, ValueError, OSError) as e:
        return f"Error saving skill: {e}"
    # The skill's scripts are codes from this moment, the same path a base
    # or catalog skill takes, and the reply gives the stored lines the way
    # save_code_spec does, so the usage line the agent just wrote is not
    # what it runs.
    try:
        stored = register_skill_scripts(
            skill_registry.runtime_dir, spec["name"], kb=skill_registry._kb
        )
        refresh_native_skills(skill_registry.runtime_dir)
    except (KeyError, ValueError, OSError) as e:
        # The skill is already written, so the reply says what stands rather
        # than reporting a failure that leaves an unexplained directory.
        return (
            f"Skill '{spec['name']}' was written to skills/{spec['name']}/, but "
            f"registering its scripts as codes failed: {e}. Its scripts are not "
            "runnable through dsagt-run. Fix what the error names and save it "
            "again, or call delete_skill to remove the skill and start over."
        )
    skill_count = len(skill_registry.list_skills())
    reply = (
        f"Skill '{spec['name']}' {action} successfully. "
        f"Registry now contains {skill_count} skills."
    )
    if stored:
        reply += " Its scripts are registered codes; run each as:\n" + "\n".join(
            f"Run it as: {line}" for line in stored
        )
    return reply


async def _handle_delete_skill(
    arguments: dict,
    *,
    skill_registry: SkillRegistry,
    kb=None,
) -> str:
    """Remove an installed skill or code from the project.

    One path for both, since a code is a skill directory whose frontmatter
    declares an executable.
    """
    from dsagt.skills import delete_skill

    name = arguments["name"]
    try:
        result = delete_skill(skill_registry.runtime_dir, name, kb=kb)
    except FileNotFoundError:
        return f"Error: no installed skill or code named '{name}'."
    except OSError as e:
        return f"Error deleting '{name}': {e}"
    what = "code" if result["was_code"] else "skill"
    reply = f"Deleted {what} '{name}' and its entry in the agent's skills dir."
    if result["entries_removed"]:
        reply += (
            f" Removed {result['entries_removed']} knowledge-base entry"
            f"{'' if result['entries_removed'] == 1 else 'ies'}; search_registry "
            "no longer returns it."
        )
    return reply


async def _handle_search_skills(
    arguments: dict,
    *,
    kb: KnowledgeBase | None,
    skill_registry: SkillRegistry | None,
) -> str:
    if skill_registry is None:
        return "search_skills is unavailable (no skill registry configured)."

    from dsagt.skills import SkillRouter

    router = SkillRouter(skill_registry=skill_registry, kb=kb)
    return router.search(
        arguments.get("query", ""),
        top_k=arguments.get("top_k", 10),
        tag=arguments.get("tag"),
        skill_name=arguments.get("skill_name"),
    )


async def _handle_install_skill(
    arguments: dict,
    *,
    runtime_dir: Path,
    kb: KnowledgeBase | None = None,
) -> str:
    """Install a catalog skill into ``<project>/skills/<name>/``.

    The skill's files land on disk and are mirrored into the agent's native
    skills dir immediately — usable right away (the agent reads/follows its
    SKILL.md, which is all native invocation does); hands-free auto-discovery
    kicks in at the agent's next session, with no user action.
    """
    from dsagt.skills import SkillRouter

    name = arguments.get("skill_name")
    if not name:
        return "install_skill requires 'skill_name'."
    installed = Path(runtime_dir) / "skills" / name
    if installed.exists():
        # Edits to an installed skill win, the same rule the base-skill
        # install follows; a re-copy would overwrite them.
        return (
            f"'{name}' is already installed at {installed}/ and was left as it "
            "is; read and follow its SKILL.md."
        )
    try:
        info = SkillRouter().install(name, runtime_dir)
    except LookupError as e:
        return f"Error: {e}"
    try:
        stored = register_skill_scripts(runtime_dir, info["name"], kb=kb)
        refresh_native_skills(runtime_dir)
    except (KeyError, ValueError, OSError) as e:
        return (
            f"'{info['name']}' was copied to {info['dest_dir']}/, but registering "
            f"its scripts as codes failed: {e}. Its scripts are not runnable "
            "through dsagt-run. Fix what the error names and install it again, or "
            "call delete_skill to remove it."
        )

    # Bare confirmation by design: the install→use model and the
    # license/PROVENANCE capture are already in the agent's instructions and on
    # disk (PROVENANCE.txt), so repeating them on every install is just noise.
    verb = "Updated" if info["action"] == "updated" else "Installed"
    reply = f"{verb} '{info['name']}' → {info['dest_dir']}/"
    if stored:
        reply += " Its scripts are registered codes; run each as:\n" + "\n".join(
            f"Run it as: {line}" for line in stored
        )
    return reply


async def _handle_add_skill_source(
    arguments: dict,
    *,
    kb: KnowledgeBase,
    runtime_dir: Path,
) -> dict:
    """Enable a skill source (known name or git URL): clone + index the catalog.

    ``force`` re-clones a cached source so skills added upstream since the
    first sync are indexed; without it a cached clone is only re-indexed.
    """
    from dsagt.skills import (
        KNOWN_SOURCES,
        SkillRouter,
        persist_source_to_config,
        resolve_source,
    )

    source = arguments.get("source")
    if not source:
        return {"error": "add_skill_source requires 'source' (known name or git URL)."}
    try:
        spec = resolve_source(source)
        if isinstance(source, str) and source in KNOWN_SOURCES:
            spec.setdefault("name", source)
        router = SkillRouter(kb=kb)
        force = bool(arguments.get("force", False))
        stats = await asyncio.to_thread(router.sync, source, force=force)
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}
    persist_source_to_config(
        runtime_dir, {"name": spec.get("name", stats["slug"]), **spec}
    )
    return {
        "source": spec["url"],
        "slug": stats["slug"],
        "skills_indexed": stats["indexed"],
        "note": "Searchable via search_skills; install one with install_skill.",
    }


async def _handle_list_skill_sources(arguments: dict, *, kb: KnowledgeBase) -> dict:
    """List known skill sources, each flagged synced/available with its count.

    A source is ``synced`` (searchable via ``search_skills``) only after an
    ``add_skill_source`` call has cloned + indexed it; otherwise it is
    ``available`` (known name + URL, nothing indexed yet).  Reporting the
    flag + ``indexed`` count inline lets the agent tell the two apart from
    this one list.
    """
    from dsagt.registry import CATALOG_COLLECTION_PREFIX, catalog_collection
    from dsagt.skills import KNOWN_SOURCES, SkillRouter, _repo_slug

    synced = {c for c in kb.collections if c.startswith(CATALOG_COLLECTION_PREFIX)}

    # Single source of truth for the per-source synced/indexed view (shared
    # with the CLI `skills list --catalog`).
    sources = {
        s["name"]: {
            "url": s["url"],
            "description": s["description"],
            "synced": s["synced"],
            "indexed": s["indexed"],
        }
        for s in SkillRouter(kb=kb).list_sources()
    }

    # Surface any synced catalog whose source isn't in KNOWN_SOURCES (added
    # by raw GitHub URL) so the count is never silently dropped.
    known_colls = {
        catalog_collection(_repo_slug(s["url"])) for s in KNOWN_SOURCES.values()
    }
    extra = sorted(synced - known_colls)

    any_synced = any(v["synced"] for v in sources.values()) or bool(extra)
    return {
        "sources": sources,
        "other_synced_collections": extra,
        "note": (
            "add_skill_source <name|url> to sync a source whose synced=false; "
            "then search_skills to browse. search_skills only sees synced sources."
            if any_synced
            else "No catalog synced yet — add_skill_source <name|url> "
            "(e.g. 'k-dense-ai') to enable one, then search_skills to browse."
        ),
    }


# ---------------------------------------------------------------------------
# Tool defs + handler map (used by the merged server and the test wrapper)
# ---------------------------------------------------------------------------


def _skill_tools_and_handlers(
    skill_registry: SkillRegistry | None,
    kb: KnowledgeBase | None = None,
    runtime_dir: str | Path | None = None,
):
    """Build the skill ``(tool defs, handler map)``.

    Combined with the other concern modules' tools under one MCP ``Server`` by
    :func:`dsagt.mcp.server.create_dsagt_server`.  ``runtime_dir`` (the project
    dir, where skills install + sources persist) falls back to the skill
    registry's ``runtime_dir`` then the KB index's parent.
    """
    rt: Path | None = Path(runtime_dir) if runtime_dir else None
    if rt is None and skill_registry is not None:
        rt = Path(skill_registry.runtime_dir)
    if rt is None and kb is not None:
        rt = kb.index_dir.parent

    handlers = {
        "save_skill": partial(_handle_save_skill, skill_registry=skill_registry),
        "search_skills": partial(
            _handle_search_skills, kb=kb, skill_registry=skill_registry
        ),
        "install_skill": partial(_handle_install_skill, runtime_dir=rt, kb=kb),
        "delete_skill": partial(
            _handle_delete_skill, skill_registry=skill_registry, kb=kb
        ),
        "add_skill_source": partial(_handle_add_skill_source, kb=kb, runtime_dir=rt),
        "list_skill_sources": partial(_handle_list_skill_sources, kb=kb),
    }

    tools = [
        types.Tool(
            name="save_skill",
            description=(
                "Register a skill (agent workflow / instructions) into "
                "<project>/skills/<name>/SKILL.md, mirrored into the agent's "
                "native skills dir immediately so future sessions "
                "auto-discover it — no restart or user action needed.  "
                "Symmetric with save_code_spec — use this when you've "
                "designed a reusable instruction set you want future "
                "sessions to load automatically."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    # ``anyOf`` for spec mirrors save_code_spec — accept
                    # both structured object and JSON-encoded string for
                    # MCP clients that serialize nested args.
                    "spec": {
                        "description": "Skill spec (object or JSON-encoded string)",
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Unique skill identifier (becomes the directory name)",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "What the skill does / when to use it",
                                    },
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Tags for categorizing the skill",
                                    },
                                },
                                "required": ["name", "description"],
                            },
                            {"type": "string"},
                        ],
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Markdown body of the SKILL.md (workflow / "
                            "instructions the agent will follow).  When "
                            "updating an existing skill, omit to preserve "
                            "the existing body."
                        ),
                    },
                    "reference_files": {
                        "description": (
                            "Optional additional files to write into the "
                            "skill directory.  Object mapping a path inside "
                            "the skill directory -> file contents, or a "
                            "JSON-encoded string.  Skill-standard layout: "
                            "reference documents under references/ (for "
                            "example 'references/spec.md') and scripts under "
                            "scripts/ (for example 'scripts/convert.py'); a "
                            "bare filename lands at the skill root."
                        ),
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            {"type": "string"},
                        ],
                    },
                },
                "required": ["spec"],
            },
        ),
        types.Tool(
            name="delete_skill",
            description=(
                "Remove an installed skill or registered code from the "
                "project: its directory under skills/, its entry in the "
                "agent's native skills dir, and, for a code, its "
                "knowledge-base entry. Use it to drop something installed by "
                "mistake, or to replace a skill installed from a catalog, "
                "which save_skill will not overwrite."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Name of the skill or code, as its directory "
                            "under skills/ is named"
                        ),
                    }
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="add_skill_source",
            description=(
                "Enable an external agent-skill source (a known name: "
                "'genesis', 'k-dense-ai', 'anthropic', 'antigravity', "
                "'composio'; or a git URL on any host). "
                "Clones it and indexes its skills into the searchable catalog "
                "(search_skills). Does NOT load them into context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Known source name or git repo URL / owner/repo",
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Re-clone a source that is already cached, picking up "
                            "skills added upstream since the last sync. Default false."
                        ),
                    },
                },
                "required": ["source"],
            },
        ),
        types.Tool(
            name="list_skill_sources",
            description="List known + synced external skill sources and their indexed catalogs.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="search_skills",
            description=(
                "Search the synced external skill catalogs by query or tag: the "
                "skills this project can install, each hit marked "
                "'[catalog · install_skill to add]'. The agent discovers an "
                "installed skill natively, and 'skill_name' returns that skill's "
                "spec from <project>/skills/."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "tag": {"type": "string", "description": "Filter by tag"},
                    "skill_name": {
                        "type": "string",
                        "description": "Exact skill name lookup",
                    },
                    "top_k": {"type": "integer", "default": 10},
                },
            },
        ),
        types.Tool(
            name="install_skill",
            description=(
                "Install a skill from the external catalog (found via search_skills) "
                "into this project. Copies SKILL.md + scripts/references and mirrors "
                "it into the agent's native skills dir — usable immediately (read and "
                "follow its SKILL.md); future sessions auto-discover it natively with "
                "no user action. A skill already in <project>/skills/ (a base skill, "
                "or one installed earlier) is left as it is."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Catalog skill name to install",
                    },
                },
                "required": ["skill_name"],
            },
        ),
    ]
    return tools, handlers


def create_skill_server(
    skill_registry: SkillRegistry | None = None,
    kb: KnowledgeBase | None = None,
    runtime_dir: str | Path | None = None,
):
    """Create a standalone MCP server exposing only the skill tools.

    Test-facing API: tests call it with mock deps and drive the server via
    ``call_tool_sync()``.  The merged ``dsagt-server`` composes
    :func:`_skill_tools_and_handlers` itself.
    """
    tools, handlers = _skill_tools_and_handlers(skill_registry, kb, runtime_dir)
    return build_dispatch_server(
        "skills", tools, handlers, {t: "skill" for t in handlers}
    )
