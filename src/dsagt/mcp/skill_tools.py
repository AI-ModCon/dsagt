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
    from dsagt.agents import refresh_native_skills

    refresh_native_skills(skill_registry.runtime_dir)
    skill_count = len(skill_registry.list_skills())
    return (
        f"Skill '{spec['name']}' {action} successfully. "
        f"Registry now contains {skill_count} skills."
    )


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
    try:
        info = SkillRouter().install(name, runtime_dir)
    except LookupError as e:
        return f"Error: {e}"
    from dsagt.agents import refresh_native_skills

    refresh_native_skills(runtime_dir)

    # Bare confirmation by design: the install→use model and the
    # license/PROVENANCE capture are already in the agent's instructions and on
    # disk (PROVENANCE.txt), so repeating them on every install is just noise.
    verb = "Updated" if info["action"] == "updated" else "Installed"
    return f"{verb} '{info['name']}' → {info['dest_dir']}/"


async def _handle_add_skill_source(
    arguments: dict,
    *,
    kb: KnowledgeBase,
    runtime_dir: Path,
) -> dict:
    """Enable a skill source (known name or GitHub URL): clone + index the catalog."""
    from dsagt.skills import (
        KNOWN_SOURCES,
        SkillRouter,
        persist_source_to_config,
        resolve_source,
    )

    source = arguments.get("source")
    if not source:
        return {
            "error": "add_skill_source requires 'source' (known name or GitHub URL)."
        }
    try:
        spec = resolve_source(source)
        if isinstance(source, str) and source in KNOWN_SOURCES:
            spec.setdefault("name", source)
        router = SkillRouter(kb=kb)
        stats = await asyncio.to_thread(router.sync, source)
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
    flag + ``indexed`` count inline means the agent doesn't have to cross-
    reference a separate ``synced_collections`` list to tell the difference.
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
        "install_skill": partial(_handle_install_skill, runtime_dir=rt),
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
                            "skill directory.  Object mapping relative "
                            "path -> file contents, or JSON-encoded string."
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
            name="add_skill_source",
            description=(
                "Enable an external agent-skill source (a known name like "
                "'k-dense-ai'/'anthropic'/'antigravity'/'composio', or a GitHub URL). "
                "Clones it and indexes its skills into the searchable catalog "
                "(search_skills). Does NOT load them into context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Known source name or GitHub repo URL / owner/repo",
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
                "Search agent skills by name, tag, or description. Spans installed "
                "skills and the external installable catalog. Catalog hits are marked "
                "'[catalog]' — use install_skill to add one to this project."
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
                "no user action."
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
    ``call_tool_sync()``.  The merged ``dsagt-server`` uses
    :func:`_skill_tools_and_handlers` directly instead of this wrapper.
    """
    tools, handlers = _skill_tools_and_handlers(skill_registry, kb, runtime_dir)
    return build_dispatch_server(
        "skills", tools, handlers, {t: "skill" for t in handlers}
    )
