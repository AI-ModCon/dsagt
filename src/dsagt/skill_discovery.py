"""SkillRouter — the thin render/MCP facade over the skill *catalog*.

The catalog data plane (clone cache + ChromaDB collections + backend
selection + the four ops) lives in
:class:`dsagt.commands.skills_catalog.SkillsCatalog`.  ``SkillRouter`` adds only
the presentation concerns that the MCP handlers and the CLI share: rendering a
ranked hit list into the ``search_skills`` string, the empty-result message, and
the exact-``skill_name`` lookup (which needs the installed-skill registry, not
the catalog).

Construct it from the same inputs at every call site (MCP ``search_skills``, CLI
``skills search/list``) so policy can't diverge between them — or hand it a
prebuilt :class:`SkillsCatalog` via ``catalog=`` so a server that already owns a
shared KB reuses one catalog instance.

Skill *materialization* (mirroring installed skills into each agent's native
skills directory) lives in the agent layer (``AgentSetup.setup_skills``), not
here: every supported agent (claude/codex/goose/cline/roo) natively
auto-discovers ``SKILL.md`` folders, so there is no agent-facing disclosure tier
for the router to own.  ``search_skills`` exists for the *catalog* tier (skills
not yet installed, which native discovery can't see) plus the no-embedder
keyword fallback.

See ``design-notes/genesis-skills-comparison.md`` §7 and
``design-notes/skills-catalog-server-merge.md`` §1.
"""

from __future__ import annotations

import logging

from dsagt.commands.skills_catalog import SkillsCatalog

logger = logging.getLogger(__name__)


def _where_label(source: str) -> str:
    """Human tag for a hit's origin, matching the legacy search output."""
    if source in ("bundled", "registered", "installed"):
        return " [installed]"
    if source.startswith("catalog:"):
        return " [catalog · install_skill to add]"
    return ""


class SkillRouter:
    """Renders catalog discovery for the MCP ``search_skills`` tool + the CLI."""

    def __init__(self, *, kb=None, skill_registry=None, cache_dir=None, catalog=None):
        self._catalog = (
            catalog
            if catalog is not None
            else SkillsCatalog(kb=kb, cache_dir=cache_dir)
        )
        self._reg = skill_registry

    # -- rendering -----------------------------------------------------------

    def _render_search(self, hits: list[dict]) -> str:
        lines = []
        for h in hits:
            lines.append(
                f"- **{h['name']}**{_where_label(h['source'])} "
                f"(score: {h['score']:.2f})\n  {h['summary']}"
            )
        return f"Found {len(hits)} skill(s):\n\n" + "\n\n".join(lines)

    def _empty_message(self) -> str:
        if not self._catalog.synced_collections() and self._catalog.has_kb:
            return (
                "No catalog skills found. No external skill catalog is synced "
                "yet — search covers the catalog (skills you can install), since "
                "installed skills are already natively discoverable. Call "
                "list_skill_sources() to see available sources, then "
                "add_skill_source(source=...) to sync one before searching again."
            )
        return "No catalog skills found matching the query."

    # -- public API ----------------------------------------------------------

    def search(self, query=None, *, top_k: int = 8, tag=None, skill_name=None) -> str:
        """Stage B. Select + render. Stateless — no session/exposure tracking."""
        if skill_name:
            import yaml

            if self._reg is None:
                return f"No skill named '{skill_name}'."
            spec = self._reg.get_skill(skill_name)
            if spec:
                return f"Found skill '{skill_name}':\n\n" + yaml.dump(
                    spec, default_flow_style=False, sort_keys=False
                )
            return f"No skill named '{skill_name}'."

        hits = self._catalog.search(query, top_k=top_k, tag=tag)
        if not hits:
            return self._empty_message()
        return self._render_search(hits)

    def sync(self, source, *, force: bool = False) -> dict:
        """Stage A passthrough — see :meth:`SkillsCatalog.sync`."""
        return self._catalog.sync(source, force=force)

    def install(self, name: str, project_dir) -> dict:
        """Stage C passthrough — see :meth:`SkillsCatalog.install`."""
        return self._catalog.install(name, project_dir)

    def list_sources(self) -> list[dict]:
        """Stage A view passthrough — see :meth:`SkillsCatalog.list_sources`."""
        return self._catalog.list_sources()
