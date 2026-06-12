"""
External skill catalog — fetch Agent-Skills repos, index, install.

Two tiers (see the skill-management plan):

* **Catalog** — every skill in a configured GitHub source repo, indexed
  into a per-source ``skills_catalog__<slug>`` KB collection.  Searchable
  via ``search_skills``, but NOT copied locally and NOT loaded into the
  agent's context.  This is the one job native skill discovery can't do
  (you can't hold thousands of skill descriptions in context).
* **Installed** — a chosen skill copied into ``<project>/skills/<name>/``.
  The agent setup then mirrors it into ``.claude/skills/`` for native
  discovery (see ``agents.base._mirror_skills_to``).

Re-sync is idempotent by dropping the per-source collection directory and
rebuilding it — no delete-by-metadata primitive required.

``clone_github`` is imported lazily inside :func:`sync_source` to avoid an
import cycle with ``setup_core_kb`` (which calls back into ``sync_source``).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from dsagt.registry import _parse_frontmatter, catalog_collection
from dsagt.session import REGISTRY_DIR

logger = logging.getLogger(__name__)

#: Default source enabled out of the box (matches dsagt_config.yaml default).
DEFAULT_SOURCE = "scientific"

#: Curated, named skill sources.  ``subdir`` scopes the recursive SKILL.md
#: walk when set (cheaper clone); when omitted the whole repo is cloned and
#: walked, which is robust to category-nested layouts.
KNOWN_SOURCES: dict[str, dict] = {
    "scientific": {
        "url": "https://github.com/K-Dense-AI/scientific-agent-skills",
        "branch": "main",
        "subdir": "skills",
        "description": "K-Dense scientific agent skills — chem/bio/medicine/materials (140+).",
    },
    "anthropic": {
        "url": "https://github.com/anthropics/skills",
        "branch": "main",
        "subdir": "skills",
        "description": "Official Anthropic skills + document-editing examples.",
    },
    "antigravity": {
        "url": "https://github.com/sickn33/antigravity-awesome-skills",
        "branch": "main",
        "subdir": None,
        "description": "Antigravity Awesome Skills — 1,500+ cross-platform agentic skills.",
    },
    "composio": {
        "url": "https://github.com/ComposioHQ/awesome-claude-skills",
        "branch": "master",
        "subdir": None,
        "description": "Composio awesome-claude-skills — workflow skills for many SaaS apps.",
    },
}

#: Shared, machine-global cache of cloned source repos (sibling of kb_index/).
SKILL_SOURCES_DIR = REGISTRY_DIR / ".skill_sources"


# ---------------------------------------------------------------------------
# Source resolution + slugging
# ---------------------------------------------------------------------------


def resolve_source(source: str | dict) -> dict:
    """Resolve a known-source name, a GitHub URL, or a full spec dict.

    Returns a dict with at least ``url``; optional ``branch`` / ``subdir``.
    """
    if isinstance(source, dict):
        if not source.get("url"):
            raise ValueError("source dict must include a 'url'")
        return source
    if source in KNOWN_SOURCES:
        return dict(KNOWN_SOURCES[source])
    if source.startswith(("http://", "https://", "git@")) or source.count("/") == 1:
        # Full URL or ``owner/repo`` shorthand.
        url = (
            source
            if "://" in source or source.startswith("git@")
            else f"https://github.com/{source}"
        )
        return {"url": url, "branch": "main", "subdir": None}
    raise ValueError(
        f"Unknown skill source '{source}'. Use a known name "
        f"({', '.join(sorted(KNOWN_SOURCES))}), a GitHub URL, or owner/repo."
    )


def persist_source_to_config(project_dir: str | Path, spec: dict) -> bool:
    """Append a resolved source to ``skills.sources`` in the project config.

    Dedupes by URL.  Returns True if the config was updated.  No-op (returns
    False) if the config file is missing — the catalog is still indexed
    either way.  Used by both the ``add_skill_source`` MCP tool and the
    ``dsagt skills add`` CLI so a CLI-added source is re-synced by a later
    config-driven ``dsagt skills sync``.
    """
    import yaml

    cfg_path = Path(project_dir) / "dsagt_config.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    sources = cfg.setdefault("skills", {}).setdefault("sources", [])
    if any(s.get("url") == spec.get("url") for s in sources):
        return False
    sources.append(
        {k: spec[k] for k in ("name", "url", "branch", "subdir") if k in spec}
    )
    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    return True


def _repo_slug(url: str) -> str:
    """Stable, collection-name-safe slug from a GitHub URL (``owner-repo``)."""
    s = url.rstrip("/")
    s = re.sub(r"^https?://github\.com/", "", s)
    s = re.sub(r"^git@github\.com:", "", s)
    s = re.sub(r"\.git$", "", s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_skill_dirs(root: Path) -> list[Path]:
    """Recursively find skill directories (any dir holding a parseable SKILL.md).

    Recursive so both flat (``skills/<name>/SKILL.md``) and category-nested
    (``skills/<domain>/<name>/SKILL.md``) repo layouts work.  A directory
    qualifies only if its SKILL.md has YAML frontmatter with a ``name``.
    """
    out: list[Path] = []
    if not root.exists():
        return out
    for skill_md in sorted(root.rglob("SKILL.md")):
        try:
            spec = _parse_frontmatter(skill_md)
        except ValueError as e:  # malformed frontmatter — skip, don't abort
            logger.warning("skipping %s: %s", skill_md, e)
            continue
        if spec.get("name"):
            out.append(skill_md.parent)
    return out


# ---------------------------------------------------------------------------
# Sync (clone + index)
# ---------------------------------------------------------------------------


def sync_source(
    source: str | dict,
    *,
    kb=None,
    cache_dir: Path = SKILL_SOURCES_DIR,
    force: bool = False,
) -> dict:
    """Clone *source* into the cache and (re)index its skills into the catalog.

    ``force`` re-clones from scratch.  Indexing wipes and rebuilds only this
    source's ``skills_catalog__<slug>`` collection, so other catalogs and the
    installed/bundled ``skills`` collection are untouched.  When *kb* is None
    the clone still happens (so ``install`` works offline-of-KB) but nothing
    is indexed.
    """
    spec = resolve_source(source)
    slug = _repo_slug(spec["url"])
    dest = cache_dir / slug

    if force and dest.exists():
        shutil.rmtree(dest)
    if not dest.exists():
        from dsagt.commands.setup_core_kb import clone_github  # lazy: break cycle

        dest.mkdir(parents=True, exist_ok=True)
        subdir = spec.get("subdir")
        include = [subdir] if subdir else None
        clone_github(
            spec["url"], dest, branch=spec.get("branch", "main"), include=include
        )

    walk_root = dest / spec["subdir"] if spec.get("subdir") else dest
    skill_dirs = _discover_skill_dirs(walk_root)
    indexed = index_catalog(skill_dirs, slug, spec["url"], kb) if kb is not None else 0
    if kb is not None and not skill_dirs:
        logger.warning(
            "source %s yielded no SKILL.md skills under %s", spec["url"], walk_root
        )

    return {
        "slug": slug,
        "url": spec["url"],
        "discovered": len(skill_dirs),
        "indexed": indexed,
        "cache_dir": str(dest),
    }


def index_catalog(skill_dirs: list[Path], slug: str, url: str, kb) -> int:
    """Wipe + rebuild source *slug*'s catalog collection from *skill_dirs*."""
    collection = catalog_collection(slug)
    coll_dir = Path(kb.index_dir) / collection
    if coll_dir.exists():
        shutil.rmtree(coll_dir)

    texts: list[str] = []
    metas: list[dict] = []
    for d in skill_dirs:
        skill_md = d / "SKILL.md"
        spec = _parse_frontmatter(skill_md)
        name = spec.get("name") or d.name
        texts.append(skill_md.read_text())
        metas.append(
            {
                "skill_name": name,
                "tags": ",".join(spec.get("tags") or []),
                "source": f"catalog:{slug}",
                "source_url": url,
                "cache_path": str(d),
            }
        )
    if texts:
        kb.add_entries(texts=texts, collection=collection, metadatas=metas)
    return len(texts)


# ---------------------------------------------------------------------------
# Lookup + install
# ---------------------------------------------------------------------------


def find_catalog_skill(name: str, *, cache_dir: Path = SKILL_SOURCES_DIR) -> Path:
    """Locate a cached catalog skill dir by name across all synced sources.

    Matches on frontmatter ``name`` first, then directory name.  Raises on
    no match, or on an ambiguous match spanning more than one source repo.
    """
    matches: list[Path] = []
    if cache_dir.exists():
        for slug_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
            for d in _discover_skill_dirs(slug_dir):
                spec = _parse_frontmatter(d / "SKILL.md")
                if spec.get("name") == name or d.name == name:
                    matches.append(d)
    if not matches:
        raise LookupError(
            f"No catalog skill named '{name}'. Run 'dsagt skills sync' or "
            f"add_skill_source first, then search_skills to find one."
        )
    # Collapse matches that point at the same source repo (slug = first path
    # part under cache_dir); ambiguity only matters across different sources.
    by_source = {p.relative_to(cache_dir).parts[0]: p for p in matches}
    if len(by_source) > 1:
        raise LookupError(
            f"Skill '{name}' exists in multiple sources "
            f"({', '.join(sorted(by_source))}); install by source with "
            f"'dsagt skills add <source>/{name}'."
        )
    return next(iter(by_source.values()))


def install_into_project(
    name: str, project_dir: str | Path, *, cache_dir: Path = SKILL_SOURCES_DIR
) -> dict:
    """Copy a catalog skill into ``<project>/skills/<name>/`` (with scripts/refs).

    The destination directory is named after the skill's frontmatter ``name``
    (falling back to its source dir name) so it matches the invocable name in
    native discovery.  Returns ``{name, source_dir, dest_dir, action}``.
    """
    src = find_catalog_skill(name, cache_dir=cache_dir)
    spec = _parse_frontmatter(src / "SKILL.md")
    skill_name = spec.get("name") or src.name

    dest = Path(project_dir) / "skills" / skill_name
    action = "updated" if dest.exists() else "added"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    return {
        "name": skill_name,
        "source_dir": str(src),
        "dest_dir": str(dest),
        "action": action,
    }
