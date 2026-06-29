"""Skill discovery — catalog data plane, keyword scorer, and the router facade.

DSAGT fetches external Agent-Skills repos, indexes each per-source into a
``skills_catalog__<slug>`` KB collection, and searches/installs them into a
project.  This is the one job native skill discovery can't do: a *catalog* skill
stays searchable without being copied locally or held in the agent's context
(you can't hold thousands of skill descriptions in context), while an
*installed* skill is copied into ``<project>/skills/<name>/`` and mirrored into
the agent's native skills dir (``agents.base.setup_skills``).  It backs the MCP
``search_skills`` tool and the ``dsagt skills`` CLI through the one
:class:`SkillRouter` facade, so search/install policy can't diverge between them.
Design-wise it stays cheap and degradable: :class:`SkillsCatalog` composes over
the host server's :class:`~dsagt.knowledge.KnowledgeBase` (shared embedder, no
second model load), falls back to a Genesis-derived keyword scorer
(:func:`rank_skills`) when no embedder/KB is configured, and indexes per-source
so re-sync is an idempotent drop-and-rebuild of just that source's collection.

Class map — every edge is ``<branch>─<rel> Class`` (``◇`` holds · ``◆`` owns)::

    SkillRouter                     render/MCP facade: the search_skills string,
    │                               the empty-result message, exact-name lookup
    ├─◇ SkillsCatalog               the catalog data plane (constructed here, or
    │   │                           shared in via catalog=)
    │   └─◇ KnowledgeBase           shared vector store + embedder; None selects
    │                               the keyword fallback over the clone cache
    └─◇ SkillRegistry               installed-skill registry, exact-name lookup only

    free fns:
      keyword scorer  score_skill · rank_skills           (Genesis parity)
      source resolve  resolve_source · _repo_slug · persist_source_to_config
      sync / index    sync_source · _discover_skill_dirs · index_catalog
      install         find_catalog_skill · install_into_project · _capture_attribution
      render          _where_label

Genesis Skills: Apache-2.0, gitlab.osti.gov/genesis/genesis-skills
(``skill_search/catalog.py``).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import yaml

from dsagt.registry import (
    CATALOG_COLLECTION_PREFIX,
    _parse_frontmatter,
    catalog_collection,
)
from dsagt.session import REGISTRY_DIR

logger = logging.getLogger(__name__)


# ===========================================================================
# Keyword scorer — Genesis-derived token-overlap fallback (stdlib only)
# ===========================================================================
#
# A faithful reimplementation (not an import) of the Genesis Skills
# ``skill-search`` engine (``skill_search/catalog.py``: ``_score_skill`` /
# ``rank_skills``).  Used by :class:`SkillsCatalog` when no embedder / KB is
# configured: keyword overlap only, deterministic.
#
# Scoring (per skill, against a query) — matching Genesis exactly:
#
# * +2 for each query token that also appears in the skill **name**
# * +1 for each query token that also appears in the **description**
# * then **at most one** substring bonus (mutually exclusive, in priority
#   order): +6 if the query equals the name, else +4 if it is a substring of
#   the name, else +2 if it is a substring of the description
#
# Tokens are casefolded ``\w+`` runs with hyphens split, single-character
# tokens and stopwords dropped.  Ties break by name (ascending); below
# ``min_score`` are dropped.

#: Stopword set — kept identical to Genesis so ranking parity holds.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "please",
        "the",
        "this",
        "to",
        "use",
        "using",
        "with",
    }
)

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Casefolded word tokens (hyphens split), single-char + stopwords removed."""
    normalized = (text or "").casefold().replace("-", " ")
    return {
        t for t in _TOKEN_RE.findall(normalized) if len(t) > 1 and t not in _STOPWORDS
    }


def score_skill(query: str, name: str, description: str) -> float:
    """Token-overlap score of one skill against *query* (0.0 = no match)."""
    qtokens = _tokens(query)
    normalized_query = (query or "").casefold().strip()
    score = 2 * len(qtokens & _tokens(name)) + len(qtokens & _tokens(description))

    if normalized_query:
        name_l = (name or "").casefold()
        if normalized_query == name_l:
            score += 6
        elif normalized_query in name_l:
            score += 4
        elif normalized_query in (description or "").casefold():
            score += 2
    return float(score)


def rank_skills(
    query: str, skills, top_k: int | None = 8, min_score: int = 1
) -> list[tuple[dict, float]]:
    """Rank *skills* (dicts with ``name`` + ``description``) against *query*.

    Returns ``[(skill, score), ...]`` for skills scoring at least *min_score*,
    sorted by score descending then name ascending, truncated to *top_k* (all
    when *top_k* is ``None``).
    """
    scored: list[tuple[dict, float]] = []
    for s in skills:
        sc = score_skill(query, s.get("name", ""), s.get("description", ""))
        if sc >= min_score:
            scored.append((s, sc))
    scored.sort(key=lambda kv: (-kv[1], (kv[0].get("name") or "")))
    return scored[:top_k] if top_k is not None else scored


# ===========================================================================
# Catalog data plane — sources, sync/index, lookup/install, SkillsCatalog
# ===========================================================================

#: Default source enabled out of the box (matches dsagt_config.yaml default).
DEFAULT_SOURCE = "k-dense-ai"

#: Curated, named skill sources.  ``subdir`` scopes the recursive SKILL.md
#: walk when set (cheaper clone); when omitted the whole repo is cloned and
#: walked, which is robust to category-nested layouts.
KNOWN_SOURCES: dict[str, dict] = {
    "k-dense-ai": {
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
    "genesis": {
        "url": "https://gitlab.osti.gov/genesis/genesis-skills",
        "branch": "main",
        "subdir": "skills",
        "description": "GENESIS skills (OSTI GitLab) — aggregated agent-skill "
        "catalog: HPC (Slurm/PBS, Perlmutter/Aurora/Frontier), HuggingFace, "
        "LangChain, OpenAI, Anthropic, plasma-sim, ModCon, and more (70+).",
    },
}

#: Shared, machine-global cache of cloned source repos (sibling of kb_index/).
SKILL_SOURCES_DIR = REGISTRY_DIR / ".skill_sources"


# ---------------------------------------------------------------------------
# Source resolution + slugging
# ---------------------------------------------------------------------------


def resolve_source(source: str | dict) -> dict:
    """Resolve a known-source name, a git URL (any host), or a full spec dict.

    A full ``http(s)://`` / ``git@`` URL works for any host (GitHub, GitLab,
    …); the bare ``owner/repo`` shorthand assumes GitHub.  Returns a dict with
    at least ``url``; optional ``branch`` / ``subdir``.
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
        f"({', '.join(sorted(KNOWN_SOURCES))}), a git URL (any host), "
        f"or owner/repo (GitHub)."
    )


def persist_source_to_config(project_dir: str | Path, spec: dict) -> bool:
    """Append a resolved source to ``skills.sources`` in the project config.

    Dedupes by URL.  Returns True if the config was updated.  No-op (returns
    False) if the config file is missing — the catalog is still indexed
    either way.  Used by both the ``add_skill_source`` MCP tool and the
    ``dsagt skills add`` CLI so a CLI-added source is re-synced by a later
    config-driven ``dsagt skills sync``.
    """
    cfg_path = Path(project_dir) / ".dsagt" / "config.yaml"
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
    """Stable, collection-name-safe slug from a repo URL (``owner-repo``).

    Host-agnostic: the scheme and host are stripped so github.com, gitlab.*,
    etc. all reduce to the ``owner/repo`` path.  GitHub URLs keep the slug
    they had before this generalization, so existing catalog collections do
    not need rebuilding.
    """
    s = url.rstrip("/")
    s = re.sub(r"^https?://", "", s)  # drop scheme
    s = re.sub(r"^git@", "", s)  # ssh form: git@host:owner/repo
    s = re.sub(r"\.git$", "", s).lower()
    s = re.sub(r"^[^/:]+[/:]", "", s)  # drop the host segment
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


def _catalog_embed_text(spec: dict, fallback_name: str) -> str:
    """Text embedded for catalog search: the frontmatter ``name`` + ``description``
    (+ ``tags``) only — *not* the SKILL.md body.

    Discovery is progressive-disclosure level 1: the description is authored to
    say *what the skill does and when to use it*, which is exactly the match
    target.  Embedding the body would dilute that signal, and the embedder
    truncates long input anyway (so a full SKILL.md is both incomplete and
    misallocated).  This also keeps the semantic backend ranking on the same
    fields as the keyword fallback.
    """
    name = spec.get("name") or fallback_name
    desc = spec.get("description") or ""
    tags = " ".join(spec.get("tags") or [])
    return f"{name}: {desc} {tags}".strip()


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
        texts.append(_catalog_embed_text(spec, d.name))
        metas.append(
            {
                "skill_name": name,
                "description": (spec.get("description") or "")[:300],
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

    Matches on frontmatter ``name`` first, then directory name.  A bare name
    must be unique across the machine-global clone cache; when the same name
    exists in more than one synced source, pass a **source-qualified**
    ``<slug>/<name>`` (the slug is the per-source cache dir / catalog-collection
    suffix, as shown by ``list_skill_sources`` / ``dsagt skills list
    --catalog``) to pick one.  Raises on no match or on a still-ambiguous bare
    name.
    """
    source_filter: str | None = None
    skill = name
    if "/" in name:
        # Skill names never contain '/', so a slash means "<source>/<skill>".
        source_filter, skill = name.split("/", 1)

    matches: list[Path] = []
    if cache_dir.exists():
        for slug_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
            if source_filter is not None and slug_dir.name != source_filter:
                continue
            for d in _discover_skill_dirs(slug_dir):
                spec = _parse_frontmatter(d / "SKILL.md")
                if spec.get("name") == skill or d.name == skill:
                    matches.append(d)
    if not matches:
        where = f" in source '{source_filter}'" if source_filter else ""
        raise LookupError(
            f"No catalog skill named '{skill}'{where}. Run 'dsagt skills sync' "
            f"or add_skill_source first, then search_skills to find one."
        )
    # Collapse matches that point at the same source repo (slug = first path
    # part under cache_dir); ambiguity only matters across different sources.
    by_source = {p.relative_to(cache_dir).parts[0]: p for p in matches}
    if len(by_source) > 1:
        sources = sorted(by_source)
        raise LookupError(
            f"Skill '{skill}' exists in multiple sources ({', '.join(sources)}); "
            f"reinstall with a source-qualified name, e.g. '{sources[0]}/{skill}'."
        )
    return next(iter(by_source.values()))


#: License / attribution files to preserve when installing a catalog skill.
_ATTRIBUTION_GLOBS = (
    "LICENSE*",
    "NOTICE*",
    "COPYING*",
    "COPYRIGHT*",
    "ATTRIBUTION*",
)


def _capture_attribution(src: Path, dest: Path, cache_dir: Path) -> list[str]:
    """Preserve license/attribution when installing a (third-party) catalog skill.

    ``copytree`` already carries files *inside* the skill dir.  A skill is often
    governed by a per-subtree or repo-root ``LICENSE`` / ``NOTICE`` /
    ``ATTRIBUTION`` that lives *outside* its own folder, so this also pulls those
    from ancestor dirs up to the source repo root (which ``clone_github`` mirrors
    into the cache root even for sparse ``subdir`` clones).  Nearest ancestor
    wins a filename collision; skill-local files (already in ``dest``) are never
    overwritten.  Always stamps a ``PROVENANCE.txt`` recording the source.
    Returns the names of files captured from ancestors.
    """
    src, dest, cache_dir = Path(src), Path(dest), Path(cache_dir)
    try:
        slug = src.relative_to(cache_dir).parts[0]
        repo_root = cache_dir / slug
        rel = src.relative_to(repo_root)
    except ValueError:  # src outside the cache (shouldn't happen) — degrade.
        slug, repo_root, rel = src.parent.name, src.parent, Path(src.name)

    captured: list[str] = []
    node = src.parent
    while True:
        for pat in _ATTRIBUTION_GLOBS:
            for f in sorted(node.glob(pat)):
                if f.is_file() and not (dest / f.name).exists():
                    shutil.copy2(f, dest / f.name)
                    captured.append(f.name)
        if node == repo_root or node == node.parent:
            break
        node = node.parent

    (dest / "PROVENANCE.txt").write_text(
        f"Installed by dsagt from catalog source: {slug}\n"
        f"Source path in repo: {rel}\n"
    )
    return captured


def install_into_project(
    name: str, project_dir: str | Path, *, cache_dir: Path = SKILL_SOURCES_DIR
) -> dict:
    """Copy a catalog skill into ``<project>/skills/<name>/`` (with scripts/refs).

    The destination directory is named after the skill's frontmatter ``name``
    (falling back to its source dir name) so it matches the invocable name in
    native discovery.  Preserves upstream license/attribution (see
    :func:`_capture_attribution`).  Returns
    ``{name, source_dir, dest_dir, action, attribution}``.
    """
    src = find_catalog_skill(name, cache_dir=cache_dir)
    spec = _parse_frontmatter(src / "SKILL.md")
    skill_name = spec.get("name") or src.name

    dest = Path(project_dir) / "skills" / skill_name
    action = "updated" if dest.exists() else "added"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    attribution = _capture_attribution(src, dest, cache_dir)

    return {
        "name": skill_name,
        "source_dir": str(src),
        "dest_dir": str(dest),
        "action": action,
        "attribution": attribution,
    }


# ---------------------------------------------------------------------------
# SkillsCatalog — the catalog data plane (composition over KnowledgeBase)
# ---------------------------------------------------------------------------


class SkillsCatalog:
    """The external-skill *catalog* behind one object.

    Composition over :class:`~dsagt.knowledge.KnowledgeBase`: it holds a KB
    handle (the host server's existing instance → shared embedder, no second
    model load) plus the clone-cache dir, and exposes the skill-domain ops —
    ``sync`` / ``search`` / ``install`` / ``list_sources``.  The skill-specific
    behavior (frontmatter-indexed catalog collections, the no-embedder keyword
    fallback over the clone cache) lives here; the vector store + embedder are
    the shared KB.  :class:`SkillRouter` is a thin render/MCP facade over this.

    Catalog tier only: installed/created skills are natively auto-discovered by
    every supported agent, so they are never search candidates.  ``search``
    covers the not-yet-installed catalog (which native discovery can't see)
    plus the Cline-skills-disabled / no-embedder case via the keyword scorer.
    """

    def __init__(self, *, kb=None, cache_dir: Path | None = None):
        """Compose a catalog over an existing KB + a clone-cache directory.

        ``kb`` is the host server's :class:`~dsagt.knowledge.KnowledgeBase` (so
        the embedder/Chroma are shared, never a second model load); pass ``None``
        for the no-embedder keyword path.  ``cache_dir`` overrides the default
        machine-global clone cache (:data:`SKILL_SOURCES_DIR`) — handy for tests.
        """
        self._kb = kb
        self._cache_dir = cache_dir  # default resolved lazily

    @property
    def has_kb(self) -> bool:
        """True when an embedder-backed KB is available (vs the keyword path)."""
        return self._kb is not None

    def _resolved_cache_dir(self) -> Path:
        """The clone-cache dir — the ``cache_dir`` override or the global default."""
        return Path(self._cache_dir or SKILL_SOURCES_DIR)

    # -- write ops (delegate to the module functions) ------------------------

    def sync(self, source, *, force: bool = False) -> dict:
        """Clone + frontmatter-index a source into its catalog collection."""
        return sync_source(
            source, kb=self._kb, cache_dir=self._resolved_cache_dir(), force=force
        )

    def install(self, name: str, project_dir) -> dict:
        """Copy a catalog skill into ``<project>/skills/<name>/`` (+ attribution)."""
        return install_into_project(
            name, project_dir, cache_dir=self._resolved_cache_dir()
        )

    # -- backend selection ---------------------------------------------------

    def synced_collections(self) -> list[str]:
        """The ``skills_catalog__*`` collections currently indexed in the KB."""
        if self._kb is None:
            return []
        return [
            c for c in self._kb.collections if c.startswith(CATALOG_COLLECTION_PREFIX)
        ]

    def search(self, query=None, *, top_k: int = 8, tag=None) -> list[dict]:
        """Rank catalog candidates → normalized hit dicts (data, not rendered).

        ChromaDB semantic search when a KB exists, else the Genesis-derived
        keyword scorer over the clone cache.
        """
        if self._kb is not None:
            return self._select_kb(query, top_k, tag)
        return self._select_keyword(query, top_k, tag)

    def _select_kb(self, query, top_k: int, tag) -> list[dict]:
        """Semantic backend: rank-fused search across every synced catalog
        collection, normalized into hit dicts.

        Fan-out + RRF live in ``KnowledgeBase.search`` (the shared substrate);
        catalog collections are homogeneous (one embedder) so the fusion is a
        clean rank merge.  When a ``tag`` filter is set we over-fetch
        (``top_k * 3``) then post-filter so the tag doesn't starve the results.
        """
        collections = self.synced_collections()
        if not collections:
            return []
        fetch_k = top_k * 3 if tag else top_k
        # collections all come from synced_collections() (they exist), and
        # KnowledgeBase.search already skips a missing collection with a warning
        # and raises only when every target fails — so a raise here is a real
        # failure worth surfacing, not a can't-happen state to swallow.
        hits = self._kb.search(
            query=query or "skill", collections=collections, top_k=fetch_k
        )
        out = []
        for r in hits:
            chunk = r.get("chunk", {})
            meta = chunk.get("metadata", {})
            out.append(
                {
                    "name": meta.get("skill_name", "unknown"),
                    "summary": (meta.get("description") or chunk.get("text", "") or "")[
                        :200
                    ],
                    "source": meta.get("source", ""),
                    "tags": meta.get("tags", ""),
                    "score": r.get("score", 0.0),
                }
            )
        if tag:
            out = [h for h in out if tag in (h["tags"] or "")]
        out.sort(key=lambda h: h["score"], reverse=True)
        return out[:top_k]

    def _candidate_skills(self) -> list[dict]:
        """Cached-catalog skills as scorer-ready dicts (no KB needed)."""
        cands: list[dict] = []
        cache = self._resolved_cache_dir()
        if cache.exists():
            for slug_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
                for d in _discover_skill_dirs(slug_dir):
                    spec = _parse_frontmatter(d / "SKILL.md")
                    if spec.get("name"):
                        cands.append(
                            {
                                "name": spec["name"],
                                "description": spec.get("description", ""),
                                "tags": ",".join(spec.get("tags", []) or []),
                                "source": f"catalog:{slug_dir.name}",
                            }
                        )
        return cands

    def _select_keyword(self, query, top_k: int, tag) -> list[dict]:
        """No-embedder backend: rank cached-catalog skills with the Genesis
        token-overlap scorer (:func:`rank_skills`) into normalized hit dicts.

        With no ``query`` there's nothing to score, so it returns the first
        ``top_k`` candidates (tag-filtered) at score 0.0 — a browse mode rather
        than a search.
        """
        cands = self._candidate_skills()
        if tag:
            cands = [c for c in cands if tag in (c["tags"] or "")]
        if not query:
            picks = cands[:top_k]
            return [
                {**c, "summary": (c["description"] or "")[:200], "score": 0.0}
                for c in picks
            ]
        ranked = rank_skills(query, cands, top_k=top_k)
        return [
            {**c, "summary": (c["description"] or "")[:200], "score": sc}
            for c, sc in ranked
        ]

    # -- source view ---------------------------------------------------------

    def list_sources(self) -> list[dict]:
        """Known sources + synced flag + indexed count (one source of truth)."""
        synced = set(self.synced_collections())
        out = []
        for name, spec in KNOWN_SOURCES.items():
            coll = catalog_collection(_repo_slug(spec["url"]))
            is_synced = coll in synced
            out.append(
                {
                    "name": name,
                    "url": spec["url"],
                    "description": spec.get("description", ""),
                    "synced": is_synced,
                    "indexed": self._indexed_count(coll) if is_synced else 0,
                }
            )
        return out

    def _indexed_count(self, collection: str) -> int:
        """Number of skills indexed in a catalog *collection* (0 if absent/no KB).

        Reads the collection's persisted ``chroma_ids.json`` directly rather than
        querying the store — cheap, and works without loading the embedder.
        """
        if self._kb is None:
            return 0
        ids = Path(self._kb.index_dir) / collection / "chroma_ids.json"
        try:
            return len(json.loads(ids.read_text()))
        except (FileNotFoundError, ValueError, OSError):
            return 0


# ===========================================================================
# SkillRouter — the thin render/MCP facade over the catalog
# ===========================================================================
#
# ``SkillRouter`` adds only the presentation concerns that the MCP handlers and
# the CLI share: rendering a ranked hit list into the ``search_skills`` string,
# the empty-result message, and the exact-``skill_name`` lookup (which needs the
# installed-skill registry, not the catalog).
#
# Construct it from the same inputs at every call site (MCP ``search_skills``,
# CLI ``skills search/list``) so policy can't diverge between them — or hand it
# a prebuilt :class:`SkillsCatalog` via ``catalog=`` so a server that already
# owns a shared KB reuses one catalog instance.
#
# Skill *materialization* (mirroring installed skills into each agent's native
# skills directory) lives in the agent layer (``AgentSetup.setup_skills``), not
# here: every supported agent (claude/codex/goose/cline) natively
# auto-discovers ``SKILL.md`` folders, so there is no agent-facing disclosure
# tier for the router to own.  ``search_skills`` exists for the *catalog* tier
# (skills not yet installed, which native discovery can't see) plus the
# no-embedder keyword fallback.


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
        """Wire the router to a catalog + (optional) installed-skill registry.

        Pass a prebuilt ``catalog`` (a :class:`SkillsCatalog`) to share one
        instance with the server; otherwise one is constructed from ``kb`` /
        ``cache_dir``.  ``skill_registry`` is only consulted for the exact
        ``skill_name`` lookup in :meth:`search` (installed skills live there,
        not in the catalog), so it may be ``None`` for catalog-only callers.
        """
        self._catalog = (
            catalog
            if catalog is not None
            else SkillsCatalog(kb=kb, cache_dir=cache_dir)
        )
        self._reg = skill_registry

    # -- rendering -----------------------------------------------------------

    def _render_search(self, hits: list[dict]) -> str:
        """Format ranked catalog hits into the ``search_skills`` markdown list.

        Each line carries the skill name, an origin tag (:func:`_where_label`),
        the score, and the summary — the human-facing string the MCP tool and
        CLI both return.
        """
        lines = []
        for h in hits:
            lines.append(
                f"- **{h['name']}**{_where_label(h['source'])} "
                f"(score: {h['score']:.2f})\n  {h['summary']}"
            )
        return f"Found {len(hits)} skill(s):\n\n" + "\n\n".join(lines)

    def _empty_message(self) -> str:
        """The no-results string, tailored to *why* nothing matched.

        When a KB exists but no catalog source is synced yet, point the agent
        at ``list_skill_sources`` / ``add_skill_source`` (the likely cause);
        otherwise it's a genuine no-match for the query.
        """
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
