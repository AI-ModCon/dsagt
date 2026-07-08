"""Unit tests for the keyword scorer and SkillRouter (no network, no embedder).

The router is exercised in its KB-free keyword mode against a real
``SkillRegistry`` (bundled skills suppressed via an empty source dir) plus a
fake catalog cache, and in KB mode against a small fake KnowledgeBase.
"""

import json

from dsagt.registry import SkillRegistry
from dsagt.skills import SkillRouter, rank_skills, score_skill


def _mkskill(d, name, desc):
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\nbody\n"
    )
    return d


def _registry(tmp_path, skills=None):
    """SkillRegistry with bundled skills suppressed and optional project skills."""
    empty_bundled = tmp_path / "no_bundled"
    empty_bundled.mkdir()
    reg = SkillRegistry(
        runtime_dir=tmp_path / "proj",
        source_skills_dir=str(empty_bundled),
        kb=None,
    )
    for name, desc in (skills or {}).items():
        _mkskill(reg.skills_dir / name, name, desc)
    return reg


class FakeKB:
    """Minimal duck-typed KnowledgeBase: collections + search + index_dir."""

    def __init__(self, collections, hits_by_collection=None, index_dir="/tmp/none"):
        self.collections = list(collections)
        self._hits = hits_by_collection or {}
        self.index_dir = index_dir

    def search(self, query, collection=None, collections=None, top_k=5, **kwargs):
        # Mirror KnowledgeBase.search's fan-out surface; a simplified stand-in
        # that merges by raw score (real cross-collection RRF is exercised in
        # test_knowledge_base.TestFederatedSearch).
        targets = collections or ([collection] if collection else [])
        hits: list[dict] = []
        for c in targets:
            hits.extend(self._hits.get(c, []))
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]


def _hit(name, text, source, score, tags=""):
    return {
        "chunk": {
            "metadata": {"skill_name": name, "source": source, "tags": tags},
            "text": text,
        },
        "score": score,
    }


# ---------------------------------------------------------------------------
# keyword scorer
# ---------------------------------------------------------------------------


def test_score_name_token_weighs_more_than_description():
    name_hit = score_skill("slurm", "slurm-submit", "unrelated text")
    desc_hit = score_skill("slurm", "other", "submit a slurm job")
    assert name_hit > desc_hit


def test_score_exact_name_beats_substring():
    exact = score_skill("datacard", "datacard", "x")
    substr = score_skill("datacard", "datacard-generator", "x")
    assert exact > substr > 0


def test_score_substring_description_bonus():
    assert score_skill("batch job", "x", "submit a batch job") > 0


def test_stopwords_do_not_score():
    # only stopwords overlap → no score
    assert score_skill("the and of", "the skill", "and of the") == 0.0


def test_empty_query_scores_zero():
    assert score_skill("", "anything", "anything") == 0.0


def test_substring_bonuses_are_mutually_exclusive():
    # Genesis parity: at most ONE of the +6/+4/+2 substring bonuses fires.
    # name-token 2 + desc-token 1 + exact-name 6 = 9; the desc-substring +2 is
    # NOT also added (a stacking bug would give 11).
    assert score_skill("alpha", "alpha", "alpha tool") == 9.0


def test_single_char_tokens_dropped():
    # "x" is a single char → not a token, so no name-token overlap.
    assert score_skill("x", "x", "y") == 6.0  # only the exact-name bonus


def test_rank_orders_and_breaks_ties_by_name():
    skills = [
        {"name": "zeta", "description": "submit jobs"},
        {"name": "alpha", "description": "submit jobs"},
        {"name": "unrelated", "description": "nothing here"},
    ]
    ranked = rank_skills("submit", skills, top_k=5)
    names = [s["name"] for s, _ in ranked]
    assert names == ["alpha", "zeta"]  # equal score → name asc; unrelated dropped


# ---------------------------------------------------------------------------
# keyword-mode search (kb is None)
# ---------------------------------------------------------------------------


def test_search_keyword_is_catalog_only(tmp_path):
    # Installed skills are NOT search candidates (they're natively discovered);
    # only the cached catalog is keyword-scored.
    reg = _registry(tmp_path, {"slurm-submit": "submit a batch job to slurm"})
    cache = tmp_path / "cache"
    _mkskill(
        cache / "genesis" / "slurm-catalog",
        "slurm-catalog",
        "submit a batch job to slurm",
    )
    r = SkillRouter(skill_registry=reg, cache_dir=cache)
    out = r.search("slurm batch")
    assert "slurm-catalog" in out  # catalog skill found
    assert "slurm-submit" not in out  # installed skill NOT surfaced by search
    assert "[catalog · install_skill to add]" in out


def test_search_keyword_includes_catalog_cache(tmp_path):
    reg = _registry(tmp_path)
    cache = tmp_path / "cache"
    _mkskill(
        cache / "genesis-skills" / "croissant",
        "croissant-validator",
        "validate a croissant metadata file",
    )
    r = SkillRouter(skill_registry=reg, cache_dir=cache)
    out = r.search("croissant")
    assert "croissant-validator" in out
    assert "[catalog · install_skill to add]" in out


def test_search_is_stateless(tmp_path):
    # No recency queue: repeating a query yields the same result, no suppression.
    reg = _registry(tmp_path)
    cache = tmp_path / "cache"
    _mkskill(cache / "src" / "slurm-x", "slurm-x", "submit a batch job to slurm")
    r = SkillRouter(skill_registry=reg, cache_dir=cache)
    first = r.search("slurm")
    second = r.search("slurm")
    assert first == second
    assert "Found 1 skill" in second


def test_search_exact_name_is_kb_free(tmp_path):
    reg = _registry(tmp_path, {"datacard-gen": "make a dataset card"})
    r = SkillRouter(skill_registry=reg)
    out = r.search(skill_name="datacard-gen")
    assert "datacard-gen" in out
    assert r.search(skill_name="nope").startswith("No skill named")


# ---------------------------------------------------------------------------
# KB-mode search
# ---------------------------------------------------------------------------


def test_search_kb_merges_catalog_collections(tmp_path):
    # Only skills_catalog__* collections are searched; the installed 'skills'
    # collection is ignored even if present.
    kb = FakeKB(
        collections=["skills", "skills_catalog__a", "skills_catalog__b"],
        hits_by_collection={
            "skills": [_hit("installed-one", "installed", "registered", 0.99)],
            "skills_catalog__a": [_hit("cat-a", "catalog a", "catalog:a", 0.9)],
            "skills_catalog__b": [_hit("cat-b", "catalog b", "catalog:b", 0.5)],
        },
    )
    r = SkillRouter(skill_registry=_registry(tmp_path), kb=kb)
    out = r.search("anything")
    assert "installed-one" not in out  # installed collection not searched
    assert "cat-a" in out and "cat-b" in out
    assert out.index("cat-a") < out.index("cat-b")  # higher score first


def test_search_kb_tag_filter(tmp_path):
    kb = FakeKB(
        collections=["skills_catalog__x"],
        hits_by_collection={
            "skills_catalog__x": [
                _hit("tagged", "x", "catalog:x", 0.9, tags="hpc,slurm"),
                _hit("untagged", "y", "catalog:x", 0.8, tags=""),
            ]
        },
    )
    r = SkillRouter(skill_registry=_registry(tmp_path), kb=kb)
    out = r.search("x", tag="slurm")
    assert "tagged" in out and "untagged" not in out


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------


def test_list_sources_flags_synced(tmp_path):
    from dsagt.skills import KNOWN_SOURCES, _repo_slug
    from dsagt.registry import catalog_collection

    genesis_coll = catalog_collection(_repo_slug(KNOWN_SOURCES["genesis"]["url"]))
    index_dir = tmp_path / "idx"
    (index_dir / genesis_coll).mkdir(parents=True)
    (index_dir / genesis_coll / "chroma_ids.json").write_text(
        json.dumps(["1", "2", "3"])
    )

    kb = FakeKB(collections=[genesis_coll], index_dir=str(index_dir))
    # list_sources needs only a KB — no skill_registry required.
    r = SkillRouter(kb=kb)
    sources = {s["name"]: s for s in r.list_sources()}
    assert sources["genesis"]["synced"] is True
    assert sources["genesis"]["indexed"] == 3
    assert sources["anthropic"]["synced"] is False
    assert sources["anthropic"]["indexed"] == 0
