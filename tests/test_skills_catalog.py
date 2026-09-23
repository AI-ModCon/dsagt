"""Unit tests for the external skill catalog (fetch / index / install) and
the native-skill mirror.  No network: ``clone_github`` is monkeypatched and
the KB is a lightweight fake that records ``add_entries`` calls."""

import json
import os

import pytest

from dsagt.agents.base import (
    _NATIVE_DESCRIPTION_CAP,
    _SKILL_MANIFEST,
    _mirror_skills_to,
)
from dsagt import skills as sc
from dsagt.registry import CATALOG_COLLECTION_PREFIX, catalog_collection


def _mkskill(d, name, desc="a short description"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\nbody\n"
    )
    return d


# ---------------------------------------------------------------------------
# slug + source resolution
# ---------------------------------------------------------------------------


def test_has_tag_matches_tokens_not_substrings():
    """Regression: tag filtering must compare comma-separated tokens, not do
    substring matching (``ml`` in ``html``)."""
    assert sc._has_tag("html,xml", "html")
    assert not sc._has_tag("html,xml", "ml")
    assert not sc._has_tag("microbiome", "bio")
    assert sc._has_tag("a, b , c", "b")  # whitespace-trimmed tokens
    assert not sc._has_tag(None, "x")
    assert not sc._has_tag("", "x")


def test_repo_slug_is_collection_safe():
    slug = sc._repo_slug("https://github.com/K-Dense-AI/scientific-agent-skills")
    assert slug == "k-dense-ai-scientific-agent-skills"
    assert sc._repo_slug("git@github.com:Foo/Bar.git") == "foo-bar"


def test_repo_slug_is_host_agnostic():
    # Non-GitHub hosts (GitLab, etc.) reduce to owner-repo, scheme/host dropped.
    assert sc._repo_slug("https://gitlab.osti.gov/genesis/genesis-skills") == (
        "genesis-genesis-skills"
    )
    assert sc._repo_slug("git@github.com:AI-ModCon/genesis-skills.git") == (
        "ai-modcon-genesis-skills"
    )


def test_known_source_genesis_covers_whole_skills_tree():
    spec = sc.resolve_source("genesis")
    assert spec["url"] == "https://github.com/AI-ModCon/genesis-skills"
    # subdir scopes the recursive SKILL.md walk to the whole skills/ tree so
    # every category (hpc, huggingface, langchain, …) is discoverable.
    assert spec["subdir"] == "skills"
    assert spec["branch"] == "main"


def test_base_skills_name_their_upstream_sources():
    # ``skill-creator`` and ``datacard-generator`` are maintained in the genesis
    # catalog; ``aidrin`` is the AIDRIN repo's own skill under .claude/skills
    # at the release tag of the installed package (a bare URL would clone the
    # whole repo including examples/sample_data).
    from importlib.metadata import version

    from dsagt.readiness import aidrin_release_tag

    by_name = {b["name"]: sc.resolve_source(b["source"]) for b in sc.base_skills()}
    assert by_name["skill-creator"]["url"] == sc.KNOWN_SOURCES["genesis"]["url"]
    assert by_name["datacard-generator"]["url"] == sc.KNOWN_SOURCES["genesis"]["url"]
    assert by_name["aidrin"]["url"] == "https://github.com/idtlab/AIDRIN"
    assert by_name["aidrin"]["subdir"] == ".claude/skills"
    assert by_name["aidrin"]["branch"] == aidrin_release_tag(version("aidrin"))
    assert "aidrin" not in sc.KNOWN_SOURCES


def test_install_base_skills_reuses_cache_and_installs(tmp_path, monkeypatch):
    """Each base skill absent from the project is installed by a
    source-qualified name from the shared source cache without a forced
    re-clone; a skill already in the project is kept as it is, and its
    codes are still registered."""
    cache = tmp_path / "cache"
    synced = []

    def fake_sync(source, *, kb=None, cache_dir, force=False):
        # Materialize the skill where a real clone would put it.
        slug = sc._repo_slug(source["url"])
        synced.append((slug, force))
        (cache_dir / slug).mkdir(parents=True, exist_ok=True)
        (cache_dir / slug / "SOURCE_COMMIT").write_text(f"{slug}-commit\n")
        subdir = source.get("subdir") or ""
        for b in sc.base_skills():
            if sc.resolve_source(b["source"])["url"] == source["url"]:
                d = _mkskill(cache_dir / slug / subdir / "x" / b["name"], b["name"])
                for code in b.get("codes", ()):
                    if "script" in code:
                        (d / code["script"]).parent.mkdir(parents=True, exist_ok=True)
                        (d / code["script"]).write_text("print('ok')\n")
        return {"slug": slug}

    monkeypatch.setattr(sc, "sync_source", fake_sync)
    proj = tmp_path / "proj"
    edited = _mkskill(proj / "skills" / "aidrin", "aidrin", desc="edited by the user")

    results = sc.install_base_skills(proj, cache_dir=cache)
    assert [(r["name"], r["action"]) for r in results] == [
        ("skill-creator", "added"),
        ("datacard-generator", "added"),
        ("aidrin", "kept"),
    ]
    # No forced re-clone: a cached source is reused as is; a skill already
    # in the project is not fetched at all.
    assert synced == [
        ("ai-modcon-genesis-skills", False),
        ("ai-modcon-genesis-skills", False),
    ]
    assert (proj / "skills" / "skill-creator" / "SKILL.md").exists()
    assert (proj / "skills" / "datacard-generator" / "SKILL.md").exists()
    assert "edited by the user" in (edited / "SKILL.md").read_text()
    provenance = (proj / "skills" / "datacard-generator" / "PROVENANCE.txt").read_text()
    assert "Commit: ai-modcon-genesis-skills-commit" in provenance
    # The scripts the datacard workflow runs, and the aidrin CLI, are codes.
    assert (proj / "skills" / "datacard-introspect" / "SKILL.md").exists()
    assert (proj / "skills" / "datacard-validate" / "SKILL.md").exists()
    assert (proj / "skills" / "aidrin" / "SKILL.md").exists()


def _base_skill_dirs(proj):
    """Each base skill's directory with a SKILL.md and the scripts its codes name."""
    for entry in sc.base_skills():
        skill_dir = proj / "skills" / entry["name"]
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {entry['name']}\ndescription: d\n---\n"
        )
        for code in entry.get("codes", ()):
            if "script" not in code:
                continue
            script = skill_dir / code["script"]
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("print('ok')\n")


def _register_base(proj, kb=None):
    stored = []
    for entry in sc.base_skills():
        stored += sc.register_skill_scripts(
            proj,
            entry["name"],
            kb=kb,
            overrides=sc._script_overrides(entry),
            cli_codes=[c for c in entry.get("codes", ()) if "executable" in c],
        )
    return stored


def test_base_skill_scripts_are_codes_through_the_shared_registration(tmp_path):
    """Each base-skill code runs the installed skill's script through
    ``dsagt-run``, with ``uv run --with`` for the entries that declare
    dependencies, and a re-run updates instead of duplicating."""
    from dsagt.registry import CodeRegistry

    proj = tmp_path / "proj"
    _base_skill_dirs(proj)
    stored = _register_base(proj)
    registry = CodeRegistry(runtime_dir=proj)
    introspect = registry.get_code("datacard-introspect")
    assert introspect["executable"] == (
        "dsagt-run --code datacard-introspect -- "
        "python skills/datacard-generator/scripts/introspect.py"
    )
    assert introspect["parameters"]["dataset_dir"]["cli"] == "positional"
    assert introspect["parameters"]["dataset_dir"]["role"] == "input"
    validate = registry.get_code("datacard-validate")
    assert validate["executable"] == (
        "dsagt-run --code datacard-validate -- uv run --with pyyaml,pydantic -- "
        "python skills/datacard-generator/scripts/validate_datacard.py"
    )
    assert validate["tags"] == ["datacard-generator"]
    # The aidrin CLI is a code whose executable is the command on the path,
    # so every call the agent makes through it is an execution record.
    aidrin = registry.get_code("aidrin")
    assert aidrin["executable"] == "dsagt-run --code aidrin -- aidrin"
    assert aidrin["parameters"]["args"]["cli"] == "positional"
    assert set(stored) == {
        introspect["executable"],
        validate["executable"],
        registry.get_code("datacard-convert-v1")["executable"],
        aidrin["executable"],
    }
    # Idempotent: the same specs, no duplicates.
    _register_base(proj)
    assert sorted(c["name"] for c in registry.list_codes_raw()) == [
        "aidrin",
        "datacard-convert-v1",
        "datacard-introspect",
        "datacard-validate",
    ]


def test_registration_requires_the_script_an_override_names(tmp_path):
    proj = tmp_path / "proj"
    (proj / "skills" / "datacard-generator").mkdir(parents=True)
    (proj / "skills" / "datacard-generator" / "SKILL.md").write_text(
        "---\nname: x\n---\n"
    )
    entry = next(e for e in sc.base_skills() if e["name"] == "datacard-generator")
    with pytest.raises(FileNotFoundError, match="datacard-generator"):
        sc.register_skill_scripts(
            proj, "datacard-generator", overrides=sc._script_overrides(entry)
        )


def test_persist_source_to_config_appends_and_dedupes(tmp_path):
    import yaml

    (tmp_path / ".dsagt").mkdir()
    cfg = tmp_path / ".dsagt" / "config.yaml"
    cfg.write_text(yaml.dump({"project": "p", "skills": {"sources": []}}))
    spec = {
        "name": "anthropic",
        "url": "https://github.com/anthropics/skills",
        "branch": "main",
    }
    assert sc.persist_source_to_config(tmp_path, spec) is True
    sources = yaml.safe_load(cfg.read_text())["skills"]["sources"]
    assert sources[-1]["name"] == "anthropic"
    # Idempotent: same URL is not appended twice.
    assert sc.persist_source_to_config(tmp_path, spec) is False
    assert len(yaml.safe_load(cfg.read_text())["skills"]["sources"]) == 1
    # No config file → no-op, no crash.
    assert sc.persist_source_to_config(tmp_path / "nope", spec) is False


def test_resolve_source_known_url_and_shorthand():
    assert (
        sc.resolve_source("k-dense-ai")["url"] == sc.KNOWN_SOURCES["k-dense-ai"]["url"]
    )
    assert (
        sc.resolve_source("https://github.com/a/b")["url"] == "https://github.com/a/b"
    )
    assert sc.resolve_source("a/b")["url"] == "https://github.com/a/b"
    with pytest.raises(ValueError):
        sc.resolve_source("not-a-known-name")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_skill_dirs_flat_and_nested(tmp_path):
    root = tmp_path / "skills"
    _mkskill(root / "flat", "flat")
    _mkskill(root / "domain" / "nested", "nested")
    # A dir whose SKILL.md has no name is ignored.
    bad = root / "noname"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("---\ndescription: x\n---\nbody")
    names = sorted(p.name for p in sc._discover_skill_dirs(tmp_path))
    assert names == ["flat", "nested"]


# ---------------------------------------------------------------------------
# find + install
# ---------------------------------------------------------------------------


def test_find_catalog_skill_and_ambiguity(tmp_path):
    cache = tmp_path / "cache"
    _mkskill(cache / "srcA" / "skills" / "alpha", "alpha")
    found = sc.find_catalog_skill("alpha", cache_dir=cache)
    assert found.name == "alpha"
    with pytest.raises(LookupError):
        sc.find_catalog_skill("missing", cache_dir=cache)
    # Same skill name in a second source → ambiguous.
    _mkskill(cache / "srcB" / "skills" / "alpha", "alpha")
    with pytest.raises(LookupError, match="multiple sources"):
        sc.find_catalog_skill("alpha", cache_dir=cache)


def test_find_catalog_skill_source_qualified(tmp_path):
    cache = tmp_path / "cache"
    _mkskill(cache / "srcA" / "skills" / "alpha", "alpha")
    _mkskill(cache / "srcB" / "skills" / "alpha", "alpha")

    # A "<slug>/<name>" qualifier disambiguates which source to install from.
    a = sc.find_catalog_skill("srcA/alpha", cache_dir=cache)
    b = sc.find_catalog_skill("srcB/alpha", cache_dir=cache)
    assert a.relative_to(cache).parts[0] == "srcA"
    assert b.relative_to(cache).parts[0] == "srcB"
    assert a.name == b.name == "alpha"

    # Qualifying with a source that lacks the skill is a clear, source-scoped miss.
    with pytest.raises(LookupError, match="in source 'srcA'"):
        sc.find_catalog_skill("srcA/missing", cache_dir=cache)


def test_install_into_project_source_qualified(tmp_path):
    cache = tmp_path / "cache"
    _mkskill(cache / "srcA" / "skills" / "dup", "dup", desc="from A")
    _mkskill(cache / "srcB" / "skills" / "dup", "dup", desc="from B")
    proj = tmp_path / "proj"
    proj.mkdir()

    # Bare ambiguous name refuses; the source-qualified form installs srcB's copy.
    with pytest.raises(LookupError, match="multiple sources"):
        sc.install_into_project("dup", proj, cache_dir=cache)
    info = sc.install_into_project("srcB/dup", proj, cache_dir=cache)
    assert info["name"] == "dup"
    assert (proj / "skills" / "dup" / "SKILL.md").read_text().count("from B") == 1


def test_install_rejects_path_traversal_name(tmp_path):
    """A hostile catalog SKILL.md whose frontmatter ``name`` escapes the skills
    dir (here ``..``, which would make dest the project root and rmtree it) is
    rejected before any filesystem mutation."""
    cache = tmp_path / "cache"
    # dir name is benign so find_catalog_skill locates it; the danger is the
    # frontmatter name, which is what install uses to build the dest path.
    _mkskill(cache / "src" / "skills" / "evil", "..")
    proj = tmp_path / "proj"
    (proj / "skills").mkdir(parents=True)

    with pytest.raises(ValueError, match="unsafe skill name"):
        sc.install_into_project("..", proj, cache_dir=cache)
    # The project skills dir was not deleted/overwritten.
    assert (proj / "skills").is_dir()


def test_install_into_project_copies_subdirs(tmp_path):
    cache = tmp_path / "cache"
    skill = _mkskill(cache / "src" / "vasp-to-isaac", "vasp-to-isaac")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.py").write_text("print(1)")
    (skill / "references").mkdir()
    (skill / "references" / "spec.md").write_text("# spec")

    proj = tmp_path / "proj"
    proj.mkdir()
    info = sc.install_into_project("vasp-to-isaac", proj, cache_dir=cache)
    dest = proj / "skills" / "vasp-to-isaac"
    assert info["action"] == "added"
    assert (dest / "SKILL.md").exists()
    assert (dest / "scripts" / "run.py").exists()
    assert (dest / "references" / "spec.md").exists()
    # Re-install reports "updated".
    assert (
        sc.install_into_project("vasp-to-isaac", proj, cache_dir=cache)["action"]
        == "updated"
    )


# ---------------------------------------------------------------------------
# sync_source (mocked clone + fake KB)
# ---------------------------------------------------------------------------


class _FakeKB:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.collections = []
        self.adds = []  # (collection, metadatas)

    def add_entries(self, texts, collection, metadatas=None):
        self.adds.append((collection, metadatas))
        if collection not in self.collections:
            self.collections.append(collection)
        return {"collection": collection, "entries_added": len(texts)}


def test_sync_source_reuses_cached_clone_without_force(tmp_path, monkeypatch):
    """A cached source at the requested ref is reused as is: a second sync
    clones nothing.  ``force``, another ref, or a cache from before the
    ``SOURCE_REF`` stamp re-clones."""
    clones = []

    def fake_clone(url, dest, branch="main", include=None):
        clones.append(branch)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SOURCE_REF").write_text(branch + "\n")
        _mkskill(dest / "skills" / "s1", "s1")

    monkeypatch.setattr("dsagt.commands.setup_core_kb.clone_github", fake_clone)
    spec = {"url": "https://github.com/x/y", "branch": "main", "subdir": "skills"}
    cache = tmp_path / "cache"

    sc.sync_source(spec, cache_dir=cache)
    sc.sync_source(spec, cache_dir=cache)
    assert clones == ["main"]

    sc.sync_source(spec, cache_dir=cache, force=True)
    assert clones == ["main", "main"]

    sc.sync_source({**spec, "branch": "v1.0"}, cache_dir=cache)
    assert clones == ["main", "main", "v1.0"]

    (cache / "x-y" / "SOURCE_REF").unlink()
    sc.sync_source({**spec, "branch": "v1.0"}, cache_dir=cache)
    assert clones == ["main", "main", "v1.0", "v1.0"]


def test_sync_source_keeps_the_cache_when_a_reclone_fails(tmp_path, monkeypatch):
    """A re-clone that fails (offline, private repository) leaves the previous
    clone in place and raises; nothing is lost and a later sync retries."""

    def failing_clone(url, dest, branch="main", include=None):
        raise RuntimeError("Git clone failed: could not read Username")

    cache = tmp_path / "cache"
    old = _mkskill(cache / "x-y" / "skills" / "s1", "s1")
    (cache / "x-y" / "SOURCE_REF").write_text("main\n")
    monkeypatch.setattr("dsagt.commands.setup_core_kb.clone_github", failing_clone)
    spec = {"url": "https://github.com/x/y", "branch": "v2.0", "subdir": "skills"}

    with pytest.raises(RuntimeError, match="Username"):
        sc.sync_source(spec, cache_dir=cache)
    assert (old / "SKILL.md").exists()
    assert (cache / "x-y" / "SOURCE_REF").read_text() == "main\n"
    assert not (cache / "x-y.previous").exists()


def test_sync_source_indexes_per_source_collection(tmp_path, monkeypatch):
    # Fake clone: populate dest/<subdir> with two skills.
    def fake_clone(url, dest, branch="main", include=None):
        sub = include[0] if include else ""
        base = dest / sub if sub else dest
        _mkskill(base / "s1", "s1")
        _mkskill(base / "s2", "s2")

    monkeypatch.setattr("dsagt.commands.setup_core_kb.clone_github", fake_clone)

    kb = _FakeKB(tmp_path / "kb_index")
    cache = tmp_path / "cache"
    stats = sc.sync_source(
        {"url": "https://github.com/x/y", "branch": "main", "subdir": "skills"},
        kb=kb,
        cache_dir=cache,
    )
    slug = sc._repo_slug("https://github.com/x/y")
    coll = catalog_collection(slug)
    assert stats["discovered"] == 2 and stats["indexed"] == 2
    assert coll.startswith(CATALOG_COLLECTION_PREFIX)
    added_coll, metas = kb.adds[-1]
    assert added_coll == coll
    assert all(m["source"] == f"catalog:{slug}" for m in metas)
    assert {m["skill_name"] for m in metas} == {"s1", "s2"}


# ---------------------------------------------------------------------------
# native mirror
# ---------------------------------------------------------------------------


def test_mirror_manifest_preserves_user_skills_and_reaps(tmp_path):
    target = tmp_path / ".claude" / "skills"
    target.mkdir(parents=True)
    # A user-authored skill dsagt must never touch.
    _mkskill(target / "user-skill", "user-skill")

    bundled = _mkskill(tmp_path / "bundled" / "skill-creator", "skill-creator")
    proj = _mkskill(tmp_path / "proj" / "alpha", "alpha")

    _mirror_skills_to(target, [bundled, proj])
    assert sorted(p.name for p in target.iterdir() if p.is_dir()) == [
        "alpha",
        "skill-creator",
        "user-skill",
    ]
    manifest = json.loads((target / _SKILL_MANIFEST).read_text())
    assert manifest == ["alpha", "skill-creator"]
    assert "user-skill" not in manifest

    # Re-run with skill-creator gone → reaped; user-skill preserved.
    _mirror_skills_to(target, [proj])
    assert sorted(p.name for p in target.iterdir() if p.is_dir()) == [
        "alpha",
        "user-skill",
    ]


def test_rewrite_cli_invocations_to_the_registered_code(tmp_path):
    """A base skill whose CLI is a registered code shows the code's executable
    in fenced and inline examples after install; other spellings stay, and
    PROVENANCE.txt records the rewrite."""
    skill = tmp_path / "skills" / "aidrin"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: aidrin\ndescription: d\n---\n"
        "Run `aidrin run completeness <file>` or, in a fence:\n"
        "```bash\naidrin list\n  aidrin data-quality f.csv --detail\n```\n"
        "Already wrapped: `dsagt-run --code aidrin -- aidrin list`.\n"
        "Untouched: `aidrin`, aidrin-mcp, `uv run aidrin list`, skills/aidrin/x.\n"
    )
    (skill / "reference").mkdir()
    (skill / "reference" / "metrics.md").write_text("`aidrin run duplicity <file>`\n")
    (skill / "PROVENANCE.txt").write_text("Installed by dsagt from catalog source: x\n")

    pairs = [("aidrin", "dsagt-run --code aidrin -- aidrin")]
    assert sc.rewrite_cli_invocations(skill, pairs) == 4
    text = (skill / "SKILL.md").read_text()
    assert "Run `dsagt-run --code aidrin -- aidrin run completeness <file>`" in text
    assert "\ndsagt-run --code aidrin -- aidrin list\n" in text
    assert "\n  dsagt-run --code aidrin -- aidrin data-quality f.csv --detail\n" in text
    assert (
        text.count("dsagt-run --code aidrin -- aidrin list") == 2
    )  # one was already wrapped
    assert (
        "Untouched: `aidrin`, aidrin-mcp, `uv run aidrin list`, skills/aidrin/x."
        in text
    )
    assert (skill / "reference" / "metrics.md").read_text() == (
        "`dsagt-run --code aidrin -- aidrin run duplicity <file>`\n"
    )
    assert (
        "CLI examples rewritten by dsagt: `aidrin` -> `dsagt-run --code aidrin -- aidrin`"
        in (skill / "PROVENANCE.txt").read_text()
    )
    # A second pass changes nothing and stamps nothing more.
    assert sc.rewrite_cli_invocations(skill, pairs) == 0
    assert (skill / "PROVENANCE.txt").read_text().count("rewritten") == 1


def test_mirror_truncates_long_description(tmp_path):
    long_desc = "x" * (_NATIVE_DESCRIPTION_CAP + 500)
    src = _mkskill(tmp_path / "src" / "big", "big", desc=long_desc)
    target = tmp_path / ".claude" / "skills"
    _mirror_skills_to(target, [src])

    import yaml

    mirrored = (target / "big" / "SKILL.md").read_text()
    front = yaml.safe_load(mirrored.split("---", 2)[1])
    assert len(front["description"]) <= _NATIVE_DESCRIPTION_CAP
    # Source untouched.
    assert len((src / "SKILL.md").read_text()) > _NATIVE_DESCRIPTION_CAP


# ---------------------------------------------------------------------------
# AgentSetup.setup_skills — per-agent native-dir mirror
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent,subdir",
    [
        ("claude", ".claude/skills"),
        ("goose", ".agents/skills"),
        ("cline", ".cline/skills"),
        ("codex", ".agents/skills"),
        ("opencode", ".agents/skills"),
    ],
)
def test_setup_skills_mirrors_into_native_dir(tmp_path, agent, subdir):
    from dsagt.agents import AGENTS

    _mkskill(tmp_path / "skills" / "myskill", "myskill")  # a project skill
    actions = AGENTS[agent]().setup_skills(tmp_path, {})
    target = tmp_path
    for part in subdir.split("/"):
        target = target / part
    assert (target / "myskill" / "SKILL.md").exists()
    assert any("kill" in a for a in actions)  # reported a mirror action


def test_setup_skills_mirrors_registered_codes(tmp_path):
    """Registered codes share the skill envelope, so they mirror natively too."""
    from dsagt.agents import AGENTS
    from dsagt.registry import CodeRegistry

    CodeRegistry(runtime_dir=tmp_path).save_tool(
        {
            "name": "my-code",
            "description": "Use when testing the native code mirror",
            "executable": "echo hi",
            "parameters": {},
        }
    )
    AGENTS["claude"]().setup_skills(tmp_path, {})
    mirrored = tmp_path / ".claude" / "skills" / "my-code" / "SKILL.md"
    assert mirrored.exists()
    # The mirrored copy carries the exact dsagt-run command the agent must run.
    assert "dsagt-run --code my-code -- echo hi" in mirrored.read_text()


def test_setup_skills_project_skill_wins_code_name_collision(tmp_path):
    """A deliberately installed instruction skill outranks a same-named code."""
    from dsagt.agents import AGENTS
    from dsagt.registry import CodeRegistry

    CodeRegistry(runtime_dir=tmp_path).save_tool(
        {
            "name": "clash",
            "description": "the code",
            "executable": "echo code",
            "parameters": {},
        }
    )
    _mkskill(tmp_path / "skills" / "clash", "clash")
    AGENTS["claude"]().setup_skills(tmp_path, {})
    text = (tmp_path / ".claude" / "skills" / "clash" / "SKILL.md").read_text()
    assert "dsagt-run" not in text  # the skill copy, not the code copy


def test_setup_skills_respects_populate_native_false(tmp_path):
    from dsagt.agents import AGENTS

    _mkskill(tmp_path / "skills" / "myskill", "myskill")
    actions = AGENTS["claude"]().setup_skills(
        tmp_path, {"skills": {"populate_native": False}}
    )
    assert actions == []
    assert not (tmp_path / ".claude" / "skills").exists()


# ---------------------------------------------------------------------------
# install_into_project — license / attribution capture
# ---------------------------------------------------------------------------


def test_install_captures_ancestor_attribution(tmp_path):
    cache = tmp_path / "cache"
    repo = cache / "srcrepo"
    repo.mkdir(parents=True)
    (repo / "LICENSE").write_text("Apache-2.0")  # repo-root license
    cat = repo / "skills" / "modcon"
    cat.mkdir(parents=True)
    (cat / "ATTRIBUTION.md").write_text("upstream credits")  # per-subtree
    _mkskill(cat / "myskill", "myskill")

    proj = tmp_path / "proj"
    proj.mkdir()
    info = sc.install_into_project("myskill", proj, cache_dir=cache)
    dest = proj / "skills" / "myskill"
    assert (dest / "SKILL.md").exists()
    assert (dest / "ATTRIBUTION.md").read_text() == "upstream credits"
    assert (dest / "LICENSE").read_text() == "Apache-2.0"
    prov = (dest / "PROVENANCE.txt").read_text()
    assert "srcrepo" in prov and "skills/modcon/myskill" in prov
    assert set(info["attribution"]) == {"ATTRIBUTION.md", "LICENSE"}


def test_install_skill_local_license_wins(tmp_path):
    cache = tmp_path / "cache"
    repo = cache / "srcrepo"
    repo.mkdir(parents=True)
    (repo / "LICENSE").write_text("ROOT")  # repo-root license
    skill = _mkskill(repo / "myskill", "myskill")
    (skill / "LICENSE").write_text("SKILL-LOCAL")  # skill bundles its own

    proj = tmp_path / "proj"
    proj.mkdir()
    info = sc.install_into_project("myskill", proj, cache_dir=cache)
    dest = proj / "skills" / "myskill"
    # The skill's own LICENSE (copied by copytree) must not be overwritten.
    assert (dest / "LICENSE").read_text() == "SKILL-LOCAL"
    assert "LICENSE" not in info["attribution"]


# ---------------------------------------------------------------------------
# index_catalog — frontmatter-only embedding (progressive disclosure)
# ---------------------------------------------------------------------------


def test_index_catalog_embeds_frontmatter_not_body(tmp_path):
    captured = {}

    class _KB:
        index_dir = tmp_path / "idx"
        collections: list = []

        def add_entries(self, texts, collection, metadatas=None):
            captured["texts"] = texts
            captured["metas"] = metadatas
            return {}

    skill = tmp_path / "myskill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: myskill\ndescription: does a thing\ntags: [hpc, slurm]\n---\n"
        "# Body\nSECRET_BODY_MARKER should not be embedded.\n"
    )
    dirs = sc._discover_skill_dirs(tmp_path)
    sc.index_catalog(dirs, "slug", "http://x", _KB())

    joined = " ".join(captured["texts"])
    assert "myskill" in joined and "does a thing" in joined  # frontmatter embedded
    assert "hpc" in joined and "slurm" in joined  # tags embedded
    assert "SECRET_BODY_MARKER" not in joined  # body NOT embedded
    # description is also carried in metadata for the search summary.
    assert captured["metas"][0]["description"] == "does a thing"


def test_base_skills_reads_the_aidrin_version_when_called(monkeypatch):
    """The aidrin release tag comes from the installed package at call time,
    so an unparseable version fails the caller, never the import of
    dsagt.skills (which every skill tool needs)."""
    monkeypatch.setattr(sc, "installed_version", lambda name: "2027.1.5")
    aidrin = next(b for b in sc.base_skills() if b["name"] == "aidrin")
    assert aidrin["source"]["branch"] == "v2027.01.5"

    monkeypatch.setattr(sc, "installed_version", lambda name: "2027.1")
    with pytest.raises(ValueError):
        sc.base_skills()


def test_install_base_skills_finishes_the_others_when_one_fetch_fails(
    tmp_path, monkeypatch
):
    """A skill whose source cannot be fetched is reported after the loop; the
    skills before and after it are installed with their codes registered."""
    cache = tmp_path / "cache"

    def fake_sync(source, *, kb=None, cache_dir, force=False):
        if "AIDRIN" in source["url"]:
            raise RuntimeError("clone failed")
        slug = sc._repo_slug(source["url"])
        (cache_dir / slug).mkdir(parents=True, exist_ok=True)
        (cache_dir / slug / "SOURCE_COMMIT").write_text("c\n")
        for b in sc.base_skills():
            if sc.resolve_source(b["source"])["url"] == source["url"]:
                d = _mkskill(cache_dir / slug / "x" / b["name"], b["name"])
                for code in b.get("codes", ()):
                    (d / code["script"]).parent.mkdir(parents=True, exist_ok=True)
                    (d / code["script"]).write_text("print('ok')\n")
        return {"slug": slug}

    monkeypatch.setattr(sc, "sync_source", fake_sync)
    proj = tmp_path / "proj"
    with pytest.raises(RuntimeError, match="aidrin: clone failed"):
        sc.install_base_skills(proj, cache_dir=cache)
    assert (proj / "skills" / "skill-creator" / "SKILL.md").exists()
    assert (proj / "skills" / "datacard-generator" / "SKILL.md").exists()
    assert (proj / "skills" / "datacard-introspect" / "SKILL.md").exists()
    assert not (proj / "skills" / "aidrin").exists()


def test_base_skill_code_specs_hold_no_project_path():
    """Every base-skill code spec is project-independent: a script runs by a
    path relative to the project directory, so one embedding serves every
    project."""
    specs = sc.base_skill_code_specs()
    names = {s["name"] for s in specs}
    assert {"aidrin", "datacard-introspect", "datacard-validate"} <= names
    for spec in specs:
        assert not spec["executable"].startswith("/")
        assert spec["tags"] and spec["parameters"] is not None
    introspect = next(s for s in specs if s["name"] == "datacard-introspect")
    assert introspect["executable"].startswith(
        "python skills/datacard-generator/scripts/"
    )


def test_registration_indexes_into_the_kb(tmp_path):
    """With a knowledge base, each registered code is added to the ``codes``
    collection, so ``search_registry`` finds it."""
    from dsagt.registry import CODES_COLLECTION

    class FakeKB:
        def __init__(self):
            self.added = []

        def add_entries(self, *, texts, collection, metadatas):
            self.added.append((collection, [m["code_name"] for m in metadatas]))

    proj = tmp_path / "proj"
    _base_skill_dirs(proj)
    kb = FakeKB()
    _register_base(proj, kb=kb)
    names = [n for coll, ns in kb.added if coll == CODES_COLLECTION for n in ns]
    assert "aidrin" in names and "datacard-introspect" in names


def test_a_previous_clone_left_behind_is_not_a_source(tmp_path, monkeypatch):
    """``<slug>.previous`` is the clone a re-clone set aside; one left by a
    sync that died is skipped by every scanner and removed by the next sync."""
    cache = tmp_path / "cache"
    live = _mkskill(cache / "x-y" / "skills" / "s1", "s1")
    (cache / "x-y" / "SOURCE_REF").write_text("main\n")
    _mkskill(cache / "x-y.previous" / "skills" / "s1", "s1")

    # One source, not an ambiguous pair.
    assert sc.find_catalog_skill("s1", cache_dir=cache) == live
    names = [c["name"] for c in sc.SkillsCatalog(cache_dir=cache)._candidate_skills()]
    assert names == ["s1"]

    def no_clone(url, dest, branch="main", include=None):
        raise AssertionError("cached clone at the requested ref is reused")

    monkeypatch.setattr("dsagt.commands.setup_core_kb.clone_github", no_clone)
    sc.sync_source(
        {"url": "https://github.com/x/y", "branch": "main", "subdir": "skills"},
        cache_dir=cache,
    )
    assert not (cache / "x-y.previous").exists()


def test_mirror_is_a_relative_symlink_to_the_live_skill(tmp_path):
    """The agent reads the live files: a script edited under skills/ is what
    the mirrored skill runs, with no copy to go stale."""
    src = _mkskill(tmp_path / "proj" / "skills" / "alpha", "alpha")
    (src / "scripts").mkdir()
    (src / "scripts" / "a.py").write_text("print(1)\n")
    target = tmp_path / "proj" / ".claude" / "skills"
    _mirror_skills_to(target, [src])
    link = target / "alpha"
    assert link.is_symlink()
    assert not os.path.isabs(os.readlink(link))
    assert (link / "scripts" / "a.py").read_text() == "print(1)\n"
    (src / "scripts" / "a.py").write_text("print(2)\n")
    assert (link / "scripts" / "a.py").read_text() == "print(2)\n"
    # A second pass replaces the link in place; reaping removes it.
    _mirror_skills_to(target, [src])
    assert link.is_symlink()
    _mirror_skills_to(target, [])
    assert not link.exists() and not link.is_symlink()


def test_a_line_naming_two_scripts_wraps_both(tmp_path):
    """The guard is per command, not per line.

    A line mentioning two scripts held a dsagt-run once the first pair was
    applied, and reading that as "already wrapped" left the second bare.
    """
    from dsagt.skills import rewrite_cli_invocations

    skill = tmp_path / "x"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "Run python scripts/extract.py and python scripts/load.py in order.\n"
    )
    pairs = [
        (
            "python scripts/extract.py",
            "dsagt-run --code x-extract -- python scripts/extract.py",
        ),
        ("python scripts/load.py", "dsagt-run --code x-load -- python scripts/load.py"),
    ]
    rewrite_cli_invocations(skill, pairs)
    text = (skill / "SKILL.md").read_text()
    assert "dsagt-run --code x-extract -- python scripts/extract.py" in text
    assert "dsagt-run --code x-load -- python scripts/load.py" in text

    # Applying the same pairs again leaves the text alone.
    rewrite_cli_invocations(skill, pairs)
    assert (skill / "SKILL.md").read_text() == text


def test_a_script_code_name_is_valid_for_a_skill_named_with_underscores(tmp_path):
    """save_tool refuses a name outside the lowercase-hyphen charset."""
    from pathlib import Path

    from dsagt.registry import _CODE_NAME_RE
    from dsagt.skills import script_code_name

    name = script_code_name("csv_inspector", Path("foo_bar.py"))
    assert name == "csv-inspector-foo-bar"
    assert _CODE_NAME_RE.match(name)
