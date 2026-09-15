"""Unit tests for the external skill catalog (fetch / index / install) and
the native-skill mirror.  No network: ``clone_github`` is monkeypatched and
the KB is a lightweight fake that records ``add_entries`` calls."""

import json

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
    """Each base skill is installed by a source-qualified name from the
    shared source cache without a forced re-clone, replacing any existing
    project copy."""
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
    stale = _mkskill(proj / "skills" / "aidrin", "aidrin", desc="stale")

    results = sc.install_base_skills(proj, cache_dir=cache)
    assert [r["name"] for r in results] == [
        "skill-creator",
        "datacard-generator",
        "aidrin",
    ]
    # No forced re-clone: a cached source is reused as is.
    assert synced == [
        ("ai-modcon-genesis-skills", False),
        ("ai-modcon-genesis-skills", False),
        ("idtlab-aidrin", False),
    ]
    assert (proj / "skills" / "skill-creator" / "SKILL.md").exists()
    assert (proj / "skills" / "datacard-generator" / "SKILL.md").exists()
    assert "stale" not in (stale / "SKILL.md").read_text()
    provenance = (proj / "skills" / "aidrin" / "PROVENANCE.txt").read_text()
    assert "Commit: idtlab-aidrin-commit" in provenance
    # The scripts the datacard workflow runs, and the aidrin CLI, are codes.
    assert (proj / "codes" / "datacard-introspect" / "SKILL.md").exists()
    assert (proj / "codes" / "datacard-validate" / "SKILL.md").exists()
    assert (proj / "codes" / "aidrin" / "SKILL.md").exists()


def test_register_base_skill_codes_wraps_scripts_in_place(tmp_path):
    """Each base-skill code runs the installed skill's script through
    ``dsagt-run``, with ``uv run --with`` for the entries that declare
    dependencies, and a re-run updates instead of duplicating."""
    from dsagt.registry import CodeRegistry

    proj = tmp_path / "proj"
    for entry in sc.base_skills():
        for code in entry.get("codes", ()):
            if "script" not in code:
                continue
            script = proj / "skills" / entry["name"] / code["script"]
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("print('ok')\n")

    actions = sc.register_base_skill_codes(proj)
    assert actions == [
        "added datacard-introspect",
        "added datacard-validate",
        "added datacard-convert-v1",
        "added aidrin",
    ]
    registry = CodeRegistry(runtime_dir=proj)
    introspect = registry.get_code("datacard-introspect")
    assert introspect["executable"] == (
        "dsagt-run --code datacard-introspect -- "
        "python skills/datacard-generator/scripts/introspect.py"
    )
    assert introspect["parameters"]["dataset_dir"]["cli"] == "positional"
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

    assert sc.register_base_skill_codes(proj)[0] == "updated datacard-introspect"


def test_register_base_skill_codes_requires_the_script(tmp_path):
    proj = tmp_path / "proj"
    (proj / "skills" / "datacard-generator").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="datacard-generator"):
        sc.register_base_skill_codes(proj)


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

    pairs = sc.native_invocations()["aidrin"]
    assert pairs == [("aidrin", "dsagt-run --code aidrin -- aidrin")]
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
