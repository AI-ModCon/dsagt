"""
Tests for the knowledge-base asset builder (dsagt.commands.setup_core_kb),
the engine behind ``dsagt init``'s KB provisioning.

These cover the helpers that don't require network access — the git clone
subprocess is mocked so the tests can run offline.  The actual end-to-end
behavior against real upstream repos is exercised manually.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import numpy as np

from dsagt.commands.setup_core_kb import (
    DEFAULT_ASSETS,
    DEFAULT_EXCLUDE_PATTERNS,
    all_assets,
    asset_collection_name,
    clone_github,
    ensure_assets,
    resolve_assets,
)

# ---------------------------------------------------------------------------
# clone_github top-level-files behavior
# ---------------------------------------------------------------------------


def _fake_clone(fake_repo: Path):
    """Build a function that simulates `git clone` by copying *fake_repo*
    into the destination passed by clone_github.

    clone_github invokes ``subprocess.run(["git", "clone", ..., dest])``.
    Our patched subprocess.run reads the dest from the args and copies
    the fake repo there, returning a successful CompletedProcess.
    """

    def _run(cmd, capture_output=True, text=True, **kwargs):
        if cmd[1] == "clone":
            # ["git", "clone", "--depth", "1", "--branch", "main", url, dest]
            shutil.copytree(fake_repo, Path(cmd[-1]))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        assert cmd[1:2] == ["-C"] and cmd[-2:] == ["rev-parse", "HEAD"], cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="fake0commit\n", stderr="")

    return _run


def _fake_github(fake_repo: Path, sha: str = "fake0commit", status: int = 200):
    """An httpx transport that answers the two GitHub API calls
    :func:`fetch_github_tree` makes: the commit for a ref, and the tarball at
    that commit, built from *fake_repo* under GitHub's one-directory layout."""
    import io
    import tarfile

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"message": "Not Found"})
        path = request.url.path
        if "/commits/" in path:
            return httpx.Response(200, json={"sha": sha})
        assert "/tarball/" in path, path
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(fake_repo, arcname=f"owner-fake-{sha[:7]}")
        return httpx.Response(200, content=buf.getvalue())

    return httpx.MockTransport(handler)


def _patched_client(transport):
    """``httpx.Client`` bound to *transport*, for patching into the module.

    Patching ``setup_core_kb.httpx.Client`` patches the httpx module itself,
    so the real class is captured first."""
    real_client = httpx.Client
    return lambda **kw: real_client(transport=transport, **kw)


@pytest.fixture
def fake_repo(tmp_path):
    """A fake repo with the structure of a typical Python library:
    docs/, src/, tests/, plus top-level packaging metadata.
    """
    repo = tmp_path / "fake_repo"
    repo.mkdir()

    # Top-level metadata files (the things we want to make sure survive)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "fakelib"\nversion = "1.2.3"\n'
        'dependencies = ["numpy>=1.26"]\n'
    )
    (repo / "setup.py").write_text(
        'from setuptools import setup\nsetup(name="fakelib")\n'
    )
    (repo / "README.md").write_text("# fakelib\n\nA fake library for tests.\n")
    (repo / "LICENSE").write_text("Apache 2.0\n")

    # Subdirectories
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text("# Guide\n\nUse fakelib.\n")

    (repo / "fakelib").mkdir()
    (repo / "fakelib" / "__init__.py").write_text('"""fakelib package."""\n')
    (repo / "fakelib" / "core.py").write_text("def f():\n    return 1\n")

    (repo / "tests").mkdir()
    (repo / "tests" / "test_core.py").write_text("def test_f():\n    assert True\n")

    return repo


def test_clone_with_include_keeps_top_level_files(fake_repo, tmp_path):
    """When `include` is set, clone_github should still copy top-level
    files (pyproject.toml, README.md, LICENSE) along with the requested
    subdirectories.  These contain critical packaging metadata that the
    agent uses to install dependencies when registering tools."""
    dest = tmp_path / "dest"
    dest.mkdir()

    with patch(
        "dsagt.commands.setup_core_kb.httpx.Client",
        _patched_client(_fake_github(fake_repo)),
    ):
        clone_github(
            url="https://github.com/owner/fake.git",
            dest=dest,
            branch="main",
            include=["docs", "fakelib"],
        )

    # Subdirectories that were requested.
    assert (dest / "docs" / "guide.md").exists()
    assert (dest / "fakelib" / "core.py").exists()

    # Top-level files that should survive even though they weren't in include.
    assert (dest / "pyproject.toml").exists()
    assert (dest / "setup.py").exists()
    assert (dest / "README.md").exists()
    assert (dest / "LICENSE").exists()

    # Subdirs not in include must NOT be copied.
    assert not (dest / "tests").exists()
    # The clone's commit and ref are recorded for consumers of the cache.
    assert (dest / "SOURCE_COMMIT").read_text() == "fake0commit\n"
    assert (dest / "SOURCE_REF").read_text() == "main\n"


def test_a_private_repository_falls_back_to_the_users_git(fake_repo, tmp_path):
    """The API refuses a private repository without a token; the user's git,
    with their ssh key, is the one fallback."""
    dest = tmp_path / "dest"
    with (
        patch(
            "dsagt.commands.setup_core_kb.httpx.Client",
            _patched_client(_fake_github(fake_repo, status=404)),
        ),
        patch("dsagt.commands.setup_core_kb.subprocess.run", _fake_clone(fake_repo)),
    ):
        clone_github(url="git@github.com:owner/fake.git", dest=dest, branch="main")
    assert (dest / "pyproject.toml").exists()
    assert (dest / "SOURCE_COMMIT").read_text() == "fake0commit\n"


def test_a_non_github_url_uses_git(fake_repo, tmp_path):
    dest = tmp_path / "dest"
    with patch("dsagt.commands.setup_core_kb.subprocess.run", _fake_clone(fake_repo)):
        clone_github(url="https://gitlab.example.org/o/r.git", dest=dest, branch="main")
    assert (dest / "README.md").exists()


def test_github_owner_repo_parses_both_forms():
    from dsagt.commands.setup_core_kb import _github_owner_repo

    assert _github_owner_repo("https://github.com/AI-ModCon/dsagt") == (
        "AI-ModCon",
        "dsagt",
    )
    assert _github_owner_repo("https://github.com/AI-ModCon/dsagt.git/") == (
        "AI-ModCon",
        "dsagt",
    )
    assert _github_owner_repo("git@github.com:idtlab/AIDRIN.git") == (
        "idtlab",
        "AIDRIN",
    )
    assert _github_owner_repo("https://gitlab.com/o/r") is None


def test_clone_without_include_copies_everything(fake_repo, tmp_path):
    """When include is None, clone_github copies the whole repo (minus .git)."""
    dest = tmp_path / "dest"
    # clone_github copies into dest, which must NOT pre-exist when include=None
    # because shutil.copytree(dirs_exist_ok=True) is used.

    with patch(
        "dsagt.commands.setup_core_kb.httpx.Client",
        _patched_client(_fake_github(fake_repo)),
    ):
        clone_github(
            url="https://github.com/owner/fake.git",
            dest=dest,
            branch="main",
        )

    assert (dest / "pyproject.toml").exists()
    assert (dest / "fakelib" / "core.py").exists()
    assert (dest / "tests" / "test_core.py").exists()


def test_default_exclude_patterns_keeps_packaging_metadata():
    """Regression: pyproject.toml, setup.py, setup.cfg must NOT be in
    DEFAULT_EXCLUDE_PATTERNS.  The agent needs them to install deps."""
    forbidden = {"pyproject.toml", "setup.py", "setup.cfg"}
    overlap = forbidden & set(DEFAULT_EXCLUDE_PATTERNS)
    assert not overlap, (
        f"DEFAULT_EXCLUDE_PATTERNS contains packaging metadata files "
        f"that the agent needs: {overlap}"
    )


# ---------------------------------------------------------------------------
# Installable-asset selection (--include / --exclude namespace)
# ---------------------------------------------------------------------------


class TestResolveAssets:

    def test_default_is_tools_plus_genesis(self):
        assert resolve_assets() == list(DEFAULT_ASSETS) == ["codes", "genesis"]

    def test_include_all_is_everything(self):
        assert resolve_assets(include=["all"]) == all_assets()

    def test_include_subset_returns_canonical_order(self):
        # input order shouldn't matter — cheap assets always built first.
        assert resolve_assets(include=["nemo_curator", "codes"]) == [
            "codes",
            "nemo_curator",
        ]

    def test_exclude_trims_the_default_set(self):
        assert resolve_assets(exclude=["genesis"]) == ["codes"]

    def test_exclude_all_is_empty(self):
        assert resolve_assets(exclude=["all"]) == []

    def test_unknown_asset_raises(self):
        with pytest.raises(ValueError, match="unknown KB asset"):
            resolve_assets(include=["not-a-real-asset"])

    def test_include_exclude_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_assets(include=["codes"], exclude=["genesis"])


class TestAssetCollectionName:

    def test_tools(self):
        assert asset_collection_name("codes") == "codes"

    def test_catalog_uses_catalog_prefix(self):
        name = asset_collection_name("genesis")
        assert name.startswith("skills_catalog__") and "genesis" in name

    def test_scientific_collection_is_its_own_name(self):
        assert asset_collection_name("nemo_curator") == "nemo_curator"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown KB asset"):
            asset_collection_name("bogus")


class TestEnsureAssetsTools:
    """``ensure_assets`` for the bundled-tools asset, with a mocked embedder
    so the test stays offline (no model download, no git clone)."""

    def _fake_embedder(self):
        emb = MagicMock()
        emb.embed = lambda texts: np.full((len(texts), 8), 0.1, dtype=np.float32)
        return emb

    def test_builds_tools_collection(self, tmp_path):
        with patch(
            "dsagt.knowledge.Embedder.create", return_value=self._fake_embedder()
        ):
            result = ensure_assets(["codes"], tmp_path)
        assert "codes" in result["built"]
        # ChromaIndex.save writes chroma_ids.json — the collection marker.
        assert (tmp_path / "codes" / "chroma_ids.json").exists()

    def test_is_idempotent(self, tmp_path):
        with patch(
            "dsagt.knowledge.Embedder.create", return_value=self._fake_embedder()
        ):
            ensure_assets(["codes"], tmp_path)
            second = ensure_assets(["codes"], tmp_path)
        assert second["skipped"] == ["codes"]
        assert second["built"] == []

    def test_rebuilds_when_the_stamp_differs(self, tmp_path):
        """The collection is rebuilt when its stamp is not the digest of the
        current dsagt version and entry texts, so an upgrade or a changed
        base-skill code refreshes the cache without a manual wipe."""
        from dsagt.commands.setup_core_kb import CODES_STAMP_FILE

        with patch(
            "dsagt.knowledge.Embedder.create", return_value=self._fake_embedder()
        ):
            ensure_assets(["codes"], tmp_path)
            stamp = tmp_path / "codes" / CODES_STAMP_FILE
            assert len(stamp.read_text().strip()) == 64
            stamp.write_text("stale\n")
            # A rebuild happens in a fresh ``dsagt init`` process; chromadb
            # caches one client per path within a process, so the cache is
            # cleared here to stand in for that process boundary.
            from chromadb.api.shared_system_client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
            third = ensure_assets(["codes"], tmp_path)
        assert third["built"] == ["codes"]
        assert len(stamp.read_text().strip()) == 64

    def test_build_holds_bundled_and_base_skill_codes(self, tmp_path):
        """The shared ``codes`` collection carries the package codes and the
        base-skill codes, each tagged by source, so a project that copies it
        needs no embedding at init."""
        from dsagt.commands.setup_core_kb import _build_bundled_tools
        from dsagt.registry import render_code_spec
        from dsagt.skills import base_skill_code_specs

        class FakeKB:
            def __init__(self):
                self.texts = []
                self.metadatas = []

            def add_entries(self, *, texts, collection, metadatas):
                assert collection == "codes"
                self.texts += texts
                self.metadatas += metadatas

        (tmp_path / "codes").mkdir()
        kb = FakeKB()
        n = _build_bundled_tools(kb, tmp_path)
        assert n == len(kb.texts)
        by_source = {}
        for m in kb.metadatas:
            by_source.setdefault(m["source"], set()).add(m["code_name"])
        assert {"aidrin", "datacard-introspect", "datacard-validate"} <= by_source[
            "base-skill"
        ]
        assert by_source["bundled"]
        expected = {render_code_spec(s) for s in base_skill_code_specs()}
        assert expected <= set(kb.texts)
        assert all("dsagt_version" in m for m in kb.metadatas)
