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
        # cmd is e.g. ["git", "clone", "--depth", "1", "--branch", "main", url, dest]
        dest = Path(cmd[-1])
        shutil.copytree(fake_repo, dest)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return _run


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

    with patch("dsagt.commands.setup_core_kb.subprocess.run", _fake_clone(fake_repo)):
        clone_github(
            url="https://example.com/fake.git",
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


def test_clone_without_include_copies_everything(fake_repo, tmp_path):
    """When include is None, clone_github copies the whole repo (minus .git)."""
    dest = tmp_path / "dest"
    # clone_github copies into dest, which must NOT pre-exist when include=None
    # because shutil.copytree(dirs_exist_ok=True) is used.

    with patch("dsagt.commands.setup_core_kb.subprocess.run", _fake_clone(fake_repo)):
        clone_github(
            url="https://example.com/fake.git",
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
        assert resolve_assets(include=["aidrin", "codes"]) == ["codes", "aidrin"]

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
