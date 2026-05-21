"""
Tests for dsagt-setup-kb internals.

These cover the helpers in dsagt.commands.setup_core_kb that don't require
network access — the git clone subprocess is mocked so the tests can run
offline.  The actual end-to-end behavior of dsagt-setup-kb against real
upstream repos is exercised manually.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dsagt.commands.setup_core_kb import (
    DEFAULT_EXCLUDE_PATTERNS,
    clone_github,
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
