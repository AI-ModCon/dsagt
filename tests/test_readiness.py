"""Readiness assessment: config block, install, and the instructions block."""

import subprocess

import pytest

from dsagt import readiness as rd


class TestConfigBlock:

    def test_defaults_to_shared_install_path(self):
        block = rd.readiness_block("aidrin")
        assert block == {
            "tool": "aidrin",
            "executable": str(rd.AIDRIN_EXECUTABLE),
            "profile": "quality",
        }

    def test_explicit_executable_and_profile(self, tmp_path):
        block = rd.readiness_block(
            "aidrin", executable=tmp_path / "aidrin", profile="supervised"
        )
        assert block["executable"] == str(tmp_path / "aidrin")
        assert block["profile"] == "supervised"

    def test_rejects_unknown_tool_or_profile(self):
        with pytest.raises(ValueError):
            rd.readiness_block("nope")
        with pytest.raises(ValueError):
            rd.readiness_block("aidrin", profile="nope")


class TestEnsureAidrin:

    def test_skips_when_installed(self, tmp_path, monkeypatch):
        exe = tmp_path / "aidrin" / "bin" / "aidrin"
        exe.parent.mkdir(parents=True)
        exe.write_text("")
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: pytest.fail("must not run uv")
        )
        assert rd.ensure_aidrin(tmp_path / "aidrin") == exe

    def test_runs_venv_then_pip(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(rd.shutil, "which", lambda name: "/usr/bin/uv")
        install = tmp_path / "aidrin"
        assert rd.ensure_aidrin(install) == install / "bin" / "aidrin"
        assert calls[0][:2] == ["uv", "venv"] and calls[0][-1] == str(install)
        assert calls[1][:3] == ["uv", "pip", "install"]
        assert calls[1][-1] == rd.AIDRIN_SPEC

    def test_failure_raises_with_stderr_and_cleans_up(self, tmp_path, monkeypatch):
        install = tmp_path / "aidrin"

        def fake_run(cmd, **kw):
            install.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 1, "", "no network")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(rd.shutil, "which", lambda name: "/usr/bin/uv")
        with pytest.raises(RuntimeError, match="no network"):
            rd.ensure_aidrin(install)
        assert not install.exists()


class TestInstructionsBlock:

    def test_carries_marker_profile_metrics_and_executable(self, tmp_path):
        block = rd.readiness_block(
            "aidrin", executable=tmp_path / "aidrin", profile="quality"
        )
        text = rd.instructions_block(block)
        assert rd.READINESS_MARKER in text
        assert "`quality`: completeness, duplicity, outliers" in text
        assert "audit/step_N_pre.aidrin.json" in text
        # The agent runs the installed ``aidrin`` skill's CLI through dsagt-run.
        assert f"dsagt-run --code aidrin -- {tmp_path / 'aidrin'} run" in text
        assert "skills/aidrin/" in text
        # The marker must not collide with the master-instructions marker.
        assert "DSAgt Pipeline Builder" not in text
