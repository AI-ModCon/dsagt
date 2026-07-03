"""Tests for ``dsagt traces`` — the frictionless MLflow viewer launcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dsagt.commands import traces as traces_cmd


def _config(pdir):
    return {"project": "proj", "project_dir": str(pdir), "agent": "claude"}


def test_missing_store_returns_error(tmp_path):
    pdir = tmp_path / "proj"
    pdir.mkdir()
    with patch.object(traces_cmd, "load_config", return_value=_config(pdir)):
        rc = traces_cmd.run("proj")
    assert rc == 1  # no mlflow.db yet


def test_runs_catchup_then_launches_deeplinked_quiet_viewer(tmp_path, capsys):
    pdir = tmp_path / "proj"
    pdir.mkdir()
    (pdir / "mlflow.db").write_text("")  # existence is all run() checks

    launched = {}

    def _fake_subprocess_run(cmd, env=None):
        launched["cmd"] = cmd
        launched["env"] = env
        return MagicMock(returncode=0)

    with (
        patch.object(traces_cmd, "load_config", return_value=_config(pdir)),
        patch.object(
            traces_cmd, "catch_up_extraction", return_value={"traces_caught_up": 2}
        ) as catchup,
        patch.object(traces_cmd, "_resolve_experiment_id", return_value="42"),
        patch.object(traces_cmd.subprocess, "run", _fake_subprocess_run),
    ):
        rc = traces_cmd.run("proj", port=5001)

    assert rc == 0
    catchup.assert_called_once()  # catch-up ran before the viewer

    cmd = launched["cmd"]
    assert cmd[:2] == ["mlflow", "ui"]
    assert "--workers" in cmd and cmd[cmd.index("--workers") + 1] == "1"
    assert cmd[cmd.index("--port") + 1] == "5001"
    assert cmd[cmd.index("--backend-store-uri") + 1].startswith("sqlite:///")
    # Quiet: warnings suppressed in the child env.
    assert launched["env"]["PYTHONWARNINGS"] == "ignore"

    out = capsys.readouterr().out
    # Deep-links to the project's Traces tab (not the empty default Runs view).
    assert "http://127.0.0.1:5001/#/experiments/42/traces" in out
    assert "Caught up 2" in out


def test_launch_survives_catchup_failure(tmp_path):
    pdir = tmp_path / "proj"
    pdir.mkdir()
    (pdir / "mlflow.db").write_text("")

    with (
        patch.object(traces_cmd, "load_config", return_value=_config(pdir)),
        patch.object(
            traces_cmd, "catch_up_extraction", side_effect=RuntimeError("boom")
        ),
        patch.object(traces_cmd, "_resolve_experiment_id", return_value=None),
        patch.object(
            traces_cmd.subprocess, "run", return_value=MagicMock(returncode=0)
        ),
    ):
        # A hiccup in catch-up must not stop the viewer from opening.
        assert traces_cmd.run("proj") == 0
