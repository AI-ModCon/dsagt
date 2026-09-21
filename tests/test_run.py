"""
Tests for dsagt-run execution wrapper.

Covers argument parsing, command execution, record writing,
exit code propagation, error handling, and env var fallbacks.
"""

import json
import sys
from pathlib import Path

import pytest

from dsagt.provenance import (
    _resolve_records_dir,
    _write_record,
    run_and_record,
)
from dsagt.commands.run_code import (
    _parse_args,
    main,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _project(tmp_path, monkeypatch):
    """A project directory the command runs from; returns its trace_archive."""
    (tmp_path / ".dsagt").mkdir(exist_ok=True)
    (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "trace_archive"


class TestParseArgs:

    def test_basic(self):
        args, command = _parse_args(["--code", "fastp", "--", "fastp", "-q", "20"])
        assert args.code == "fastp"
        assert command == ["fastp", "-q", "20"]

    def test_no_separator_exits(self):
        """Missing '--' separator causes a SystemExit (from argparse --help)."""
        with pytest.raises(SystemExit):
            _parse_args(["--code", "fastp", "fastp", "-q", "20"])


class TestFileRolesFromCommand:
    """The record's input and output files come from the spec's parameter
    roles, read off the command line the way the parameters' ``cli`` says."""

    SPEC = {
        "name": "conv",
        "executable": "dsagt-run --code conv -- python codes/conv/scripts/conv.py",
        "parameters": {
            "src": {"type": "string", "cli": "positional", "role": "input"},
            "out": {"type": "string", "cli": "--out", "role": "output"},
            "grid": {"type": "string", "cli": "--grid=", "role": "input"},
            "verbose": {"type": "boolean", "cli": "-v"},
            "n": {"type": "integer", "cli": "-n"},
        },
    }

    def test_reads_positional_spaced_and_glued_values(self):
        from dsagt.provenance import file_roles_from_command

        cmd = [
            "python", "codes/conv/scripts/conv.py",
            "-v", "-n", "3", "data/in.h5", "--out", "data/out.h5", "--grid=data/grid.npy",
        ]  # fmt: skip
        inputs, outputs = file_roles_from_command(self.SPEC, cmd)
        assert inputs == ["data/in.h5", "data/grid.npy"]
        assert outputs == ["data/out.h5"]

    def test_absent_parameters_name_nothing(self):
        from dsagt.provenance import file_roles_from_command

        cmd = ["python", "codes/conv/scripts/conv.py", "data/in.h5"]
        assert file_roles_from_command(self.SPEC, cmd) == (["data/in.h5"], [])

    def test_a_command_that_is_not_the_spec_executable_names_nothing(self):
        """A different prefix is not this code's invocation; the roles are
        not applied to it."""
        from dsagt.provenance import file_roles_from_command

        cmd = ["python3", "other.py", "data/in.h5", "--out", "x"]
        assert file_roles_from_command(self.SPEC, cmd) == ([], [])


class TestResolveRecordsDir:

    def test_uses_cwd_dsagt_config(self, tmp_path, monkeypatch):
        """With no DSAGT_PROJECT_DIR the cwd is the project: reads
        ``<cwd>/.dsagt/config.yaml`` and uses ``<cwd>/trace_archive``."""
        monkeypatch.delenv("DSAGT_PROJECT_DIR", raising=False)
        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
        monkeypatch.chdir(tmp_path)
        assert _resolve_records_dir() == tmp_path / "trace_archive"

    def test_project_dir_env_wins_over_cwd(self, tmp_path, monkeypatch):
        """``DSAGT_PROJECT_DIR`` (exported by ``dsagt start``) names the
        project even when the command runs from a subdirectory."""
        project = tmp_path / "proj"
        (project / ".dsagt").mkdir(parents=True)
        (project / ".dsagt" / "config.yaml").write_text("project: t\n")
        (project / "data").mkdir()
        monkeypatch.setenv("DSAGT_PROJECT_DIR", str(project))
        monkeypatch.chdir(project / "data")
        assert _resolve_records_dir() == project / "trace_archive"

    def test_no_config_in_cwd_raises(self, tmp_path, monkeypatch):
        """A cwd without .dsagt/config.yaml fails with one line naming the
        rule; there is no walk up the tree."""
        monkeypatch.delenv("DSAGT_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not a dsagt project"):
            _resolve_records_dir()


# ---------------------------------------------------------------------------
# Record writing
# ---------------------------------------------------------------------------


class TestWriteRecord:

    def test_creates_directory_and_file(self, tmp_path):
        records_dir = tmp_path / "nested" / "records"
        record = {
            "record_id": "abc123",
            "code_name": "fastp",
            "session_id": None,
            "execution": {
                "exact_command": ["fastp", "-q", "20"],
                "return_code": 0,
                "stdout": "done\n",
                "stderr": "",
                "timestamp_start": "2025-01-01T00:00:00+00:00",
                "timestamp_end": "2025-01-01T00:00:01+00:00",
                "input_files": [],
                "output_files": [],
            },
        }
        path = _write_record(record, records_dir)

        assert path.exists()
        assert path.suffix == ".json"
        assert "fastp" in path.name
        assert "abc123" in path.name

        data = json.loads(path.read_text())
        assert data["code_name"] == "fastp"
        assert data["execution"]["return_code"] == 0

    def test_record_is_valid_json(self, tmp_path):
        record = {
            "record_id": "x",
            "code_name": "t",
            "session_id": None,
            "execution": {
                "exact_command": ["echo"],
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "timestamp_start": "",
                "timestamp_end": "",
                "input_files": [],
                "output_files": [],
            },
        }
        path = _write_record(record, tmp_path)
        # Should parse without error
        json.loads(path.read_text())


# ---------------------------------------------------------------------------
# run_and_record
# ---------------------------------------------------------------------------


class TestRunAndRecord:

    def test_successful_command(self, tmp_path):
        """Runs echo, captures output, writes record, returns 0."""
        exit_code = run_and_record(
            code_name="echo_test",
            command=["echo", "hello world"],
            records_dir=tmp_path,
            record_id="test-001",
        )

        assert exit_code == 0

        records = list(tmp_path.glob("*.json"))
        assert len(records) == 1

        data = json.loads(records[0].read_text())
        assert data["code_name"] == "echo_test"
        assert data["record_id"] == "test-001"
        assert data["execution"]["return_code"] == 0
        assert "hello world" in data["execution"]["stdout"]
        assert data["execution"]["exact_command"] == ["echo", "hello world"]

    def test_failing_command(self, tmp_path):
        """A command that fails returns non-zero and captures stderr."""
        exit_code = run_and_record(
            code_name="false_test",
            command=["bash", "-c", "echo oops >&2; exit 42"],
            records_dir=tmp_path,
            record_id="test-002",
        )

        assert exit_code == 42

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["return_code"] == 42
        assert "oops" in data["execution"]["stderr"]

    def test_command_not_found(self, tmp_path):
        """A missing command returns exit code 127."""
        exit_code = run_and_record(
            code_name="missing",
            command=["this_command_does_not_exist_xyz"],
            records_dir=tmp_path,
            record_id="test-003",
        )

        assert exit_code == 127

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["return_code"] == 127
        assert "command not found" in data["execution"]["stderr"]

    def test_session_from_state(self, tmp_path, monkeypatch):
        """Session ID falls back to the current tag in ``.dsagt/state.yaml``
        when not passed — the MCP server mints it there at startup and
        ``dsagt-run`` (cwd == project dir) reads it."""
        from dsagt.session import append_session, write_config_file, build_config

        write_config_file(tmp_path, build_config("t", "claude"))
        append_session(tmp_path)  # mints session id 1 → tag "t-1"
        monkeypatch.chdir(tmp_path)
        run_and_record(
            code_name="t",
            command=["echo"],
            records_dir=tmp_path,
            record_id="test-004",
        )

        data = json.loads(list(tmp_path.glob("*_test-004.json"))[0].read_text())
        assert data["session_id"] == "t-1"

    def test_explicit_session_overrides_state(self, tmp_path, monkeypatch):
        """Explicit --session takes precedence over the state-file tag."""
        from dsagt.session import append_session, write_config_file, build_config

        write_config_file(tmp_path, build_config("t", "claude"))
        append_session(tmp_path)
        monkeypatch.chdir(tmp_path)
        run_and_record(
            code_name="t",
            command=["echo"],
            records_dir=tmp_path,
            session_id="explicit-session",
            record_id="test-005",
        )

        data = json.loads(list(tmp_path.glob("*_test-005.json"))[0].read_text())
        assert data["session_id"] == "explicit-session"

    def test_file_lists_recorded(self, tmp_path):
        """Input and output file lists appear in the record."""
        run_and_record(
            code_name="t",
            command=["echo"],
            records_dir=tmp_path,
            input_files=["in1.fq", "in2.fq"],
            output_files=["out.fa"],
            record_id="test-006",
        )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["input_files"] == ["in1.fq", "in2.fq"]
        assert data["execution"]["output_files"] == ["out.fa"]

    def test_timestamps_are_populated(self, tmp_path):
        """Start and end timestamps are non-empty ISO strings."""
        run_and_record(
            code_name="t",
            command=["echo"],
            records_dir=tmp_path,
            record_id="test-007",
        )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["timestamp_start"]
        assert data["execution"]["timestamp_end"]
        assert (
            data["execution"]["timestamp_start"] <= data["execution"]["timestamp_end"]
        )

    def test_output_is_echoed_while_the_command_runs(self, tmp_path, monkeypatch):
        """A line the command prints is echoed before the command exits, so a
        slow code shows progress instead of looking hung until it finishes."""
        import sys
        import time

        class TimedWriter:
            def __init__(self):
                self.first_write_at = None

            def write(self, text):
                if self.first_write_at is None and text.strip():
                    self.first_write_at = time.monotonic()

            def flush(self):
                pass

        writer = TimedWriter()
        monkeypatch.setattr(sys, "stdout", writer)
        run_and_record(
            code_name="slow",
            command=[
                sys.executable,
                "-c",
                "import time; print('started', flush=True); "
                "time.sleep(1.0); print('done')",
            ],
            records_dir=tmp_path,
            record_id="test-008",
        )
        finished_at = time.monotonic()

        assert writer.first_write_at is not None
        assert (
            finished_at - writer.first_write_at >= 0.9
        ), "the first line was echoed only after the command exited"
        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["stdout"] == "started\ndone\n"

    def test_auto_generates_record_id(self, tmp_path):
        """Omitting record_id auto-generates one."""
        run_and_record(
            code_name="t",
            command=["echo"],
            records_dir=tmp_path,
        )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["record_id"]
        assert len(data["record_id"]) == 12


# ---------------------------------------------------------------------------
# main() CLI entry point
# ---------------------------------------------------------------------------


class TestMain:

    @pytest.fixture(autouse=True)
    def _project_and_store(self, tmp_path, monkeypatch):
        """Run from a project directory, as dsagt-run does, and point MLflow
        at a scratch store so init_tracing has a real backend.  The project
        name comes from ``.dsagt/config.yaml`` and the session id from
        ``.dsagt/state.yaml``, both under the working directory."""
        (tmp_path / ".dsagt").mkdir(exist_ok=True)
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: test\n")
        monkeypatch.chdir(tmp_path)
        # Serverless: init_tracing resolves a sqlite store from the project
        # dir via MLflow's native provider — no OTLP exporter.  Stub the
        # resolver to a known sqlite URI so a shell-set MLFLOW_TRACKING_URI
        # can't redirect the test.
        from dsagt import observability as obs_module

        cfg = {"project": "test"}
        monkeypatch.setattr(
            obs_module,
            "find_project_config",
            lambda: (tmp_path, cfg),
        )
        monkeypatch.setattr(
            obs_module,
            "resolve_tracking_uri",
            lambda c: f"sqlite:///{tmp_path}/mlflow.db",
        )

    def test_trace_root_carries_the_minted_session(self, tmp_path, monkeypatch):
        """The MCP server mints the session into ``.dsagt/state.yaml``; the
        ``code.execute`` root must carry it, or every execution trace lands
        in an unbucketed ``(no-session)`` group in ``dsagt info``."""
        import mlflow

        from dsagt import observability as obs_module

        (tmp_path / ".dsagt" / "state.yaml").write_text(
            "sessions:\n- id: 7\n  started_at: '2026-01-01T00:00:00Z'\n"
        )
        monkeypatch.setattr(obs_module, "_initialized", False)
        monkeypatch.setattr(obs_module, "_default_session_id", None)

        from dsagt.commands import log_trace as log_trace_module
        from dsagt.commands import run_code

        # The trace is logged by a detached process; run that step in-process.
        monkeypatch.setattr(
            run_code,
            "_log_trace_detached",
            lambda project_dir: lambda path: log_trace_module.log_trace(path),
        )
        assert main(["--code", "t", "--", "true"]) == 0

        assert obs_module._default_session_id == "test-7"
        trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
        assert trace.info.trace_metadata.get("mlflow.trace.session") == "test-7"

    def test_the_trace_is_logged_by_a_detached_process(self, tmp_path, monkeypatch):
        """``main`` loads no trace store: it hands the record's path to a
        ``dsagt.commands.log_trace`` process started in the project
        directory."""
        import subprocess

        from dsagt.commands import run_code

        started = []
        real_popen = subprocess.Popen

        def popen(argv, **kw):
            if "dsagt.commands.log_trace" not in argv:
                return real_popen(argv, **kw)
            started.append((argv, kw))

        monkeypatch.setattr(subprocess, "Popen", popen)
        records = tmp_path / "trace_archive"
        assert main(["--code", "t", "--", "true"]) == 0

        [(argv, kw)] = started
        [record] = records.glob("*.json")
        assert argv[1:4] == [
            "-m",
            "dsagt.commands.log_trace",
            str(record.resolve()),
        ]
        assert kw["cwd"] == tmp_path and kw["start_new_session"] is True
        assert run_code._log_trace_detached(tmp_path / "elsewhere") is None

    def test_record_files_come_from_the_spec_roles(self, tmp_path, monkeypatch):
        """Without --input-files/--output-files, dsagt-run reads the spec of
        --code from the project and records the files its role parameters
        name, so the dependency graph has edges without the agent passing
        the flags."""
        from dsagt.registry import CodeRegistry

        project = tmp_path / "proj"
        (project / ".dsagt").mkdir(parents=True)
        (project / ".dsagt" / "config.yaml").write_text("project: t\n")
        script = project / "codes" / "conv" / "scripts" / "conv.py"
        script.parent.mkdir(parents=True)
        script.write_text("import sys; print(sys.argv[1:])\n")
        CodeRegistry(runtime_dir=project).save_tool(
            {
                "name": "conv",
                "description": "Convert.",
                "executable": "python codes/conv/scripts/conv.py",
                "parameters": {
                    "src": {
                        "type": "string",
                        "description": "in",
                        "cli": "positional",
                        "role": "input",
                    },
                    "out": {
                        "type": "string",
                        "description": "out",
                        "cli": "--out",
                        "role": "output",
                    },
                },
            }
        )
        monkeypatch.delenv("DSAGT_PROJECT_DIR", raising=False)
        monkeypatch.chdir(project)
        rc = main(
            ["--code", "conv", "--", "python", "codes/conv/scripts/conv.py",
             "data/in.csv", "--out", "data/out.csv"]
        )  # fmt: skip
        assert rc == 0
        records = list((project / "trace_archive").glob("*.json"))
        assert len(records) == 1
        execution = json.loads(records[0].read_text())["execution"]
        assert execution["input_files"] == ["data/in.csv"]
        assert execution["output_files"] == ["data/out.csv"]

    def test_basic_invocation(self, tmp_path):
        """main() runs a command and returns its exit code."""
        exit_code = main(
            [
                "--code",
                "echo_tool",
                "--",
                "echo",
                "from main",
            ]
        )

        assert exit_code == 0
        records = list((tmp_path / "trace_archive").glob("*.json"))
        assert len(records) == 1

    def test_empty_command_returns_1(self, tmp_path):
        """No command after '--' returns exit code 1."""
        exit_code = main(
            [
                "--code",
                "empty",
                "--",
            ]
        )
        assert exit_code == 1

    def test_exit_code_propagation(self, tmp_path):
        """main() returns the wrapped command's exit code."""
        exit_code = main(
            [
                "--code",
                "fail",
                "--",
                "bash",
                "-c",
                "exit 7",
            ]
        )
        assert exit_code == 7


class TestChildEnv:
    """dsagt-run resolves a command from dsagt's own environment when PATH
    lacks it, which is the case under pipx and ``uv tool install``."""

    def test_interpreter_dir_is_appended_once(self, monkeypatch):
        import os
        import sys

        from dsagt.provenance import _child_env

        bin_dir = str(Path(sys.executable).parent)
        monkeypatch.setenv("PATH", "/usr/bin")
        assert _child_env()["PATH"] == f"/usr/bin{os.pathsep}{bin_dir}"
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin")
        assert _child_env()["PATH"] == f"{bin_dir}{os.pathsep}/usr/bin"

    def test_command_from_the_interpreter_dir_runs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory
        exit_code = run_and_record(
            code_name="py",
            command=["python", "-c", "print('found')"],
            records_dir=tmp_path,
            record_id="env-001",
        )
        assert exit_code == 0
        (record,) = tmp_path.glob("*.json")
        assert "found" in json.loads(record.read_text())["execution"]["stdout"]


# ---------------------------------------------------------------------------
# A run ended by a signal
# ---------------------------------------------------------------------------


class TestSignal:

    def test_a_terminated_run_still_writes_its_record(self, tmp_path):
        """SIGTERM to dsagt-run reaches the child and the record says so."""
        import os
        import signal
        import subprocess
        import sys
        import time

        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dsagt.commands.run_code",
                "--code",
                "sleeper",
                "--",
                "sleep",
                "30",
            ],
            cwd=str(tmp_path),
        )
        time.sleep(1.5)
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
        records = list((tmp_path / "trace_archive").glob("*.json"))
        assert len(records) == 1
        record = json.loads(records[0].read_text())
        assert record["execution"]["return_code"] == -signal.SIGTERM
        assert "SIGTERM" in record["execution"]["stderr"]
        assert proc.returncode != 0


# ---------------------------------------------------------------------------
# File hashes and argument-derived files
# ---------------------------------------------------------------------------


class TestFileHashes:

    def test_inputs_and_outputs_are_hashed(self, tmp_path, monkeypatch):
        import hashlib

        _project(tmp_path, monkeypatch)
        (tmp_path / "in.txt").write_text("alpha\n")
        main(
            [
                "--code",
                "copy",
                "--",
                "cp",
                "in.txt",
                "out.txt",
            ]
        )
        record = json.loads(
            next((tmp_path / "trace_archive").glob("*.json")).read_text()
        )
        digest = hashlib.sha256(b"alpha\n").hexdigest()
        assert record["execution"]["file_hashes"] == {
            "in.txt": digest,
            "out.txt": digest,
        }

    def test_a_missing_output_has_no_hash(self, tmp_path, monkeypatch):
        """A spec role naming a file the run never wrote leaves no hash."""
        monkeypatch.chdir(tmp_path)
        run_and_record(
            "t",
            ["true"],
            tmp_path / "trace_archive",
            output_files=["never.txt"],
            log_trace=None,
        )
        record = json.loads(
            next((tmp_path / "trace_archive").glob("*.json")).read_text()
        )
        assert record["execution"]["file_hashes"] == {}


class TestArgumentDerivedFiles:
    """With no spec roles, an argument that is a file is an input, and one
    that exists only after the run is an output."""

    def test_a_spec_with_no_roles_falls_back_to_the_arguments(
        self, tmp_path, monkeypatch
    ):
        from dsagt.registry import CodeRegistry

        project = tmp_path / "proj"
        (project / "trace_archive").mkdir(parents=True)
        (project / ".dsagt").mkdir()
        (project / ".dsagt" / "config.yaml").write_text("project: t\n")
        CodeRegistry(runtime_dir=project).save_tool(
            {
                "name": "aidrin",
                "description": "d",
                "executable": "cat",
                "parameters": {
                    "args": {"type": "string", "required": True, "cli": "positional"}
                },
            }
        )
        (project / "data").mkdir()
        (project / "data" / "t.csv").write_text("a,b\n1,2\n")
        monkeypatch.chdir(project)
        main(
            [
                "--code",
                "aidrin",
                "--",
                "cat",
                "data/t.csv",
            ]
        )
        record = json.loads(
            next((project / "trace_archive").glob("*.json")).read_text()
        )
        assert record["execution"]["input_files"] == ["data/t.csv"]
        assert record["execution"]["output_files"] == []
        assert set(record["execution"]["file_hashes"]) == {"data/t.csv"}


class TestRolesAndArgumentsPerSide:

    def test_arguments_fill_the_side_the_roles_leave_empty(self, tmp_path, monkeypatch):
        """fastp's spec names -i and -o; the agent ran --in1/--out1. The role
        match gives nothing on either side, and the scan fills both."""
        _project(tmp_path, monkeypatch)
        (tmp_path / "in.fq").write_text("@r\nA\n+\nF\n")
        main(
            [
                "--code",
                "fastp",
                "--",
                "cp",
                "in.fq",
                "out.fq",
            ]
        )
        record = json.loads(
            next((tmp_path / "trace_archive").glob("*.json")).read_text()
        )
        assert record["execution"]["input_files"] == ["in.fq"]
        assert record["execution"]["output_files"] == ["out.fq"]

    def test_role_inputs_keep_and_outputs_come_from_the_scan(
        self, tmp_path, monkeypatch
    ):
        """The spec names the input; the output side, which the roles leave
        empty, is filled from the arguments."""
        from dsagt.registry import CodeRegistry

        project = tmp_path / "proj"
        (project / "trace_archive").mkdir(parents=True)
        (project / ".dsagt").mkdir()
        (project / ".dsagt" / "config.yaml").write_text("project: t\n")
        CodeRegistry(runtime_dir=project).save_tool(
            {
                "name": "t",
                "description": "d",
                "executable": "cp",
                "parameters": {
                    "src": {
                        "type": "string",
                        "description": "in",
                        "cli": "positional",
                        "role": "input",
                    },
                    "dest": {
                        "type": "string",
                        "description": "out",
                        "cli": "positional",
                    },
                },
            }
        )
        (project / "in.fq").write_text("x\n")
        monkeypatch.chdir(project)
        main(
            [
                "--code",
                "t",
                "--",
                "cp",
                "in.fq",
                "out.fq",
            ]
        )
        record = json.loads(
            next((project / "trace_archive").glob("*.json")).read_text()
        )
        assert record["execution"]["input_files"] == ["in.fq"]
        assert record["execution"]["output_files"] == ["out.fq"]


class TestNestedRuns:

    def test_a_nested_run_carries_its_parent_and_the_reconstruction_skips_it(
        self, tmp_path, monkeypatch
    ):
        import os
        import sys

        from dsagt.provenance import load_pipeline_records

        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
        _project(tmp_path, monkeypatch)
        records = tmp_path / "trace_archive"
        inner = (
            f"{sys.executable} -m dsagt.commands.run_code --code inner -- echo inner"
        )
        script = tmp_path / "loop.sh"
        script.write_text(f"#!/bin/bash\n{inner}\n")
        monkeypatch.delenv("DSAGT_RUN_PARENT", raising=False)
        rc = main(["--code", "loop", "--", "bash", str(script)])
        assert rc == 0
        by_name = {
            r.get("code_name"): r
            for r in (json.loads(p.read_text()) for p in records.glob("*.json"))
        }
        outer, child = by_name["loop"], by_name["inner"]
        assert "parent_record_id" not in outer
        assert child["parent_record_id"] == outer["record_id"]
        assert [r["code_name"] for r in load_pipeline_records(records)] == ["loop"]
        assert "DSAGT_RUN_PARENT" not in os.environ


class TestRolesTolerateTheUvWrapper:

    def test_a_command_without_the_wrapper_still_matches_the_spec(self):
        from dsagt.provenance import file_roles_from_command

        spec = {
            "executable": "dsagt-run --code c -- uv run --with h5py -- python skills/c/scripts/c.py",
            "parameters": {
                "case": {"cli": "--case", "role": "input"},
                "out": {"cli": "--out", "role": "output"},
            },
        }
        with_wrapper = [
            "uv",
            "run",
            "--with",
            "h5py",
            "--",
            "python",
            "skills/c/scripts/c.py",
            "--case",
            "d",
            "--out",
            "o.json",
        ]
        without = ["python", "skills/c/scripts/c.py", "--case", "d", "--out", "o.json"]
        assert file_roles_from_command(spec, with_wrapper) == (["d"], ["o.json"])
        assert file_roles_from_command(spec, without) == (["d"], ["o.json"])


class TestArgumentScanDetails:

    def test_a_directory_is_an_input_and_the_interpreted_script_is_not(
        self, tmp_path, monkeypatch
    ):
        from dsagt.provenance import files_from_arguments

        monkeypatch.chdir(tmp_path)
        (tmp_path / "case").mkdir()
        (tmp_path / "tool.py").write_text("print(1)\n")
        (tmp_path / "t.csv").write_text("a\n")
        assert files_from_arguments(
            ["python", "tool.py", "case", "t.csv", "--n", "3"]
        ) == [
            "case",
            "t.csv",
        ]
        assert files_from_arguments(["mytool", "tool.py", "case"]) == [
            "tool.py",
            "case",
        ]


class TestFindingsFromThe0919Runs:

    def test_a_rewritten_output_is_an_output_on_every_run(self, tmp_path, monkeypatch):
        """With no roles, the second run of a converter found its output
        already present, listed it as an input, and named no output."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.txt").write_text("a")
        write = "import sys, time; open(sys.argv[2], 'w').write(str(time.time()))"
        records = tmp_path / "trace_archive"
        for _ in range(2):
            run_and_record(
                "",
                [sys.executable, "-c", write, "in.txt", "out.txt"],
                records,
                log_trace=None,
            )
        runs = [json.loads(p.read_text())["execution"] for p in records.glob("*.json")]
        second = max(runs, key=lambda e: e["timestamp_start"])
        assert "out.txt" in second["output_files"]
        assert (
            second["file_hashes"]["out.txt"]
            == __import__("hashlib")
            .sha256((tmp_path / "out.txt").read_bytes())
            .hexdigest()
        )

    def test_a_failed_run_names_no_output_it_did_not_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        records = tmp_path / "trace_archive"
        rc = run_and_record(
            "plot", ["false"], records, output_files=["plots/p.png"], log_trace=None
        )
        assert rc != 0
        [path] = records.glob("*.json")
        assert json.loads(path.read_text())["execution"]["output_files"] == []

    def test_the_children_of_a_killed_parent_stay_in_the_reconstruction(self, tmp_path):
        from dsagt.provenance import load_pipeline_records

        def record(record_id, rc, parent=None):
            raw = {
                "record_id": record_id,
                "code_name": "",
                "execution": {
                    "exact_command": ["true"],
                    "return_code": rc,
                    "timestamp_start": record_id,
                },
            }
            if parent:
                raw["parent_record_id"] = parent
            (tmp_path / f"{record_id}.json").write_text(json.dumps(raw))

        record("a-loop-killed", -15)
        record("b-child", 0, parent="a-loop-killed")
        record("c-loop-done", 0)
        record("d-child", 0, parent="c-loop-done")
        kept = [r["record_id"] for r in load_pipeline_records(tmp_path)]
        assert kept == ["a-loop-killed", "b-child", "c-loop-done"]


def test_a_moved_input_is_not_an_output(tmp_path, monkeypatch):
    """aidrin, codex: `mv a b` recorded a as an output with no hash."""
    _project(tmp_path, monkeypatch)
    (tmp_path / "a.csv").write_text("x\n")
    records = tmp_path / "trace_archive"
    run_and_record("move", ["mv", "a.csv", "b.csv"], records, log_trace=None)
    [path] = records.glob("*.json")
    execution = json.loads(path.read_text())["execution"]
    assert execution["input_files"] == ["a.csv"]
    assert execution["output_files"] == ["b.csv"]
    assert set(execution["file_hashes"]) == {"a.csv", "b.csv"}


def test_a_run_without_a_code_name_is_refused(tmp_path, capsys):
    """--code is what names the record; argparse requires it."""
    with pytest.raises(SystemExit):
        main(["--records-dir", str(tmp_path), "--", "true"])
    assert "--code" in capsys.readouterr().err
    assert list(tmp_path.glob("*.json")) == []
