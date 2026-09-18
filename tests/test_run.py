"""
Tests for dsagt-run execution wrapper.

Covers argument parsing, command execution, record writing,
exit code propagation, error handling, and env var fallbacks.
"""

import json
from pathlib import Path

import pytest

from dsagt.provenance import (
    _parse_file_list,
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


class TestParseArgs:

    def test_basic(self):
        args, command = _parse_args(["--code", "fastp", "--", "fastp", "-q", "20"])
        assert args.code == "fastp"
        assert command == ["fastp", "-q", "20"]

    def test_all_flags(self):
        args, command = _parse_args(
            [
                "--code",
                "megahit",
                "--session",
                "sess-1",
                "--record-id",
                "rec-42",
                "--records-dir",
                "/tmp/records",
                "--input-files",
                "a.fq,b.fq",
                "--output-files",
                "out/contigs.fa",
                "--",
                "megahit",
                "-1",
                "a.fq",
            ]
        )
        assert args.code == "megahit"
        assert args.session == "sess-1"
        assert args.record_id == "rec-42"
        assert args.records_dir == "/tmp/records"
        assert args.input_files == "a.fq,b.fq"
        assert args.output_files == "out/contigs.fa"
        assert command == ["megahit", "-1", "a.fq"]

    def test_no_separator_exits(self):
        """Missing '--' separator causes a SystemExit (from argparse --help)."""
        with pytest.raises(SystemExit):
            _parse_args(["--code", "fastp", "fastp", "-q", "20"])

    def test_defaults(self):
        args, _ = _parse_args(["--code", "x", "--", "echo"])
        assert args.session is None
        assert args.record_id is None
        assert args.records_dir is None
        assert args.input_files is None
        assert args.output_files is None


# ---------------------------------------------------------------------------
# File list parsing
# ---------------------------------------------------------------------------


class TestParseFileList:

    def test_none(self):
        assert _parse_file_list(None) == []

    def test_empty_string(self):
        assert _parse_file_list("") == []

    def test_single(self):
        assert _parse_file_list("reads.fq.gz") == ["reads.fq.gz"]

    def test_multiple(self):
        assert _parse_file_list("a.fq, b.fq,c.fq") == ["a.fq", "b.fq", "c.fq"]

    def test_trailing_comma(self):
        assert _parse_file_list("a.fq,") == ["a.fq"]


# ---------------------------------------------------------------------------
# Records directory resolution
# ---------------------------------------------------------------------------


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

    def test_explicit_wins(self):
        assert _resolve_records_dir("/custom/dir") == Path("/custom/dir")

    def test_uses_cwd_dsagt_config(self, tmp_path, monkeypatch):
        """No --records-dir and no DSAGT_PROJECT_DIR → the cwd is the
        project: reads ``<cwd>/.dsagt/config.yaml`` and uses
        ``<cwd>/trace_archive``."""
        monkeypatch.delenv("DSAGT_PROJECT_DIR", raising=False)
        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
        monkeypatch.chdir(tmp_path)
        assert _resolve_records_dir(None) == tmp_path / "trace_archive"

    def test_project_dir_env_wins_over_cwd(self, tmp_path, monkeypatch):
        """``DSAGT_PROJECT_DIR`` (exported by ``dsagt start``) names the
        project even when the command runs from a subdirectory."""
        project = tmp_path / "proj"
        (project / ".dsagt").mkdir(parents=True)
        (project / ".dsagt" / "config.yaml").write_text("project: t\n")
        (project / "data").mkdir()
        monkeypatch.setenv("DSAGT_PROJECT_DIR", str(project))
        monkeypatch.chdir(project / "data")
        assert _resolve_records_dir(None) == project / "trace_archive"

    def test_no_config_in_cwd_raises(self, tmp_path, monkeypatch):
        """A cwd without .dsagt/config.yaml fails with one line naming the
        rule; there is no walk up the tree."""
        monkeypatch.delenv("DSAGT_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not a dsagt project"):
            _resolve_records_dir(None)


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
        when not passed: the MCP server mints it there at startup and
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
    def _mlflow_file_store(self, tmp_path, monkeypatch):
        """Point MLflow tracing at a scratch file-store so init_tracing has a
        real backend.  In production dsagt-run runs with cwd inside the
        project directory, where ``.dsagt/config.yaml`` (project name) is
        and the session id comes from ``.dsagt/state.yaml``;
        tests mirror that by chdir-ing into tmp_path.
        """
        # Serverless: init_tracing resolves a sqlite store from the project
        # dir via MLflow's native provider.  Stub the resolver to a known
        # sqlite URI so a shell-set MLFLOW_TRACKING_URI cannot redirect the
        # test.
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
        ``code.execute`` root must carry it, or every execution trace falls
        into an unbucketed ``(no-session)`` group in ``dsagt info``."""
        import mlflow

        from dsagt import observability as obs_module

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: test\n")
        (tmp_path / ".dsagt" / "state.yaml").write_text(
            "sessions:\n- id: 7\n  started_at: '2026-01-01T00:00:00Z'\n"
        )
        monkeypatch.setattr(obs_module, "_initialized", False)
        monkeypatch.setattr(obs_module, "_default_session_id", None)

        assert main(["--code", "t", "--records-dir", str(tmp_path), "--", "true"]) == 0

        assert obs_module._default_session_id == "test-7"
        trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
        assert trace.info.trace_metadata.get("mlflow.trace.session") == "test-7"

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
                "--records-dir",
                str(tmp_path),
                "--",
                "echo",
                "from main",
            ]
        )

        assert exit_code == 0
        records = list(tmp_path.glob("*.json"))
        assert len(records) == 1

    def test_empty_command_returns_1(self, tmp_path):
        """No command after '--' returns exit code 1."""
        exit_code = main(
            [
                "--code",
                "empty",
                "--records-dir",
                str(tmp_path),
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
                "--records-dir",
                str(tmp_path),
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
# Ad-hoc runs, --stdout, and a run ended by a signal
# ---------------------------------------------------------------------------


class TestAdHocRun:

    def test_code_is_optional(self):
        args, command = _parse_args(["--", "python", "x.py"])
        assert args.code is None
        assert command == ["python", "x.py"]

    def test_record_without_a_code(self, tmp_path):
        """A run with no --code is recorded with an empty code name."""
        exit_code = main(["--records-dir", str(tmp_path), "--", "echo", "adhoc"])
        assert exit_code == 0
        records = list(tmp_path.glob("*.json"))
        assert len(records) == 1
        assert records[0].name.startswith("adhoc_")
        record = json.loads(records[0].read_text())
        assert record["code_name"] == ""
        assert record["execution"]["exact_command"] == ["echo", "adhoc"]
        assert record["execution"]["stdout"] == "adhoc\n"


class TestStdoutFile:

    def test_stdout_is_written_to_the_file_and_recorded_as_an_output(
        self, tmp_path, capsys
    ):
        target = tmp_path / "audit" / "report.json"
        exit_code = main(
            [
                "--code",
                "echo_tool",
                "--records-dir",
                str(tmp_path / "records"),
                "--stdout",
                str(target),
                "--",
                "echo",
                '{"ok": true}',
            ]
        )
        assert exit_code == 0
        assert target.read_text() == '{"ok": true}\n'
        record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
        assert record["execution"]["output_files"] == [str(target)]
        assert record["execution"]["stdout"] == '{"ok": true}\n'
        # The file holds the output; the terminal gets one line saying so.
        out = capsys.readouterr().out
        assert '{"ok": true}' not in out
        assert str(target) in out

    def test_stdout_file_joins_the_role_outputs(self, tmp_path):
        target = tmp_path / "report.txt"
        main(
            [
                "--code",
                "t",
                "--records-dir",
                str(tmp_path / "records"),
                "--output-files",
                "data/out.csv",
                "--stdout",
                str(target),
                "--",
                "echo",
                "x",
            ]
        )
        record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
        assert record["execution"]["output_files"] == ["data/out.csv", str(target)]


class TestSignal:

    def test_a_terminated_run_still_writes_its_record(self, tmp_path):
        """SIGTERM to dsagt-run reaches the child and the record says so."""
        import os
        import signal
        import subprocess
        import sys
        import time

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dsagt.commands.run_code",
                "--code",
                "sleeper",
                "--records-dir",
                str(tmp_path),
                "--",
                "sleep",
                "30",
            ],
            cwd=str(tmp_path),
        )
        time.sleep(1.5)
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
        records = list(tmp_path.glob("*.json"))
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

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.txt").write_text("alpha\n")
        main(
            [
                "--code",
                "copy",
                "--records-dir",
                str(tmp_path / "records"),
                "--input-files",
                "in.txt",
                "--output-files",
                "out.txt",
                "--",
                "cp",
                "in.txt",
                "out.txt",
            ]
        )
        record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
        digest = hashlib.sha256(b"alpha\n").hexdigest()
        assert record["execution"]["file_hashes"] == {
            "in.txt": digest,
            "out.txt": digest,
        }

    def test_a_missing_output_has_no_hash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--code",
                "t",
                "--records-dir",
                str(tmp_path / "records"),
                "--output-files",
                "never.txt",
                "--",
                "true",
            ]
        )
        record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
        assert record["execution"]["file_hashes"] == {}


class TestArgumentDerivedFiles:
    """With no spec roles, an argument that is a file is an input, and one
    that exists only after the run is an output."""

    def test_ad_hoc_run_names_its_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.csv").write_text("x\n")
        main(
            [
                "--records-dir",
                str(tmp_path / "records"),
                "--",
                "cp",
                "a.csv",
                "b.csv",
            ]
        )
        record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
        assert record["execution"]["input_files"] == ["a.csv"]
        assert record["execution"]["output_files"] == ["b.csv"]

    def test_a_spec_with_no_roles_falls_back_to_the_arguments(
        self, tmp_path, monkeypatch
    ):
        from dsagt.registry import CodeRegistry

        project = tmp_path / "proj"
        (project / "trace_archive").mkdir(parents=True)
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
                "--records-dir",
                str(project / "trace_archive"),
                "--stdout",
                "audit/pre.json",
                "--",
                "cat",
                "data/t.csv",
            ]
        )
        record = json.loads(
            next((project / "trace_archive").glob("*.json")).read_text()
        )
        assert record["execution"]["input_files"] == ["data/t.csv"]
        assert record["execution"]["output_files"] == ["audit/pre.json"]
        assert set(record["execution"]["file_hashes"]) == {
            "data/t.csv",
            "audit/pre.json",
        }
