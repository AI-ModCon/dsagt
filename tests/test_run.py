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


class TestResolveRecordsDir:

    def test_explicit_wins(self):
        assert _resolve_records_dir("/custom/dir") == Path("/custom/dir")

    def test_uses_cwd_dsagt_config(self, tmp_path, monkeypatch):
        """No --records-dir → reads ``<cwd>/.dsagt/config.yaml`` and uses
        ``<cwd>/trace_archive``.  Env vars are not consulted; the project
        dir is the single source of truth."""
        (tmp_path / ".dsagt").mkdir()
        (tmp_path / ".dsagt" / "config.yaml").write_text("project: t\n")
        monkeypatch.chdir(tmp_path)
        assert _resolve_records_dir(None) == tmp_path / "trace_archive"

    def test_no_config_in_cwd_raises(self, tmp_path, monkeypatch):
        """If cwd has no .dsagt/config.yaml, fail clearly — don't walk
        up the tree, don't fall back to env vars."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No .dsagt/config.yaml"):
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
        project directory, where ``.dsagt/config.yaml`` (project name) lives
        and the session id comes from ``.dsagt/state.yaml``;
        tests mirror that by chdir-ing into tmp_path.
        """
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
