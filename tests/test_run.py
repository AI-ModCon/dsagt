"""
Tests for dsagt-run execution wrapper.

Covers argument parsing, command execution, record writing,
exit code propagation, error handling, and env var fallbacks.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dsagt.provenance import (
    _inject_cache_breakpoints,
    _parse_file_list,
    _resolve_records_dir,
    _write_record,
    run_and_record,
)
from dsagt.commands.run_tool import (
    _parse_args,
    main,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:

    def test_basic(self):
        args, command = _parse_args(["--tool", "fastp", "--", "fastp", "-q", "20"])
        assert args.tool == "fastp"
        assert command == ["fastp", "-q", "20"]

    def test_all_flags(self):
        args, command = _parse_args([
            "--tool", "megahit",
            "--session", "sess-1",
            "--record-id", "rec-42",
            "--records-dir", "/tmp/records",
            "--input-files", "a.fq,b.fq",
            "--output-files", "out/contigs.fa",
            "--", "megahit", "-1", "a.fq",
        ])
        assert args.tool == "megahit"
        assert args.session == "sess-1"
        assert args.record_id == "rec-42"
        assert args.records_dir == "/tmp/records"
        assert args.input_files == "a.fq,b.fq"
        assert args.output_files == "out/contigs.fa"
        assert command == ["megahit", "-1", "a.fq"]

    def test_no_separator_exits(self):
        """Missing '--' separator causes a SystemExit (from argparse --help)."""
        with pytest.raises(SystemExit):
            _parse_args(["--tool", "fastp", "fastp", "-q", "20"])

    def test_defaults(self):
        args, _ = _parse_args(["--tool", "x", "--", "echo"])
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

    def test_env_var(self):
        with patch.dict(os.environ, {"DSAGT_RECORDS_DIR": "/from/env"}):
            assert _resolve_records_dir(None) == Path("/from/env")

    def test_no_env_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DSAGT_RECORDS_DIR", None)
            os.environ.pop("DSAGT_PROJECT_DIR", None)
            with pytest.raises(ValueError, match="No records directory"):
                _resolve_records_dir(None)


# ---------------------------------------------------------------------------
# Record writing
# ---------------------------------------------------------------------------

class TestWriteRecord:

    def test_creates_directory_and_file(self, tmp_path):
        records_dir = tmp_path / "nested" / "records"
        record = {
            "record_id": "abc123",
            "tool_name": "fastp",
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
        assert data["tool_name"] == "fastp"
        assert data["execution"]["return_code"] == 0

    def test_record_is_valid_json(self, tmp_path):
        record = {
            "record_id": "x",
            "tool_name": "t",
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
            tool_name="echo_test",
            command=["echo", "hello world"],
            records_dir=tmp_path,
            record_id="test-001",
        )

        assert exit_code == 0

        records = list(tmp_path.glob("*.json"))
        assert len(records) == 1

        data = json.loads(records[0].read_text())
        assert data["tool_name"] == "echo_test"
        assert data["record_id"] == "test-001"
        assert data["execution"]["return_code"] == 0
        assert "hello world" in data["execution"]["stdout"]
        assert data["execution"]["exact_command"] == ["echo", "hello world"]

    def test_failing_command(self, tmp_path):
        """A command that fails returns non-zero and captures stderr."""
        exit_code = run_and_record(
            tool_name="false_test",
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
            tool_name="missing",
            command=["this_command_does_not_exist_xyz"],
            records_dir=tmp_path,
            record_id="test-003",
        )

        assert exit_code == 127

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["return_code"] == 127
        assert "command not found" in data["execution"]["stderr"]

    def test_session_from_env(self, tmp_path):
        """Session ID falls back to DSAGT_SESSION_ID env var."""
        with patch.dict(os.environ, {"DSAGT_SESSION_ID": "env-session"}):
            run_and_record(
                tool_name="t",
                command=["echo"],
                records_dir=tmp_path,
                record_id="test-004",
            )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["session_id"] == "env-session"

    def test_explicit_session_overrides_env(self, tmp_path):
        """Explicit --session takes precedence over env var."""
        with patch.dict(os.environ, {"DSAGT_SESSION_ID": "env-session"}):
            run_and_record(
                tool_name="t",
                command=["echo"],
                records_dir=tmp_path,
                session_id="explicit-session",
                record_id="test-005",
            )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["session_id"] == "explicit-session"

    def test_file_lists_recorded(self, tmp_path):
        """Input and output file lists appear in the record."""
        run_and_record(
            tool_name="t",
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
            tool_name="t",
            command=["echo"],
            records_dir=tmp_path,
            record_id="test-007",
        )

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["execution"]["timestamp_start"]
        assert data["execution"]["timestamp_end"]
        assert data["execution"]["timestamp_start"] <= data["execution"]["timestamp_end"]

    def test_auto_generates_record_id(self, tmp_path):
        """Omitting record_id auto-generates one."""
        run_and_record(
            tool_name="t",
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
        real backend.  dsagt-run is only ever invoked inside a `dsagt start`
        context in production, where MLFLOW_TRACKING_URI is always set; the
        tests need to mirror that so init_tracing doesn't (correctly) refuse
        to silently no-op."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
        monkeypatch.setenv("DSAGT_PROJECT", "test")

    def test_basic_invocation(self, tmp_path):
        """main() runs a command and returns its exit code."""
        exit_code = main([
            "--tool", "echo_tool",
            "--records-dir", str(tmp_path),
            "--", "echo", "from main",
        ])

        assert exit_code == 0
        records = list(tmp_path.glob("*.json"))
        assert len(records) == 1

    def test_empty_command_returns_1(self, tmp_path):
        """No command after '--' returns exit code 1."""
        exit_code = main([
            "--tool", "empty",
            "--records-dir", str(tmp_path),
            "--",
        ])
        assert exit_code == 1

    def test_exit_code_propagation(self, tmp_path):
        """main() returns the wrapped command's exit code."""
        exit_code = main([
            "--tool", "fail",
            "--records-dir", str(tmp_path),
            "--", "bash", "-c", "exit 7",
        ])
        assert exit_code == 7


# ---------------------------------------------------------------------------
# Prompt cache breakpoint injection
# ---------------------------------------------------------------------------

class TestInjectCacheBreakpoints:
    """Cache markers must land on the LAST tool and on the system message."""

    _MARK = {"type": "ephemeral"}

    def test_string_system_promoted_to_blocks_with_marker(self):
        messages = [{"role": "system", "content": "you are an agent"}]
        kwargs = {}
        _inject_cache_breakpoints(messages, kwargs)
        assert messages[0]["content"] == [{
            "type": "text",
            "text": "you are an agent",
            "cache_control": self._MARK,
        }]

    def test_block_system_marker_on_last_block(self):
        messages = [{
            "role": "system",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "last"},
            ],
        }]
        _inject_cache_breakpoints(messages, {})
        assert "cache_control" not in messages[0]["content"][0]
        assert messages[0]["content"][1]["cache_control"] == self._MARK

    def test_marker_on_last_tool_only(self):
        kwargs = {
            "tools": [
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}},
                {"type": "function", "function": {"name": "c"}},
            ],
        }
        _inject_cache_breakpoints([], kwargs)
        assert "cache_control" not in kwargs["tools"][0]
        assert "cache_control" not in kwargs["tools"][1]
        assert kwargs["tools"][2]["cache_control"] == self._MARK

    def test_no_tools_no_crash(self):
        messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]
        _inject_cache_breakpoints(messages, {})  # no tools key
        # System still got marked
        assert messages[0]["content"][0]["cache_control"] == self._MARK

    def test_no_system_no_crash(self):
        messages = [{"role": "user", "content": "hi"}]
        kwargs = {"tools": [{"type": "function", "function": {"name": "a"}}]}
        _inject_cache_breakpoints(messages, kwargs)
        assert kwargs["tools"][0]["cache_control"] == self._MARK
        # User message untouched
        assert messages[0] == {"role": "user", "content": "hi"}

    def test_only_first_system_message_marked(self):
        messages = [
            {"role": "system", "content": "primary"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "injected later"},
        ]
        _inject_cache_breakpoints(messages, {})
        # First system promoted with marker
        assert messages[0]["content"][0]["cache_control"] == self._MARK
        # Second one untouched (no cache_control)
        assert messages[2]["content"] == "injected later"
