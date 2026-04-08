"""
Tests for agent auditing metrics.
"""

import json
from pathlib import Path

import pytest

from dsagt.auditing import (
    audit_session,
    detect_retries,
    execution_timing,
    load_all_records,
    session_summary,
    tool_frequency,
    tool_success_rates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_record(trace_dir: Path, record: dict) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    rid = record.get("record_id", "r0")
    tool = record.get("tool_name", "tool")
    path = trace_dir / f"{tool}_{rid}.json"
    path.write_text(json.dumps(record))


def _proxy_record(tool_name: str, session_id: str = "s1", record_id: str = "r0") -> dict:
    return {
        "record_id": record_id,
        "tool_name": tool_name,
        "session_id": session_id,
        "intent": {
            "command": tool_name,
            "parameters": {},
            "timestamp_requested": "2024-01-15T10:00:00+00:00",
        },
        "execution": None,
        "report": {"agent_output": "done"},
    }


def _wrapper_record(
    tool_name: str,
    return_code: int = 0,
    session_id: str = "s1",
    record_id: str = "r0",
    timestamp_start: str = "2024-01-15T10:00:00+00:00",
    timestamp_end: str = "2024-01-15T10:01:00+00:00",
) -> dict:
    return {
        "record_id": record_id,
        "tool_name": tool_name,
        "session_id": session_id,
        "execution": {
            "exact_command": [tool_name],
            "return_code": return_code,
            "stdout": "",
            "stderr": "",
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "input_files": [],
            "output_files": [],
        },
    }


# ---------------------------------------------------------------------------
# load_all_records
# ---------------------------------------------------------------------------

class TestLoadAllRecords:

    def test_loads_proxy_and_wrapper(self, tmp_path):
        _write_record(tmp_path, _proxy_record("fastp", record_id="r1"))
        _write_record(tmp_path, _wrapper_record("fastp", record_id="r2"))

        records = load_all_records(tmp_path)
        assert len(records) == 2

    def test_filters_by_session(self, tmp_path):
        _write_record(tmp_path, _wrapper_record("a", session_id="s1", record_id="r1"))
        _write_record(tmp_path, _wrapper_record("b", session_id="s2", record_id="r2"))

        records = load_all_records(tmp_path, session_id="s1")
        assert len(records) == 1

    def test_empty(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert load_all_records(tmp_path) == []


# ---------------------------------------------------------------------------
# tool_frequency
# ---------------------------------------------------------------------------

class TestToolFrequency:

    def test_counts_tools(self):
        records = [
            _wrapper_record("fastp", record_id="r1"),
            _wrapper_record("fastp", record_id="r2"),
            _wrapper_record("megahit", record_id="r3"),
        ]
        freq = tool_frequency(records)
        assert freq["fastp"] == 2
        assert freq["megahit"] == 1

    def test_empty(self):
        assert tool_frequency([]) == {}


# ---------------------------------------------------------------------------
# tool_success_rates
# ---------------------------------------------------------------------------

class TestToolSuccessRates:

    def test_counts_success_and_failure(self):
        records = [
            _wrapper_record("fastp", return_code=0, record_id="r1"),
            _wrapper_record("fastp", return_code=1, record_id="r2"),
            _wrapper_record("fastp", return_code=0, record_id="r3"),
        ]
        rates = tool_success_rates(records)
        assert rates["fastp"]["success"] == 2
        assert rates["fastp"]["failure"] == 1
        assert rates["fastp"]["total"] == 3

    def test_skips_proxy_records(self):
        records = [_proxy_record("fastp")]
        rates = tool_success_rates(records)
        assert rates == {}

    def test_multiple_tools(self):
        records = [
            _wrapper_record("fastp", return_code=0, record_id="r1"),
            _wrapper_record("megahit", return_code=1, record_id="r2"),
        ]
        rates = tool_success_rates(records)
        assert rates["fastp"]["success"] == 1
        assert rates["megahit"]["failure"] == 1


# ---------------------------------------------------------------------------
# detect_retries
# ---------------------------------------------------------------------------

class TestDetectRetries:

    def test_detects_repeated_tool(self):
        records = [
            _wrapper_record("fastp", record_id="r1"),
            _wrapper_record("fastp", record_id="r2"),
            _wrapper_record("megahit", record_id="r3"),
        ]
        retries = detect_retries(records)
        assert len(retries) == 1
        assert retries[0]["tool_name"] == "fastp"
        assert retries[0]["count"] == 2

    def test_no_retries(self):
        records = [
            _wrapper_record("fastp", record_id="r1"),
            _wrapper_record("megahit", record_id="r2"),
        ]
        assert detect_retries(records) == []


# ---------------------------------------------------------------------------
# execution_timing
# ---------------------------------------------------------------------------

class TestExecutionTiming:

    def test_computes_duration(self):
        records = [_wrapper_record(
            "fastp",
            timestamp_start="2024-01-15T10:00:00+00:00",
            timestamp_end="2024-01-15T10:00:30+00:00",
        )]
        timing = execution_timing(records)
        assert timing["fastp"]["min"] == 30.0
        assert timing["fastp"]["max"] == 30.0
        assert timing["fastp"]["count"] == 1

    def test_multiple_runs(self):
        records = [
            _wrapper_record("fastp", record_id="r1",
                            timestamp_start="2024-01-15T10:00:00+00:00",
                            timestamp_end="2024-01-15T10:00:10+00:00"),
            _wrapper_record("fastp", record_id="r2",
                            timestamp_start="2024-01-15T10:01:00+00:00",
                            timestamp_end="2024-01-15T10:01:30+00:00"),
        ]
        timing = execution_timing(records)
        assert timing["fastp"]["min"] == 10.0
        assert timing["fastp"]["max"] == 30.0
        assert timing["fastp"]["mean"] == 20.0
        assert timing["fastp"]["count"] == 2

    def test_skips_proxy_records(self):
        records = [_proxy_record("fastp")]
        assert execution_timing(records) == {}


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------

class TestSessionSummary:

    def test_basic_summary(self):
        records = [
            _wrapper_record("fastp", return_code=0, record_id="r1"),
            _wrapper_record("megahit", return_code=0, record_id="r2"),
            _wrapper_record("quast", return_code=1, record_id="r3"),
        ]
        summary = session_summary(records)
        assert summary["total_calls"] == 3
        assert summary["unique_tools"] == 3
        assert sorted(summary["tools_used"]) == ["fastp", "megahit", "quast"]
        assert summary["success_rate"] == 0.67

    def test_empty(self):
        summary = session_summary([])
        assert summary["total_calls"] == 0
        assert summary["success_rate"] is None

    def test_duration(self):
        records = [
            _wrapper_record("a", record_id="r1",
                            timestamp_start="2024-01-15T10:00:00+00:00",
                            timestamp_end="2024-01-15T10:01:00+00:00"),
            _wrapper_record("b", record_id="r2",
                            timestamp_start="2024-01-15T10:05:00+00:00",
                            timestamp_end="2024-01-15T10:06:00+00:00"),
        ]
        summary = session_summary(records)
        assert summary["duration_seconds"] == 360.0  # 6 minutes


# ---------------------------------------------------------------------------
# audit_session (end-to-end)
# ---------------------------------------------------------------------------

class TestAuditSession:

    def test_full_audit(self, tmp_path):
        _write_record(tmp_path, _wrapper_record("fastp", return_code=0, record_id="r1"))
        _write_record(tmp_path, _wrapper_record("fastp", return_code=1, record_id="r2"))
        _write_record(tmp_path, _wrapper_record("megahit", return_code=0, record_id="r3"))

        report = audit_session(tmp_path, session_id="s1")

        assert report["session_id"] == "s1"
        assert report["summary"]["total_calls"] == 3
        assert report["tool_frequency"]["fastp"] == 2
        assert report["success_rates"]["fastp"]["failure"] == 1
        assert len(report["retries"]) == 1
        assert report["retries"][0]["tool_name"] == "fastp"
        assert "fastp" in report["timing"]

    def test_empty_session(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        report = audit_session(tmp_path, session_id="s1")
        assert report["summary"]["total_calls"] == 0
