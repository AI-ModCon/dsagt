"""
Tests for tool execution record indexing.

Tests render_execution_text, execution_metadata, index_execution_record,
and index_trace_archive.  KnowledgeBase embedding is mocked.

Record formats match the actual output from:
  - proxy_callback.py (intent + report, execution=None)
  - run.py (execution only, no intent/report)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dsagt.tool_executions import (
    COLLECTION_NAME,
    execution_metadata,
    index_execution_record,
    index_trace_archive,
    render_execution_text,
)
from dsagt.knowledge import CollectionRoute, KnowledgeBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_embed(texts: list[str]) -> np.ndarray:
    dim = 8
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.RandomState(hash(t) & 0xFFFFFFFF)
        vecs[i] = rng.randn(dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


def make_proxy_record(
    tool="fastp",
    params=None,
    session="s1",
    record_id="toolu_001",
    agent_output="Filtering complete.",
):
    """Proxy callback record: intent + report, execution=None."""
    return {
        "record_id": record_id,
        "tool_name": tool,
        "session_id": session,
        "intent": {
            "command": tool,
            "parameters": params or {"q": 20, "in1": "sample.fq.gz"},
            "timestamp_requested": "2024-01-15T10:30:00+00:00",
            "session_id": session,
        },
        "execution": None,
        "report": {
            "agent_output": agent_output,
            "timestamp_reported": "2024-01-15T10:30:13+00:00",
            "wrapper_used": False,
        },
    }


def make_wrapper_record(
    tool="fastp",
    session="s1",
    record_id="rec_001",
    return_code=0,
    stdout="done\n",
    stderr="",
    input_files=None,
    output_files=None,
):
    """Wrapper (dsagt-run) record: execution only, no intent/report."""
    return {
        "record_id": record_id,
        "tool_name": tool,
        "session_id": session,
        "execution": {
            "exact_command": [tool, "-q", "20", "--in1", "sample.fq.gz"],
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp_start": "2024-01-15T10:30:01+00:00",
            "timestamp_end": "2024-01-15T10:30:12+00:00",
            "input_files": input_files or ["sample.fq.gz"],
            "output_files": output_files or ["sample_trimmed.fq.gz"],
        },
    }


# ---------------------------------------------------------------------------
# render_execution_text
# ---------------------------------------------------------------------------

class TestRenderExecutionText:

    def test_proxy_record_uses_intent(self):
        """Proxy record renders parameters from intent layer."""
        record = make_proxy_record()
        text = render_execution_text(record)
        assert "Tool: fastp" in text
        assert "Parameters:" in text
        assert "Agent report:" in text

    def test_wrapper_record_uses_exact_command(self):
        """Wrapper record renders exact command from execution layer."""
        record = make_wrapper_record()
        text = render_execution_text(record)
        assert "fastp -q 20 --in1 sample.fq.gz" in text
        assert "succeeded" in text
        assert "sample.fq.gz" in text
        assert "sample_trimmed.fq.gz" in text

    def test_failed_execution(self):
        """Failed execution renders exit code."""
        record = make_wrapper_record(return_code=1)
        text = render_execution_text(record)
        assert "failed (exit code 1)" in text

    def test_stderr_included(self):
        """Stderr content is included in the text."""
        record = make_wrapper_record(stderr="WARNING: low quality reads detected")
        text = render_execution_text(record)
        assert "low quality reads detected" in text

    def test_long_stderr_truncated(self):
        """Very long stderr is truncated."""
        record = make_wrapper_record(stderr="x" * 500)
        text = render_execution_text(record)
        assert "..." in text

    def test_long_agent_output_truncated(self):
        """Long agent output in proxy record is truncated."""
        record = make_proxy_record(agent_output="y" * 600)
        text = render_execution_text(record)
        assert "..." in text

    def test_exact_command_list_joined(self):
        """exact_command as a list is joined with spaces."""
        record = make_wrapper_record()
        text = render_execution_text(record)
        # Should be joined, not a Python list repr
        assert "[" not in text.split("Command:")[1].split("\n")[0]

    def test_timing_from_wrapper(self):
        """Wrapper record includes timing info."""
        record = make_wrapper_record()
        text = render_execution_text(record)
        assert "Duration:" in text

    def test_minimal_record(self):
        """Record with only tool_name renders."""
        text = render_execution_text({"tool_name": "ls"})
        assert "Tool: ls" in text

    def test_empty_record(self):
        """Record with nothing renders tool as unknown."""
        text = render_execution_text({})
        assert "unknown" in text


# ---------------------------------------------------------------------------
# execution_metadata
# ---------------------------------------------------------------------------

class TestExecutionMetadata:

    def test_proxy_record_metadata(self):
        """Proxy record extracts metadata from top-level and intent fields."""
        record = make_proxy_record()
        meta = execution_metadata(record)

        assert meta["tool_name"] == "fastp"
        assert meta["session_id"] == "s1"
        assert meta["wrapper_used"] == 0
        assert "return_code" not in meta
        assert meta["timestamp"] == "2024-01-15T10:30:00+00:00"
        assert meta["record_id"] == "toolu_001"

    def test_wrapper_record_metadata(self):
        """Wrapper record extracts metadata from top-level and execution fields."""
        record = make_wrapper_record()
        meta = execution_metadata(record)

        assert meta["tool_name"] == "fastp"
        assert meta["session_id"] == "s1"
        assert meta["return_code"] == 0
        assert meta["wrapper_used"] == 1
        assert meta["timestamp"] == "2024-01-15T10:30:01+00:00"
        assert meta["record_id"] == "rec_001"

    def test_failed_execution_metadata(self):
        """Failed execution has non-zero return_code."""
        record = make_wrapper_record(return_code=137)
        meta = execution_metadata(record)
        assert meta["return_code"] == 137

    def test_missing_session_defaults_to_unknown(self):
        """Missing session_id defaults to 'unknown'."""
        record = {"tool_name": "ls"}
        meta = execution_metadata(record)
        assert meta["session_id"] == "unknown"

    def test_top_level_fields_preferred(self):
        """Top-level tool_name/session_id are used over intent fields."""
        record = make_proxy_record(tool="fastp", session="s1")
        record["tool_name"] = "override_tool"
        record["session_id"] = "override_session"
        meta = execution_metadata(record)
        assert meta["tool_name"] == "override_tool"
        assert meta["session_id"] == "override_session"


# ---------------------------------------------------------------------------
# index_execution_record
# ---------------------------------------------------------------------------

class TestIndexExecutionRecord:

    def test_indexes_into_tool_executions_collection(self, tmp_path):
        """Single record is indexed into the tool_executions collection."""
        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            record = make_proxy_record()
            result = index_execution_record(record, kb)

            assert result["collection"] == COLLECTION_NAME
            assert result["entries_added"] == 1

            results = kb.search(
                "fastp quality", collection=COLLECTION_NAME,
                top_k=5, rerank=False,
            )
            assert len(results) > 0
            assert "fastp" in results[0]["chunk"]["text"]

            kb.close()

    def test_indexes_wrapper_record(self, tmp_path):
        """Wrapper-only record is also indexable."""
        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            record = make_wrapper_record()
            result = index_execution_record(record, kb)

            assert result["entries_added"] == 1
            kb.close()


# ---------------------------------------------------------------------------
# index_trace_archive
# ---------------------------------------------------------------------------

class TestIndexTraceArchive:

    def _write_records(self, trace_dir: Path, records: list[dict]):
        trace_dir.mkdir(parents=True, exist_ok=True)
        for i, record in enumerate(records):
            rid = record.get("record_id", f"rec_{i}")
            path = trace_dir / f"{record.get('tool_name', 'unknown')}_{rid}.json"
            path.write_text(json.dumps(record))

    def test_indexes_all_records(self, tmp_path):
        """Indexes all records in trace_dir."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(trace_dir, [
            make_proxy_record(record_id="t1"),
            make_wrapper_record(tool="megahit", record_id="t2"),
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(trace_dir, kb)

            assert result["indexed"] == 2
            assert result["skipped"] == 0
            assert result["total_files"] == 2
            kb.close()

    def test_skips_already_indexed(self, tmp_path):
        """Records with IDs in indexed_ids are skipped."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(trace_dir, [
            make_proxy_record(record_id="t1"),
            make_proxy_record(record_id="t2"),
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            indexed_ids = {"t1"}
            result = index_trace_archive(trace_dir, kb, indexed_ids=indexed_ids)

            assert result["indexed"] == 1
            assert result["skipped"] == 1
            assert "t2" in indexed_ids
            kb.close()

    def test_updates_indexed_ids(self, tmp_path):
        """indexed_ids set is updated with newly indexed record IDs."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(trace_dir, [
            make_proxy_record(record_id="t1"),
            make_wrapper_record(record_id="t2"),
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            indexed_ids: set[str] = set()
            index_trace_archive(trace_dir, kb, indexed_ids=indexed_ids)

            assert indexed_ids == {"t1", "t2"}
            kb.close()

    def test_empty_directory(self, tmp_path):
        """Empty trace_dir returns zeros."""
        trace_dir = tmp_path / "trace_archive"
        trace_dir.mkdir()

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(trace_dir, kb)

            assert result["indexed"] == 0
            assert result["total_files"] == 0
            kb.close()

    def test_nonexistent_directory(self, tmp_path):
        """Non-existent trace_dir returns zeros."""
        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(tmp_path / "missing", kb)

            assert result["indexed"] == 0
            kb.close()

    def test_skips_records_without_intent_or_execution(self, tmp_path):
        """Records missing both intent and execution are skipped."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(trace_dir, [
            {"record_id": "bad", "tool_name": "unknown", "report": {"agent_output": "something"}},
            make_proxy_record(record_id="good"),
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(trace_dir, kb)

            assert result["indexed"] == 1
            assert result["errors"] == 1
            kb.close()

    def test_accepts_wrapper_only_records(self, tmp_path):
        """Wrapper-only records (no intent) are valid and indexed."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(trace_dir, [
            make_wrapper_record(record_id="w1"),
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(trace_dir, kb)

            assert result["indexed"] == 1
            assert result["errors"] == 0
            kb.close()

    def test_idempotent_reindex(self, tmp_path):
        """Running index_trace_archive twice doesn't duplicate entries."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(trace_dir, [make_proxy_record(record_id="t1")])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            indexed_ids: set[str] = set()

            r1 = index_trace_archive(trace_dir, kb, indexed_ids=indexed_ids)
            assert r1["indexed"] == 1

            r2 = index_trace_archive(trace_dir, kb, indexed_ids=indexed_ids)
            assert r2["indexed"] == 0
            assert r2["skipped"] == 1
            kb.close()


# ---------------------------------------------------------------------------
# End-to-end: index + filtered search
# ---------------------------------------------------------------------------

class TestIndexAndSearch:

    def _index_records(self, tmp_path, records):
        """Helper: write records to disk and index them."""
        trace_dir = tmp_path / "trace_archive"
        trace_dir.mkdir(exist_ok=True)
        for r in records:
            rid = r.get("record_id", "x")
            path = trace_dir / f"{r.get('tool_name', 'unknown')}_{rid}.json"
            path.write_text(json.dumps(r))
        return trace_dir

    def test_search_by_tool_name(self, tmp_path):
        """Index multiple records, search filtered by tool_name."""
        records = [
            make_proxy_record(tool="fastp", session="s1", record_id="t1"),
            make_proxy_record(tool="megahit", session="s1", record_id="t2"),
            make_proxy_record(tool="fastp", session="s2", record_id="t3"),
        ]
        trace_dir = self._index_records(tmp_path, records)

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            index_trace_archive(trace_dir, kb)

            results = kb.search(
                "quality filtering parameters",
                collection=COLLECTION_NAME,
                top_k=10, rerank=False,
                where={"tool_name": "fastp"},
            )
            for r in results:
                assert r["chunk"]["metadata"]["tool_name"] == "fastp"
            kb.close()

    def test_search_by_session(self, tmp_path):
        """Index multiple records, search filtered by session_id."""
        records = [
            make_proxy_record(tool="fastp", session="s1", record_id="t1"),
            make_proxy_record(tool="fastp", session="s2", record_id="t2"),
        ]
        trace_dir = self._index_records(tmp_path, records)

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            index_trace_archive(trace_dir, kb)

            results = kb.search(
                "fastp",
                collection=COLLECTION_NAME,
                top_k=10, rerank=False,
                where={"session_id": "s1"},
            )
            for r in results:
                assert r["chunk"]["metadata"]["session_id"] == "s1"
            kb.close()

    def test_search_wrapper_records_by_return_code(self, tmp_path):
        """Wrapper records can be filtered by return_code."""
        records = [
            make_wrapper_record(tool="fastp", return_code=0, record_id="t1"),
            make_wrapper_record(tool="megahit", return_code=1, record_id="t2"),
        ]
        trace_dir = self._index_records(tmp_path, records)

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            index_trace_archive(trace_dir, kb)

            results = kb.search(
                "tool execution",
                collection=COLLECTION_NAME,
                top_k=10, rerank=False,
                where={"return_code": 1},
            )
            for r in results:
                assert r["chunk"]["metadata"]["return_code"] == 1
            kb.close()
