"""
Tests for tool execution record indexing.

Tests render_execution_text, execution_metadata, index_execution_record,
and index_trace_archive.  KnowledgeBase embedding is mocked.

Records match the output of ``dsagt-run`` (run.py): an ``execution`` block
with no intent/report.  (The pre-BYOA proxy_callback.py producer, which wrote
intent/report records, was removed — see scratch/excised_proxy_provenance.py.)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from dsagt.provenance import (
    CODE_USE_COLLECTION as COLLECTION_NAME,
    CodeUseIndexer,
    execution_metadata,
    index_trace_archive,
    render_execution_text,
)
from dsagt.knowledge import KnowledgeBase

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
        "code_name": tool,
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
        """Record with only code_name renders."""
        text = render_execution_text({"code_name": "ls"})
        assert "Code: ls" in text

    def test_empty_record(self):
        """Record with nothing renders tool as unknown."""
        text = render_execution_text({})
        assert "unknown" in text


# ---------------------------------------------------------------------------
# execution_metadata
# ---------------------------------------------------------------------------


class TestExecutionMetadata:

    def test_wrapper_record_metadata(self):
        """Wrapper record extracts metadata from top-level and execution fields."""
        record = make_wrapper_record()
        meta = execution_metadata(record)

        assert meta["code_name"] == "fastp"
        assert meta["session_id"] == "s1"
        assert meta["return_code"] == 0
        assert meta["timestamp"] == "2024-01-15T10:30:01+00:00"
        assert meta["record_id"] == "rec_001"

    def test_failed_execution_metadata(self):
        """Failed execution has non-zero return_code."""
        record = make_wrapper_record(return_code=137)
        meta = execution_metadata(record)
        assert meta["return_code"] == 137

    def test_missing_session_defaults_to_unknown(self):
        """Missing session_id defaults to 'unknown'."""
        record = {"code_name": "ls"}
        meta = execution_metadata(record)
        assert meta["session_id"] == "unknown"


# ---------------------------------------------------------------------------
# CodeUseIndexer — idempotent, incremental heartbeat indexing
# ---------------------------------------------------------------------------


class TestCodeUseIndexer:

    def _write(self, trace_dir: Path, record: dict):
        trace_dir.mkdir(parents=True, exist_ok=True)
        rid = record["record_id"]
        (trace_dir / f"{record.get('code_name', 'x')}_{rid}.json").write_text(
            json.dumps(record)
        )

    def test_incremental_and_idempotent(self, tmp_path):
        """Each tick indexes only new records; a re-tick with nothing new is a
        no-op (the bug the cursor-less batch had — re-indexing everything)."""
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            pdir = tmp_path / "proj"
            (pdir / ".dsagt").mkdir(parents=True)
            kb = KnowledgeBase(index_dir=pdir / "kb")
            indexer = CodeUseIndexer(kb, pdir)

            r1 = make_wrapper_record()
            r1["record_id"] = "r1"
            self._write(pdir / "trace_archive", r1)
            assert indexer.tick() == 1  # first record indexed
            assert indexer.tick() == 0  # nothing new → no-op (idempotent)

            r2 = make_wrapper_record()
            r2["record_id"] = "r2"
            self._write(pdir / "trace_archive", r2)
            assert indexer.tick() == 1  # only the new one
            assert indexer.tick() == 0

            # The ack set persists exactly the indexed record ids.
            acks = json.loads((pdir / ".dsagt" / "code_use_acks.json").read_text())
            assert set(acks) == {"r1", "r2"}
            kb.close()


# ---------------------------------------------------------------------------
# index_trace_archive
# ---------------------------------------------------------------------------


class TestIndexTraceArchive:

    def _write_records(self, trace_dir: Path, records: list[dict]):
        trace_dir.mkdir(parents=True, exist_ok=True)
        for i, record in enumerate(records):
            rid = record.get("record_id", f"rec_{i}")
            path = trace_dir / f"{record.get('code_name', 'unknown')}_{rid}.json"
            path.write_text(json.dumps(record))

    def test_indexes_all_records(self, tmp_path):
        """Indexes all records in trace_dir."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(
            trace_dir,
            [
                make_wrapper_record(record_id="t1"),
                make_wrapper_record(tool="megahit", record_id="t2"),
            ],
        )

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
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
        self._write_records(
            trace_dir,
            [
                make_wrapper_record(record_id="t1"),
                make_wrapper_record(record_id="t2"),
            ],
        )

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
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
        self._write_records(
            trace_dir,
            [
                make_wrapper_record(record_id="t1"),
                make_wrapper_record(record_id="t2"),
            ],
        )

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
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

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
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
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(tmp_path / "missing", kb)

            assert result["indexed"] == 0
            kb.close()

    def test_skips_records_without_execution(self, tmp_path):
        """Records with no execution layer are skipped (counted as errors)."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(
            trace_dir,
            [
                {"record_id": "bad", "code_name": "unknown"},
                make_wrapper_record(record_id="good"),
            ],
        )

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = index_trace_archive(trace_dir, kb)

            assert result["indexed"] == 1
            assert result["errors"] == 1
            kb.close()

    def test_accepts_wrapper_only_records(self, tmp_path):
        """Wrapper records (execution only) are valid and indexed."""
        trace_dir = tmp_path / "trace_archive"
        self._write_records(
            trace_dir,
            [
                make_wrapper_record(record_id="w1"),
            ],
        )

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
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
        self._write_records(trace_dir, [make_wrapper_record(record_id="t1")])

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
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
            path = trace_dir / f"{r.get('code_name', 'unknown')}_{rid}.json"
            path.write_text(json.dumps(r))
        return trace_dir

    def test_search_by_tool_name(self, tmp_path):
        """Index multiple records, search filtered by code_name."""
        records = [
            make_wrapper_record(tool="fastp", session="s1", record_id="t1"),
            make_wrapper_record(tool="megahit", session="s1", record_id="t2"),
            make_wrapper_record(tool="fastp", session="s2", record_id="t3"),
        ]
        trace_dir = self._index_records(tmp_path, records)

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            index_trace_archive(trace_dir, kb)

            results = kb.search(
                "quality filtering parameters",
                collection=COLLECTION_NAME,
                top_k=10,
                rerank=False,
                where={"code_name": "fastp"},
            )
            for r in results:
                assert r["chunk"]["metadata"]["code_name"] == "fastp"
            kb.close()

    def test_search_by_session(self, tmp_path):
        """Index multiple records, search filtered by session_id."""
        records = [
            make_wrapper_record(tool="fastp", session="s1", record_id="t1"),
            make_wrapper_record(tool="fastp", session="s2", record_id="t2"),
        ]
        trace_dir = self._index_records(tmp_path, records)

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            index_trace_archive(trace_dir, kb)

            results = kb.search(
                "fastp",
                collection=COLLECTION_NAME,
                top_k=10,
                rerank=False,
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

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            index_trace_archive(trace_dir, kb)

            results = kb.search(
                "tool execution",
                collection=COLLECTION_NAME,
                top_k=10,
                rerank=False,
                where={"return_code": 1},
            )
            for r in results:
                assert r["chunk"]["metadata"]["return_code"] == 1
            kb.close()
