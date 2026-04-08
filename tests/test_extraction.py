"""
Tests for session logging and memory extraction.

Tests the proxy callback's session log writing (log_exchange,
_extract_response_content) and the extraction module's prompt building,
response parsing, and end-to-end extract_session flow.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dsagt.proxy_callback import (
    SESSION_LOG_FILE,
    ToolRecordStore,
    _extract_response_content,
    _handle_success,
)
from dsagt.memory_extraction import (
    COLLECTION_NAME,
    build_extraction_prompt,
    delete_session_log,
    drain_session_log,
    extract_session,
    load_session_log,
    parse_extraction_response,
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


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# _extract_response_content
# ---------------------------------------------------------------------------

class TestExtractResponseContent:

    def test_anthropic_format(self):
        response = {
            "content": [
                {"type": "text", "text": "I'll run fastp."},
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "fastp"}},
            ]
        }
        result = _extract_response_content(response)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"

    def test_openai_format_text(self):
        response = {
            "choices": [{"message": {"content": "Hello!", "tool_calls": None}}]
        }
        result = _extract_response_content(response)
        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "Hello!"}

    def test_openai_format_tool_call(self):
        response = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }],
                }
            }]
        }
        result = _extract_response_content(response)
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "bash"

    def test_empty_response(self):
        assert _extract_response_content({}) == []


# ---------------------------------------------------------------------------
# ToolRecordStore.log_exchange
# ---------------------------------------------------------------------------

class TestLogExchange:

    @pytest.fixture
    def store(self, tmp_path):
        return ToolRecordStore(records_dir=tmp_path, session_id="test-session")

    def test_writes_session_log(self, store):
        """log_exchange creates session_log.jsonl with call_id."""
        kwargs = {
            "model": "test-model",
            "litellm_call_id": "call_xyz",
            "messages": [{"role": "user", "content": "hello"}],
        }
        response = {"content": [{"type": "text", "text": "Hi!"}]}
        store.log_exchange(kwargs, response)

        assert store.session_log_path.exists()
        entries = [json.loads(line) for line in store.session_log_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["model"] == "test-model"
        assert entries[0]["call_id"] == "call_xyz"
        assert entries[0]["new_messages"] == [{"role": "user", "content": "hello"}]

    def test_deduplicates_messages(self, store):
        """Only new messages since last call are logged."""
        msgs1 = [{"role": "user", "content": "first"}]
        msgs2 = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]

        store.log_exchange({"model": "m", "messages": msgs1}, {"content": []})
        store.log_exchange({"model": "m", "messages": msgs2}, {"content": []})

        entries = [json.loads(line) for line in store.session_log_path.read_text().splitlines()]
        assert len(entries) == 2
        assert len(entries[0]["new_messages"]) == 1
        assert entries[0]["new_messages"][0]["content"] == "first"
        assert len(entries[1]["new_messages"]) == 2
        assert entries[1]["new_messages"][0]["content"] == "ok"
        assert entries[1]["new_messages"][1]["content"] == "second"

    def test_appends_to_existing_log(self, store):
        """Multiple exchanges append to the same file."""
        for i in range(3):
            store.log_exchange(
                {"model": "m", "messages": [{"role": "user", "content": f"msg{i}"}]},
                {"content": []},
            )
            store._logged_message_count = 0  # Reset for test simplicity

        lines = store.session_log_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_response_content_extracted(self, store):
        """Response content is extracted, not the full response object."""
        kwargs = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        response = {
            "choices": [{"message": {"content": "Hello!", "tool_calls": None}}]
        }
        store.log_exchange(kwargs, response)

        entry = json.loads(store.session_log_path.read_text().strip())
        assert entry["response"] == [{"type": "text", "text": "Hello!"}]


class TestHandleSuccessSessionLog:

    def test_handle_success_logs_exchange(self, tmp_path):
        """_handle_success writes to the session log."""
        store = ToolRecordStore(records_dir=tmp_path, session_id="test")
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "run ls"}],
        }
        response = {"choices": [{"message": {"content": "Here are the files.", "tool_calls": None}}]}

        _handle_success(store, kwargs, response, _now(), _now())

        assert store.session_log_path.exists()
        entry = json.loads(store.session_log_path.read_text().strip())
        assert entry["new_messages"][0]["content"] == "run ls"
        assert entry["response"][0]["text"] == "Here are the files."


# ---------------------------------------------------------------------------
# load_session_log / delete_session_log
# ---------------------------------------------------------------------------

class TestSessionLogIO:

    def test_load_empty(self, tmp_path):
        assert load_session_log(tmp_path) == []

    def test_load_nonexistent_dir(self, tmp_path):
        assert load_session_log(tmp_path / "missing") == []

    def test_load_returns_entries(self, tmp_path):
        log_path = tmp_path / SESSION_LOG_FILE
        entries = [
            {"timestamp": "t1", "model": "m", "new_messages": [], "response": []},
            {"timestamp": "t2", "model": "m", "new_messages": [], "response": []},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        result = load_session_log(tmp_path)
        assert len(result) == 2

    def test_delete_removes_file(self, tmp_path):
        log_path = tmp_path / SESSION_LOG_FILE
        log_path.write_text("{}\n")

        assert delete_session_log(tmp_path) is True
        assert not log_path.exists()

    def test_delete_nonexistent(self, tmp_path):
        assert delete_session_log(tmp_path) is False


class TestDrainSessionLog:

    def test_drain_reads_and_removes(self, tmp_path):
        """drain reads entries and removes the file atomically."""
        log_path = tmp_path / SESSION_LOG_FILE
        entries = [
            {"timestamp": "t1", "model": "m", "new_messages": [], "response": []},
            {"timestamp": "t2", "model": "m", "new_messages": [], "response": []},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        result = drain_session_log(tmp_path)
        assert len(result) == 2
        assert not log_path.exists()
        assert not log_path.with_suffix(".consumed").exists()

    def test_drain_empty(self, tmp_path):
        assert drain_session_log(tmp_path) == []

    def test_drain_allows_new_writes(self, tmp_path):
        """After drain, new appends create a fresh file."""
        log_path = tmp_path / SESSION_LOG_FILE
        log_path.write_text(json.dumps({"old": True}) + "\n")

        drained = drain_session_log(tmp_path)
        assert len(drained) == 1

        # New write goes to fresh file
        with open(log_path, "a") as f:
            f.write(json.dumps({"new": True}) + "\n")

        assert log_path.exists()
        fresh = load_session_log(tmp_path)
        assert len(fresh) == 1
        assert fresh[0]["new"] is True


class TestVolumeTriggeredExtraction:

    def test_counter_increments(self, tmp_path):
        """Exchange counter tracks calls to log_exchange."""
        store = ToolRecordStore(records_dir=tmp_path, session_id="test")
        for i in range(5):
            store.log_exchange(
                {"model": "m", "messages": [{"role": "user", "content": f"msg{i}"}]},
                {"content": []},
            )
            store._logged_message_count = 0

        assert store._exchange_count == 5

    def test_threshold_zero_no_trigger(self, tmp_path):
        """Threshold of 0 disables auto-extraction."""
        store = ToolRecordStore(records_dir=tmp_path, session_id="test")
        assert store._extraction_threshold == 0

        for i in range(100):
            store.log_exchange(
                {"model": "m", "messages": [{"role": "user", "content": f"msg{i}"}]},
                {"content": []},
            )
            store._logged_message_count = 0

        assert not store._extracting


# ---------------------------------------------------------------------------
# build_extraction_prompt
# ---------------------------------------------------------------------------

class TestBuildExtractionPrompt:

    def _make_exchanges(self):
        return [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "model": "claude-sonnet-4-20250514",
                "new_messages": [{"role": "user", "content": "Run fastp on sample1.fq.gz"}],
                "response": [
                    {"type": "text", "text": "I'll run fastp with Q20 filtering."},
                    {"type": "tool_use", "name": "bash", "input": {"cmd": "fastp -q 20 --in1 sample1.fq.gz"}},
                ],
            },
            {
                "timestamp": "2024-01-15T10:31:00Z",
                "model": "claude-sonnet-4-20250514",
                "new_messages": [
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "98% reads passed"},
                    ]},
                ],
                "response": [
                    {"type": "text", "text": "Filtering complete. 98% of reads passed Q20."},
                ],
            },
        ]

    def test_includes_conversation(self):
        prompt = build_extraction_prompt(self._make_exchanges())
        assert "fastp" in prompt
        assert "sample1.fq.gz" in prompt
        assert "98% reads passed" in prompt

    def test_includes_categories(self):
        prompt = build_extraction_prompt(self._make_exchanges())
        assert "quality_control" in prompt
        assert "data_management" in prompt

    def test_custom_categories_merged(self):
        custom = {"my_category": "custom stuff"}
        prompt = build_extraction_prompt(self._make_exchanges(), categories=custom)
        assert "my_category" in prompt
        assert "custom stuff" in prompt
        assert "quality_control" in prompt  # stock still present

    def test_output_format_described(self):
        prompt = build_extraction_prompt(self._make_exchanges())
        assert '"facts"' in prompt
        assert '"summary"' in prompt
        assert '"insights"' in prompt

    def test_empty_exchanges(self):
        prompt = build_extraction_prompt([])
        assert "facts" in prompt  # prompt structure still present


# ---------------------------------------------------------------------------
# parse_extraction_response
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:

    def test_valid_json(self):
        response = json.dumps({
            "facts": [{"text": "fastp used Q20", "category": "quality_control"}],
            "summary": "Ran fastp on sample1.",
            "insights": [{"text": "Q20 is sufficient for isolates", "category": "quality_control"}],
        })
        result = parse_extraction_response(response)
        assert len(result["facts"]) == 1
        assert result["summary"] == "Ran fastp on sample1."
        assert len(result["insights"]) == 1

    def test_strips_markdown_fences(self):
        response = "```json\n" + json.dumps({
            "facts": [], "summary": "test", "insights": [],
        }) + "\n```"
        result = parse_extraction_response(response)
        assert result["summary"] == "test"

    def test_missing_fields_default(self):
        result = parse_extraction_response("{}")
        assert result["facts"] == []
        assert result["summary"] == ""
        assert result["insights"] == []


# ---------------------------------------------------------------------------
# extract_session (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractSession:

    def _write_session_log(self, trace_dir: Path, exchanges: list[dict]):
        trace_dir.mkdir(parents=True, exist_ok=True)
        log_path = trace_dir / SESSION_LOG_FILE
        log_path.write_text("\n".join(json.dumps(e) for e in exchanges))

    def _mock_llm_response(self):
        return json.dumps({
            "facts": [
                {"text": "fastp was run with Q20 on sample1", "category": "quality_control"},
                {"text": "98% of reads passed filtering", "category": "results"},
            ],
            "summary": "Ran quality filtering on sample1 using fastp with Q20 threshold.",
            "insights": [
                {"text": "Q20 filtering is sufficient for high-quality isolate data", "category": "quality_control"},
            ],
        })

    def test_extracts_and_stores(self, tmp_path):
        """End-to-end: load log → call LLM → store in KB → delete log."""
        trace_dir = tmp_path / "trace_archive"
        self._write_session_log(trace_dir, [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "model": "m",
                "new_messages": [{"role": "user", "content": "run fastp"}],
                "response": [{"type": "text", "text": "Running fastp."}],
            },
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")

            with patch("dsagt.memory_extraction.call_extraction_llm") as mock_llm:
                mock_llm.return_value = self._mock_llm_response()

                result = extract_session(
                    trace_dir=trace_dir,
                    kb=kb,
                    api_key="test-key",
                    session_id="test-session",
                )

            assert result["status"] == "ok"
            assert result["facts"] == 2
            assert result["insights"] == 1
            assert result["summary"] == 1
            assert result["total_entries"] == 4  # 2 facts + 1 summary + 1 insight

            # Session log should be deleted
            assert not (trace_dir / SESSION_LOG_FILE).exists()

            # Entries should be searchable
            results = kb.search(
                "fastp quality",
                collection=COLLECTION_NAME,
                top_k=10, rerank=False,
            )
            assert len(results) > 0

            kb.close()

    def test_empty_log_returns_early(self, tmp_path):
        """No session log → returns status=empty, no LLM call."""
        trace_dir = tmp_path / "trace_archive"
        trace_dir.mkdir()

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")

            with patch("dsagt.memory_extraction.call_extraction_llm") as mock_llm:
                result = extract_session(
                    trace_dir=trace_dir,
                    kb=kb,
                    api_key="test-key",
                )

            assert result["status"] == "empty"
            mock_llm.assert_not_called()
            kb.close()

    def test_stores_with_correct_metadata(self, tmp_path):
        """Each entry has source_type, category, session_id, timestamps, and trace_refs."""
        trace_dir = tmp_path / "trace_archive"
        self._write_session_log(trace_dir, [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "call_id": "call_abc",
                "model": "m",
                "new_messages": [{"role": "user", "content": "test"}],
                "response": [{"type": "text", "text": "ok"}],
            },
            {
                "timestamp": "2024-01-15T10:35:00Z",
                "call_id": "call_def",
                "model": "m",
                "new_messages": [{"role": "user", "content": "next"}],
                "response": [{"type": "text", "text": "done"}],
            },
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")

            with patch("dsagt.memory_extraction.call_extraction_llm") as mock_llm:
                mock_llm.return_value = self._mock_llm_response()

                extract_session(
                    trace_dir=trace_dir,
                    kb=kb,
                    api_key="test-key",
                    session_id="s1",
                )

            results = kb.search(
                "fastp", collection=COLLECTION_NAME,
                top_k=10, rerank=False,
            )

            source_types = {r["chunk"]["metadata"]["source_type"] for r in results}
            assert "extraction" in source_types or "summary" in source_types or "insight" in source_types

            for r in results:
                meta = r["chunk"]["metadata"]
                assert meta["session_id"] == "s1"
                assert meta["timestamp_start"] == "2024-01-15T10:30:00Z"
                assert meta["timestamp_end"] == "2024-01-15T10:35:00Z"
                assert "call_abc" in meta["trace_refs"]
                assert "call_def" in meta["trace_refs"]

            kb.close()

    def test_session_log_deleted_after_extraction(self, tmp_path):
        """The session log is always deleted, even if storage succeeds partially."""
        trace_dir = tmp_path / "trace_archive"
        self._write_session_log(trace_dir, [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "model": "m",
                "new_messages": [{"role": "user", "content": "test"}],
                "response": [{"type": "text", "text": "ok"}],
            },
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")

            with patch("dsagt.memory_extraction.call_extraction_llm") as mock_llm:
                mock_llm.return_value = json.dumps({
                    "facts": [], "summary": "", "insights": [],
                })

                extract_session(
                    trace_dir=trace_dir,
                    kb=kb,
                    api_key="test-key",
                )

            assert not (trace_dir / SESSION_LOG_FILE).exists()
            kb.close()

    def test_filtered_search_by_source_type(self, tmp_path):
        """Extracted entries can be filtered by source_type."""
        trace_dir = tmp_path / "trace_archive"
        self._write_session_log(trace_dir, [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "model": "m",
                "new_messages": [{"role": "user", "content": "test"}],
                "response": [{"type": "text", "text": "ok"}],
            },
        ])

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")

            with patch("dsagt.memory_extraction.call_extraction_llm") as mock_llm:
                mock_llm.return_value = self._mock_llm_response()

                extract_session(
                    trace_dir=trace_dir,
                    kb=kb,
                    api_key="test-key",
                    session_id="s1",
                )

            # Filter for just insights
            results = kb.search(
                "quality filtering",
                collection=COLLECTION_NAME,
                top_k=10, rerank=False,
                where={"source_type": "insight"},
            )

            for r in results:
                assert r["chunk"]["metadata"]["source_type"] == "insight"

            kb.close()
