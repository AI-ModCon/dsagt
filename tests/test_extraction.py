"""
Tests for memory extraction (post-proxy).

Conversation history now comes from MLflow traces, not a local
``session_log.jsonl``.  Tests inject ``exchanges=[...]`` directly into
``extract_session`` so they don't need to mock the MLflow SDK; the
MLflow-trace-to-exchange formatter is exercised by its own unit tests
on ``_trace_to_exchange``.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dsagt.memory import (
    EPISODIC_COLLECTION as COLLECTION_NAME,
    _trace_to_exchange,
    build_extraction_prompt,
    extract_session,
    parse_extraction_response,
)
from dsagt.knowledge import KnowledgeBase


def fake_embed(texts: list[str]) -> np.ndarray:
    dim = 8
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.RandomState(hash(t) & 0xFFFFFFFF)
        vecs[i] = rng.randn(dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


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
                    {"type": "tool_use", "name": "bash",
                     "input": {"cmd": "fastp -q 20 --in1 sample1.fq.gz"}},
                ],
            },
            {
                "timestamp": "2024-01-15T10:31:00Z",
                "model": "claude-sonnet-4-20250514",
                "new_messages": [
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1",
                         "content": "98% reads passed"},
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
            "insights": [{"text": "Q20 is sufficient for isolates",
                          "category": "quality_control"}],
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
# _trace_to_exchange — MLflow trace row → extraction exchange dict
# ---------------------------------------------------------------------------

class TestTraceToExchange:

    def test_anthropic_response_shape(self):
        """Anthropic-shape response: ``content`` is already a block list."""
        row = {
            "request_time": "2024-05-01T12:00:00Z",
            "trace_id": "tr-abc",
            "request": json.dumps({
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
            }),
            "response": json.dumps({
                "content": [{"type": "text", "text": "hello"}],
            }),
        }
        ex = _trace_to_exchange(row)
        assert ex is not None
        assert ex["new_messages"] == [{"role": "user", "content": "hi"}]
        assert ex["response"] == [{"type": "text", "text": "hello"}]
        assert ex["trace_id"] == "tr-abc"
        assert ex["model"] == "claude-sonnet-4-5"

    def test_openai_response_shape_with_tool_calls(self):
        """OpenAI-shape: choices[].message.content + tool_calls."""
        row = {
            "request_time": "2024-05-01T12:00:00Z",
            "trace_id": "tr-xyz",
            "request": json.dumps({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "run it"}],
            }),
            "response": json.dumps({
                "choices": [{
                    "message": {
                        "content": "Sure",
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"cmd": "ls"}),
                            },
                        }],
                    },
                }],
            }),
        }
        ex = _trace_to_exchange(row)
        assert ex is not None
        types = [b["type"] for b in ex["response"]]
        assert "text" in types
        assert "tool_use" in types
        tool_use = next(b for b in ex["response"] if b["type"] == "tool_use")
        assert tool_use["name"] == "bash"
        assert tool_use["input"] == {"cmd": "ls"}

    def test_unrecognised_shape_returns_none(self):
        """Non-LLM-call traces (kb.* / tool.execute spans) get skipped."""
        row = {
            "request_time": "2024-05-01T12:00:00Z",
            "trace_id": "tr-kb",
            "request": json.dumps({"query": "stuff"}),  # no messages
            "response": "{}",
        }
        assert _trace_to_exchange(row) is None

    def test_handles_missing_response(self):
        row = {
            "trace_id": "tr-empty",
            "request": json.dumps({
                "messages": [{"role": "user", "content": "x"}],
            }),
            "response": None,
        }
        ex = _trace_to_exchange(row)
        assert ex is not None
        assert ex["response"] == []


# ---------------------------------------------------------------------------
# extract_session (end-to-end with injected exchanges)
# ---------------------------------------------------------------------------

class TestExtractSession:

    def _mock_llm_response(self):
        return json.dumps({
            "facts": [
                {"text": "fastp was run with Q20 on sample1",
                 "category": "quality_control"},
                {"text": "98% of reads passed filtering", "category": "results"},
            ],
            "summary": "Ran quality filtering on sample1 using fastp with Q20 threshold.",
            "insights": [
                {"text": "Q20 filtering is sufficient for high-quality isolate data",
                 "category": "quality_control"},
            ],
        })

    def test_extracts_and_stores(self, tmp_path):
        """End-to-end: injected exchanges → mocked LLM → KB write."""
        exchanges = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "trace_id": "tr-1",
                "model": "m",
                "new_messages": [{"role": "user", "content": "run fastp"}],
                "response": [{"type": "text", "text": "Running fastp."}],
            },
        ]
        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            with patch("dsagt.memory.call_extraction_llm") as mock_llm:
                mock_llm.return_value = self._mock_llm_response()
                result = extract_session(
                    project_name="proj",
                    kb=kb,
                    api_key="test-key",
                    session_id="test-session",
                    exchanges=exchanges,
                )
            assert result["status"] == "ok"
            assert result["facts"] == 2
            assert result["insights"] == 1
            assert result["summary"] == 1
            kb.close()

    def test_empty_exchanges_returns_status_empty(self, tmp_path):
        """No exchanges → returns status=empty, no LLM call."""
        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            with patch("dsagt.memory.call_extraction_llm") as mock_llm:
                result = extract_session(
                    project_name="proj",
                    kb=kb,
                    api_key="test-key",
                    session_id="test-session",
                    exchanges=[],
                )
                mock_llm.assert_not_called()
            assert result["status"] == "empty"
            kb.close()

    def test_missing_session_id_returns_empty(self, tmp_path):
        """``session_id`` is required when ``exchanges`` isn't supplied."""
        with patch("dsagt.knowledge._make_embedder") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            result = extract_session(
                project_name="proj", kb=kb, api_key="test-key",
            )
            assert result["status"] == "empty"
            assert result.get("reason") == "no_session_id"
            kb.close()
