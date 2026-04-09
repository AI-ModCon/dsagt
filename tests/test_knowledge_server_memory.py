"""
Tests for kb_remember, kb_get_memories, and extended kb_search handlers.

Drop this file into tests/test_knowledge_server_memory.py

These tests use a real ExplicitMemory (file-backed, no mocking needed)
and a mocked KnowledgeBase (same pattern as existing server tests).
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import mcp.types as types

from dsagt.commands.knowledge_server import create_knowledge_server
from dsagt.memory import ExplicitMemory
from mcp_helpers import call_tool_json as call_tool


def make_search_result(text, source_file, chunk_index=0, score=0.9):
    return {
        "chunk": {
            "text": text,
            "metadata": {
                "source_file": source_file,
                "collection": "test_collection",
                "chunk_index": chunk_index,
                "file_type": ".md",
            },
        },
        "score": score,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kb(tmp_path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.list_collections.return_value = [
        {"name": "docs", "description": "Documentation"},
    ]
    kb.search.return_value = [
        make_search_result("Result one", "/path/to/file.md", 0, 0.95),
    ]
    kb.ingest.return_value = {"collection": "docs", "files": 1, "chunks": 10}
    kb.append.return_value = {"collection": "docs", "files": 1, "chunks_added": 5, "total_chunks": 15}
    return kb


@pytest.fixture
def server(mock_kb, tmp_path):
    return create_knowledge_server(mock_kb, use_rerank=False, runtime_dir=tmp_path)


@pytest.fixture
def memory(tmp_path):
    return ExplicitMemory(runtime_dir=tmp_path)


# ---------------------------------------------------------------------------
# kb_remember
# ---------------------------------------------------------------------------


class TestKbRemember:

    def test_stores_a_fact(self, server):
        result = call_tool(server, "kb_remember", {
            "text": "fastp quality threshold is Q20",
        })

        assert result["status"] == "ok"
        assert result["entry_id"]
        assert result["total_memories"] == 1

    def test_stores_with_metadata(self, server):
        result = call_tool(server, "kb_remember", {
            "text": "some fact",
            "category": "quality_control",
            "session_id": "sess_01",
        })

        assert result["status"] == "ok"

    def test_supersede_existing(self, server):
        r1 = call_tool(server, "kb_remember", {
            "text": "old threshold Q20",
        })
        r2 = call_tool(server, "kb_remember", {
            "text": "new threshold Q30",
            "supersedes": r1["entry_id"],
        })

        assert r2["status"] == "ok"
        assert r2["superseded_id"] == r1["entry_id"]
        assert r2["total_memories"] == 1

    def test_supersede_nonexistent_returns_error(self, server):
        result = call_tool(server, "kb_remember", {
            "text": "new fact",
            "supersedes": "bad_id",
        })

        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_multiple_facts(self, server):
        call_tool(server, "kb_remember", {"text": "fact one"})
        call_tool(server, "kb_remember", {"text": "fact two"})
        result = call_tool(server, "kb_remember", {"text": "fact three"})

        assert result["total_memories"] == 3


# ---------------------------------------------------------------------------
# kb_get_memories
# ---------------------------------------------------------------------------


class TestKbGetMemories:

    def test_empty_returns_zero(self, server):
        result = call_tool(server, "kb_get_memories", {})

        assert result["status"] == "ok"
        assert result["count"] == 0
        assert result["memories"] == []

    def test_returns_stored_memories(self, server):
        call_tool(server, "kb_remember", {"text": "fact one"})
        call_tool(server, "kb_remember", {"text": "fact two"})

        result = call_tool(server, "kb_get_memories", {})

        assert result["count"] == 2
        texts = {m["text"] for m in result["memories"]}
        assert texts == {"fact one", "fact two"}

    def test_excludes_superseded(self, server):
        r1 = call_tool(server, "kb_remember", {"text": "old fact"})
        call_tool(server, "kb_remember", {
            "text": "new fact",
            "supersedes": r1["entry_id"],
        })

        result = call_tool(server, "kb_get_memories", {})

        assert result["count"] == 1
        assert result["memories"][0]["text"] == "new fact"


# ---------------------------------------------------------------------------
# kb_search — multi-collection fan-out
# ---------------------------------------------------------------------------


class TestKbSearchMultiCollection:

    def test_single_collection_backward_compat(self, server, mock_kb):
        """Plain search with collection still works."""
        result = call_tool(server, "kb_search", {
            "query": "test",
            "collection": "docs",
        })

        assert result["status"] == "ok"
        mock_kb.search.assert_called_once()

    def test_multi_collection_fanout(self, server, mock_kb):
        """Searching multiple collections calls search for each."""
        mock_kb.search.return_value = [
            make_search_result("result", "/file.md", 0, 0.9),
        ]

        result = call_tool(server, "kb_search", {
            "query": "test",
            "collections": ["docs", "papers"],
        })

        assert result["status"] == "ok"
        assert mock_kb.search.call_count == 2

    def test_no_collection_returns_error(self, server):
        result = call_tool(server, "kb_search", {
            "query": "test",
        })

        assert result["status"] == "error"

    def test_multi_collection_merges_results(self, server, mock_kb):
        """Results from multiple collections are merged and sorted."""
        call_count = [0]

        def varying_results(**kwargs):
            call_count[0] += 1
            score = 0.9 if call_count[0] == 1 else 0.7
            return [make_search_result(
                f"result_{call_count[0]}",
                f"/file_{call_count[0]}.md",
                score=score,
            )]

        mock_kb.search.side_effect = varying_results

        result = call_tool(server, "kb_search", {
            "query": "test",
            "collections": ["docs", "papers"],
            "top_k": 5,
        })

        assert result["result_count"] == 2
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_missing_collection_skipped(self, server, mock_kb):
        """A missing collection logs a warning but doesn't fail the search."""
        def search_with_error(**kwargs):
            if kwargs["collection"] == "missing":
                raise ValueError("Collection 'missing' not found")
            return [make_search_result("result", "/file.md")]

        mock_kb.search.side_effect = search_with_error

        result = call_tool(server, "kb_search", {
            "query": "test",
            "collections": ["docs", "missing"],
        })

        assert result["status"] == "ok"
        assert result["result_count"] == 1


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:

    def _get_tool(self, server, name):
        req = types.ListToolsRequest(method="tools/list")
        handler = server.request_handlers[types.ListToolsRequest]
        result = asyncio.run(handler(req))
        for tool in result.root.tools:
            if tool.name == name:
                return tool
        return None

    def test_kb_remember_exists(self, server):
        tool = self._get_tool(server, "kb_remember")
        assert tool is not None
        assert "text" in tool.inputSchema["properties"]
        assert tool.inputSchema["required"] == ["text"]

    def test_kb_remember_has_optional_params(self, server):
        tool = self._get_tool(server, "kb_remember")
        for param in ("category", "session_id", "supersedes"):
            assert param in tool.inputSchema["properties"]

    def test_kb_get_memories_exists(self, server):
        tool = self._get_tool(server, "kb_get_memories")
        assert tool is not None

    def test_kb_search_has_collections_param(self, server):
        tool = self._get_tool(server, "kb_search")
        assert "collections" in tool.inputSchema["properties"]

    def test_kb_search_query_is_only_required(self, server):
        tool = self._get_tool(server, "kb_search")
        assert tool.inputSchema["required"] == ["query"]
