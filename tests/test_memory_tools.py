"""
Tests for the explicit-memory MCP tools (kb_remember, kb_get_memories).

These tests use a real ExplicitMemory (file-backed, no mocking needed) and a
mocked KnowledgeBase (same pattern as the other server tests).  The tools live
in :mod:`dsagt.mcp.memory_tools`; ``create_memory_server`` exposes just that
concern for driving via ``call_tool_sync``.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
import mcp.types as types

from dsagt.mcp.memory_tools import create_memory_server
from dsagt.memory import ExplicitMemory
from mcp_helpers import call_tool_json as call_tool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kb(tmp_path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    return kb


@pytest.fixture
def server(mock_kb, tmp_path):
    return create_memory_server(mock_kb, runtime_dir=tmp_path)


@pytest.fixture
def memory(tmp_path):
    return ExplicitMemory(runtime_dir=tmp_path)


# ---------------------------------------------------------------------------
# kb_remember
# ---------------------------------------------------------------------------


class TestKbRemember:

    def test_stores_a_fact(self, server):
        result = call_tool(
            server,
            "kb_remember",
            {
                "text": "fastp quality threshold is Q20",
            },
        )

        assert result["status"] == "ok"
        assert result["entry_id"]
        assert result["total_memories"] == 1

    def test_stores_with_metadata(self, server):
        result = call_tool(
            server,
            "kb_remember",
            {
                "text": "some fact",
                "category": "quality_control",
                "session_id": "sess_01",
            },
        )

        assert result["status"] == "ok"

    def test_supersede_existing(self, server):
        r1 = call_tool(
            server,
            "kb_remember",
            {
                "text": "old threshold Q20",
            },
        )
        r2 = call_tool(
            server,
            "kb_remember",
            {
                "text": "new threshold Q30",
                "supersedes": r1["entry_id"],
            },
        )

        assert r2["status"] == "ok"
        assert r2["superseded_id"] == r1["entry_id"]
        assert r2["total_memories"] == 1

    def test_supersede_nonexistent_returns_error(self, server):
        result = call_tool(
            server,
            "kb_remember",
            {
                "text": "new fact",
                "supersedes": "bad_id",
            },
        )

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
        call_tool(
            server,
            "kb_remember",
            {
                "text": "new fact",
                "supersedes": r1["entry_id"],
            },
        )

        result = call_tool(server, "kb_get_memories", {})

        assert result["count"] == 1
        assert result["memories"][0]["text"] == "new fact"


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
