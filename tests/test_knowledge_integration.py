"""
Integration tests for the knowledge server with a real KnowledgeBase.

These tests require:
  - An embedding API key (LLM_API_KEY or OPENAI_API_KEY env var)
  - Network access to the embedding API

Skip condition: tests are skipped if no API key is available.

Usage:
    pytest test_knowledge_integration.py -v
    pytest test_knowledge_integration.py -v -k test_search
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest
import mcp.types as types
from dsagt.knowledge import KnowledgeBase

from dsagt.knowledge_server import create_knowledge_server, setup_runtime_kb

# Skip all tests in this module if no API key
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
pytestmark = pytest.mark.skipif(not API_KEY, reason="No embedding API key set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_tool(server, name: str, arguments: dict) -> dict:
    """Invoke a tool handler and return the parsed JSON response."""
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    handler = server.request_handlers[types.CallToolRequest]
    result = asyncio.run(handler(req))
    return json.loads(result.root.content[0].text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def smoke_test_dir():
    """Path to the smoke test knowledge documents."""
    path = Path(__file__).parent / "smoke_test" / "knowledge"
    if not path.exists():
        pytest.skip(f"Smoke test knowledge dir not found: {path}")
    return path


@pytest.fixture
def kb_server(tmp_path, smoke_test_dir):
    """
    Knowledge server backed by a real KnowledgeBase.

    Ingests the smoke test documents, then provides a server for search tests.
    """

    index_dir = tmp_path / "kb_index"
    index_dir.mkdir()

    kb = KnowledgeBase(index_dir=index_dir)
    # Ingest the smoke test docs
    result = kb.ingest(smoke_test_dir)
    assert result["chunks"] > 0, "Ingest produced no chunks"

    server = create_knowledge_server(kb, use_rerank=False)
    yield server
    kb.close()


# ---------------------------------------------------------------------------
# Ingest via MCP handler
# ---------------------------------------------------------------------------

class TestIngestIntegration:

    def test_ingest_smoke_test_docs(self, tmp_path, smoke_test_dir):
        """Ingest smoke test documents through the MCP handler."""


        kb = KnowledgeBase(index_dir=tmp_path / "kb_index")
        server = create_knowledge_server(kb, use_rerank=False)

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(smoke_test_dir),
        })

        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_created"] > 0
        assert result["collection"] == "knowledge"
        kb.close()


# ---------------------------------------------------------------------------
# Search via MCP handler
# ---------------------------------------------------------------------------

class TestSearchIntegration:

    def test_search_returns_results(self, kb_server):
        """Search through MCP handler returns relevant chunks."""
        result = call_tool(kb_server, "kb_search", {
            "query": "how to handle large files",
            "collection": "knowledge",
        })

        assert result["status"] == "ok"
        assert result["result_count"] > 0
        # Troubleshooting doc should be the top hit
        sources = [r["source_file"] for r in result["results"]]
        assert any("troubleshooting" in s for s in sources)

    def test_search_nonexistent_collection(self, kb_server):
        """Searching a nonexistent collection returns an error."""
        result = call_tool(kb_server, "kb_search", {
            "query": "anything",
            "collection": "nonexistent",
        })

        assert result["status"] == "error"

    def test_search_with_top_k(self, kb_server):
        """top_k limits the number of results."""
        result = call_tool(kb_server, "kb_search", {
            "query": "installation",
            "collection": "knowledge",
            "top_k": 2,
        })

        assert result["status"] == "ok"
        assert result["result_count"] <= 2

    def test_search_result_format(self, kb_server):
        """Each result has expected fields."""
        result = call_tool(kb_server, "kb_search", {
            "query": "API reference",
            "collection": "knowledge",
            "top_k": 1,
        })

        assert result["status"] == "ok"
        if result["result_count"] > 0:
            hit = result["results"][0]
            assert "text" in hit
            assert "score" in hit
            assert "source_file" in hit
            assert "chunk_index" in hit


# ---------------------------------------------------------------------------
# List collections via MCP handler
# ---------------------------------------------------------------------------

class TestListCollectionsIntegration:

    def test_list_after_ingest(self, kb_server):
        """After ingesting, the collection appears in list."""
        result = call_tool(kb_server, "kb_list_collections", {})

        assert result["status"] == "ok"
        assert result["count"] >= 1
        names = [c["name"] for c in result["collections"]]
        assert "knowledge" in names


# ---------------------------------------------------------------------------
# Setup runtime KB with symlinks
# ---------------------------------------------------------------------------

class TestSetupRuntimeKB:

    def test_symlinks_base_collections(self, tmp_path, smoke_test_dir):
        """setup_runtime_kb symlinks base collections into runtime."""

        # Create a base index by ingesting
        base_dir = tmp_path / "base_kb"
        base_dir.mkdir()
        kb = KnowledgeBase(index_dir=base_dir)
        kb.ingest(smoke_test_dir)
        kb.close()

        # Now setup runtime from that base
        runtime_dir = tmp_path / "runtime"
        runtime_kb_dir = setup_runtime_kb(base_dir, runtime_dir)

        # Should have a symlink to the knowledge collection
        knowledge_link = runtime_kb_dir / "knowledge"
        assert knowledge_link.exists()
        assert knowledge_link.is_symlink()
        assert (knowledge_link / "index.faiss").exists()

    def test_runtime_search_via_symlink(self, tmp_path, smoke_test_dir):
        """A KB pointing at runtime symlinks can search successfully."""

        # Build base index
        base_dir = tmp_path / "base_kb"
        base_dir.mkdir()
        kb = KnowledgeBase(index_dir=base_dir)
        kb.ingest(smoke_test_dir)
        kb.close()

        # Setup runtime with symlinks
        runtime_dir = tmp_path / "runtime"
        runtime_kb_dir = setup_runtime_kb(base_dir, runtime_dir)

        # Search via runtime KB
        kb2 = KnowledgeBase(index_dir=runtime_kb_dir)
        results = kb2.search("large files", "knowledge", top_k=2, rerank=False)
        assert len(results) > 0
        kb2.close()
