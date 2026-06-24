"""
Integration tests for the knowledge server with a real KnowledgeBase.

These tests require:
  - tests/test_site_config.yaml (copy from test_site_config.yaml.example)
  - A valid embedding API key and base URL in the config
  - Network access to the embedding API

Tests FAIL if the config or credentials are missing.

Usage:
    pytest test_knowledge_integration.py -v
    pytest test_knowledge_integration.py -v -k test_search
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from dsagt.knowledge import KnowledgeBase
from dsagt.mcp.knowledge_tools import create_knowledge_server, setup_runtime_kb
from mcp_helpers import call_tool_json as call_tool, call_tool_async as _call_tool_async_raw


async def _call_tool_async(server, name: str, arguments: dict) -> dict:
    """Async tool call returning parsed JSON."""
    return json.loads(await _call_tool_async_raw(server, name, arguments))


async def call_tool_and_await_job(server, name: str, arguments: dict) -> tuple[dict, dict]:
    """Call a tool that starts a background job, wait for it, return (initial, final)."""
    initial = await _call_tool_async(server, name, arguments)
    assert initial["status"] == "started"
    job_id = initial["job_id"]

    for _ in range(600):  # 60s timeout for real embedding API calls
        await asyncio.sleep(0.1)
        status = await _call_tool_async(server, "kb_job_status", {"job_id": job_id})
        if status["status"] != "running":
            return initial, status

    raise TimeoutError(f"Job {job_id} did not complete")


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


def _make_kb(tmp_path, embedding_config):
    """Create a KnowledgeBase configured for the current institution.

    Env vars (OPENAI_BASE_URL, LLM_API_KEY, EMBEDDING_MODEL) are set
    by the embedding_config fixture in conftest.py.
    """
    index_dir = tmp_path / "kb_index"
    index_dir.mkdir()
    return KnowledgeBase(
        index_dir=index_dir,
        default_embedder="api",
    )


@pytest.fixture
def kb_server(tmp_path, smoke_test_dir, embedding_config):
    """Knowledge server backed by a real KnowledgeBase.

    Ingests the smoke test documents, then provides a server for search tests.
    Uses embedding_config fixture (skips if no site config or API key).
    """
    kb = _make_kb(tmp_path, embedding_config)
    result = kb.ingest(smoke_test_dir)
    assert result["chunks"] > 0, "Ingest produced no chunks"

    server = create_knowledge_server(kb, use_rerank=False)
    yield server
    kb.close()


# ---------------------------------------------------------------------------
# Ingest via MCP handler
# ---------------------------------------------------------------------------

class TestIngestIntegration:

    def test_ingest_smoke_test_docs(self, tmp_path, smoke_test_dir, embedding_config):
        """Ingest smoke test documents through the MCP handler."""
        kb = _make_kb(tmp_path, embedding_config)
        server = create_knowledge_server(kb, use_rerank=False)

        async def run():
            initial, final = await call_tool_and_await_job(
                server, "kb_ingest", {"folder_path": str(smoke_test_dir)}
            )
            assert final["status"] == "complete"
            result = final["result"]
            assert result["files"] > 0
            assert result["chunks"] > 0
            assert result["collection"] == "knowledge"

        try:
            asyncio.run(run())
        finally:
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

    def test_symlinks_base_collections(self, tmp_path, smoke_test_dir, embedding_config):
        """setup_runtime_kb symlinks base collections into runtime."""
        base_dir = tmp_path / "base_kb"
        base_dir.mkdir()
        kb = _make_kb(tmp_path, embedding_config)
        # Point KB at base_dir for this test
        kb.index_dir = base_dir
        kb.index_dir.mkdir(exist_ok=True)
        kb.ingest(smoke_test_dir)
        kb.close()

        runtime_dir = tmp_path / "runtime"
        runtime_kb_dir = setup_runtime_kb(base_dir, runtime_dir)

        knowledge_link = runtime_kb_dir / "knowledge"
        assert knowledge_link.exists()
        assert knowledge_link.is_symlink()
        assert (knowledge_link / "index.faiss").exists()

    def test_runtime_search_via_symlink(self, tmp_path, smoke_test_dir, embedding_config):
        """A KB pointing at runtime symlinks can search successfully."""
        base_dir = tmp_path / "base_kb"
        base_dir.mkdir()
        kb = _make_kb(tmp_path, embedding_config)
        kb.index_dir = base_dir
        kb.index_dir.mkdir(exist_ok=True)
        kb.ingest(smoke_test_dir)
        kb.close()

        runtime_dir = tmp_path / "runtime"
        runtime_kb_dir = setup_runtime_kb(base_dir, runtime_dir)

        kb2 = KnowledgeBase(index_dir=runtime_kb_dir, default_embedder="api")
        results = kb2.search("large files", "knowledge", top_k=2, rerank=False)
        assert len(results) > 0
        kb2.close()
