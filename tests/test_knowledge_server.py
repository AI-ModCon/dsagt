"""
Tests for the knowledge base MCP server.

Tests the tool handlers and setup_runtime_kb utility.
The KnowledgeBase is mocked — these tests verify the server's
handler logic, argument parsing, error handling, and response formatting.

Ingest and append are background jobs: the handler returns immediately
with {"status": "started", "job_id": ...}. Tests that need to verify
the job completed use an async helper that lets the event loop tick.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import mcp.types as types

from dsagt.commands.knowledge_server import create_knowledge_server, setup_runtime_kb
from mcp_helpers import call_tool_json as call_tool


async def _call_tool_async(server, name: str, arguments: dict) -> dict:
    """Invoke a tool handler inside a running event loop."""
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    handler = server.request_handlers[types.CallToolRequest]
    result = await handler(req)
    return json.loads(result.root.content[0].text)


async def call_tool_and_await_job(server, name: str, arguments: dict) -> tuple[dict, dict]:
    """Call a tool that starts a background job, wait for it, return (initial, final)."""
    initial = await _call_tool_async(server, name, arguments)
    assert initial["status"] == "started"
    job_id = initial["job_id"]

    # Let the background task complete
    for _ in range(100):
        await asyncio.sleep(0.01)
        status = await _call_tool_async(server, "kb_job_status", {"job_id": job_id})
        if status["status"] != "running":
            return initial, status

    raise TimeoutError(f"Job {job_id} did not complete")


def make_search_result(text: str, source_file: str, chunk_index: int = 0, score: float = 0.9):
    """Create a search result in the format KnowledgeBase.search returns."""
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
    """A mocked KnowledgeBase with default behaviors."""
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.default_rerank = True
    kb.list_collections.return_value = [
        {"name": "docs", "description": "Project documentation"},
        {"name": "papers", "description": "Research papers"},
    ]
    kb.search.return_value = [
        make_search_result("First result text", "/path/to/file1.md", 0, 0.95),
        make_search_result("Second result text", "/path/to/file2.md", 1, 0.80),
    ]
    kb.ingest.return_value = {"collection": "new_docs", "files": 5, "chunks": 42}
    kb.append.return_value = {"collection": "docs", "files": 2, "chunks_added": 10, "total_chunks": 50}
    return kb


@pytest.fixture
def server(mock_kb):
    """Knowledge server with mocked KB (reranking enabled via mock_kb.default_rerank)."""
    return create_knowledge_server(mock_kb)


# ---------------------------------------------------------------------------
# kb_list_collections
# ---------------------------------------------------------------------------

class TestListCollections:

    def test_returns_collections(self, server, mock_kb):
        """Lists all collections with descriptions."""
        result = call_tool(server, "kb_list_collections", {})

        assert result["status"] == "ok"
        assert result["count"] == 2
        assert len(result["collections"]) == 2
        assert result["collections"][0]["name"] == "docs"
        mock_kb.list_collections.assert_called_once()

    def test_empty_collections(self, mock_kb):
        """Empty knowledge base returns zero count."""
        mock_kb.list_collections.return_value = []
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_list_collections", {})

        assert result["status"] == "ok"
        assert result["count"] == 0
        assert result["collections"] == []


# ---------------------------------------------------------------------------
# kb_search
# ---------------------------------------------------------------------------

class TestSearch:

    def test_search_success(self, server, mock_kb):
        """Successful search returns formatted results."""
        result = call_tool(server, "kb_search", {
            "query": "how to install",
            "collection": "docs",
        })

        assert result["status"] == "ok"
        assert result["query"] == "how to install"
        assert result["collection"] == "docs"
        assert result["result_count"] == 2

        first = result["results"][0]
        assert first["text"] == "First result text"
        assert first["score"] == 0.95
        assert first["source_file"] == "/path/to/file1.md"
        assert first["chunk_index"] == 0

    def test_search_passes_parameters(self, server, mock_kb):
        """Search forwards top_k and rerank to the knowledge base."""
        call_tool(server, "kb_search", {
            "query": "test",
            "collection": "docs",
            "top_k": 10,
            "rerank": False,
        })

        mock_kb.search.assert_called_once_with(
            query="test",
            collection="docs",
            top_k=10,
            rerank=False,
        )

    def test_search_defaults(self, server, mock_kb):
        """Search uses default top_k=5 and server's use_rerank setting."""
        call_tool(server, "kb_search", {
            "query": "test",
            "collection": "docs",
        })

        mock_kb.search.assert_called_once_with(
            query="test",
            collection="docs",
            top_k=5,
            rerank=None,  # agent didn't specify → kb.default_rerank resolves it
        )

    def test_search_nonexistent_collection(self, server, mock_kb):
        """Searching a missing collection returns an error."""
        mock_kb.search.side_effect = ValueError("Collection 'missing' not found")

        result = call_tool(server, "kb_search", {
            "query": "test",
            "collection": "missing",
        })

        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_search_rerank_score_forwarded(self, server, mock_kb):
        """Rerank scores from the KB are included in the response."""
        mock_kb.search.return_value = [
            {**make_search_result("text", "file.md"), "rerank_score": 0.99},
        ]

        result = call_tool(server, "kb_search", {
            "query": "test",
            "collection": "docs",
        })

        assert result["results"][0]["rerank_score"] == 0.99


# ---------------------------------------------------------------------------
# kb_ingest (background job pattern)
# ---------------------------------------------------------------------------

class TestIngest:

    def test_ingest_returns_started(self, server, mock_kb, tmp_path):
        """Ingesting a folder returns immediately with a job_id."""
        folder = tmp_path / "new_docs"
        folder.mkdir()

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "started"
        assert "job_id" in result
        assert result["collection"] == "new_docs"

    def test_ingest_job_completes(self, server, mock_kb, tmp_path):
        """Background ingest job completes successfully."""
        folder = tmp_path / "new_docs"
        folder.mkdir()

        async def run():
            initial, final = await call_tool_and_await_job(
                server, "kb_ingest", {"folder_path": str(folder)}
            )
            assert final["status"] == "complete"
            assert final["result"]["files"] == 5
            assert final["result"]["chunks"] == 42

        asyncio.run(run())

    def test_ingest_with_file_types(self, server, mock_kb, tmp_path):
        """File types are forwarded to kb.ingest."""
        folder = tmp_path / "docs2"
        folder.mkdir()

        async def run():
            await call_tool_and_await_job(
                server, "kb_ingest", {
                    "folder_path": str(folder),
                    "file_types": ["md", "txt"],
                }
            )
            # New server always passes collection_name to kb.ingest
            mock_kb.ingest.assert_called_once_with(
                folder, collection_name="docs2", file_types=["md", "txt"],
            )

        asyncio.run(run())

    def test_ingest_folder_not_found(self, server):
        """Ingesting a nonexistent folder returns an error immediately."""
        result = call_tool(server, "kb_ingest", {
            "folder_path": "/nonexistent/folder",
        })

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_ingest_not_a_directory(self, server, tmp_path):
        """Ingesting a file (not a directory) returns an error immediately."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("I'm a file")

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(file_path),
        })

        assert result["status"] == "error"
        assert "Not a directory" in result["error"]

    def test_ingest_unexpected_error(self, server, mock_kb, tmp_path):
        """Unexpected errors during ingestion are reported via job status."""
        folder = tmp_path / "bad_docs"
        folder.mkdir()
        mock_kb.ingest.side_effect = RuntimeError("disk full")

        async def run():
            initial, final = await call_tool_and_await_job(
                server, "kb_ingest", {"folder_path": str(folder)}
            )
            assert final["status"] == "error"
            assert "disk full" in final["error"]

        asyncio.run(run())

    def test_ingest_deconflicts_existing_collection(self, server, mock_kb, tmp_path):
        """Ingesting into an existing collection name creates a numbered variant."""
        folder = tmp_path / "docs"
        folder.mkdir()

        # Simulate "docs" already exists with a FAISS index from a different source.
        # _collection_exists() requires a marker file, and deconflict only triggers
        # when source.txt records a different folder than the one being ingested.
        existing = mock_kb.index_dir / "docs"
        existing.mkdir()
        (existing / "index.faiss").write_bytes(b"fake")
        (existing / "source.txt").write_text("/some/other/folder")

        mock_kb.ingest.return_value = {"collection": "docs1", "files": 3, "chunks": 10}

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "started"
        assert result["collection"] == "docs1"
        assert "warning" in result
        assert "docs1" in result["warning"]

    def test_ingest_deconflicts_symlinked_collection(self, server, mock_kb, tmp_path):
        """Ingesting when collection is a symlink deconflicts without corrupting base."""
        folder = tmp_path / "docs"
        folder.mkdir()

        # Simulate "docs" is a symlink to a base collection with index
        base_dir = tmp_path / "base_docs"
        base_dir.mkdir()
        (base_dir / "index.faiss").write_bytes(b"fake")
        (base_dir / "source.txt").write_text("/some/other/folder")

        (mock_kb.index_dir / "docs").symlink_to(base_dir)
        mock_kb.ingest.return_value = {"collection": "docs1", "files": 3, "chunks": 10}

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "started"
        assert result["collection"] == "docs1"
        assert "warning" in result
        assert "docs1" in result["warning"]
        # Base symlink should still exist untouched
        assert (mock_kb.index_dir / "docs").is_symlink()


# ---------------------------------------------------------------------------
# kb_job_status
# ---------------------------------------------------------------------------

class TestJobStatus:

    def test_unknown_job(self, server):
        """Polling an unknown job_id returns an error."""
        result = call_tool(server, "kb_job_status", {"job_id": "nonexistent"})

        assert result["status"] == "error"
        assert "Unknown job" in result["error"]

    def test_running_job(self, server, mock_kb, tmp_path):
        """A job that hasn't completed reports running status."""
        folder = tmp_path / "slow_docs"
        folder.mkdir()

        # Make ingest block so the job stays in "running"
        def blocking_ingest(*args, **kwargs):
            time.sleep(10)
            return {"collection": "slow_docs", "files": 1, "chunks": 5}
        mock_kb.ingest.side_effect = blocking_ingest

        async def run():
            initial = await _call_tool_async(server, "kb_ingest", {
                "folder_path": str(folder),
            })
            assert initial["status"] == "started"
            job_id = initial["job_id"]

            # Immediately check — should still be running
            status = await _call_tool_async(server, "kb_job_status", {"job_id": job_id})
            assert status["status"] == "running"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# kb_append (background job pattern)
# ---------------------------------------------------------------------------

class TestAppend:

    def test_append_returns_started(self, server, mock_kb, tmp_path):
        """Appending to a collection returns immediately with a job_id."""
        # Create a fake existing collection
        coll_dir = mock_kb.index_dir / "docs"
        coll_dir.mkdir(exist_ok=True)
        (coll_dir / "index.faiss").write_text("fake")

        result = call_tool(server, "kb_append", {
            "collection": "docs",
            "paths": [str(tmp_path)],
        })

        assert result["status"] == "started"
        assert "job_id" in result
        assert result["collection"] == "docs"

    def test_append_job_completes(self, server, mock_kb, tmp_path):
        """Background append job completes successfully."""
        coll_dir = mock_kb.index_dir / "docs"
        coll_dir.mkdir(exist_ok=True)
        (coll_dir / "index.faiss").write_text("fake")

        async def run():
            initial, final = await call_tool_and_await_job(
                server, "kb_append", {
                    "collection": "docs",
                    "paths": [str(tmp_path)],
                }
            )
            assert final["status"] == "complete"
            assert final["result"]["chunks_added"] == 10

        asyncio.run(run())

    def test_append_collection_not_found(self, server, mock_kb):
        """Appending to a nonexistent collection returns an error immediately."""
        result = call_tool(server, "kb_append", {
            "collection": "nonexistent",
            "paths": ["/some/path"],
        })

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# kb_search — error handling (transport-closed diagnostics)
# ---------------------------------------------------------------------------

class TestSearchErrorHandling:
    """Verify the server returns error responses (not crashes) for common
    failure modes that would otherwise cause 'transport closed'."""

    def test_search_httpx_connect_error(self, mock_kb):
        """Network unreachable during search returns error, not crash."""
        import httpx
        mock_kb.search.side_effect = httpx.ConnectError("Connection refused")
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_search", {
            "query": "test", "collection": "docs",
        })

        assert result["status"] == "error"
        assert "Connection refused" in result["error"]

    def test_search_httpx_timeout(self, mock_kb):
        """Embedding API timeout during search returns error, not crash."""
        import httpx
        mock_kb.search.side_effect = httpx.ReadTimeout("Read timed out")
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_search", {
            "query": "test", "collection": "docs",
        })

        assert result["status"] == "error"
        assert "timed out" in result["error"].lower()

    def test_search_httpx_401(self, mock_kb):
        """Expired/invalid API key during search returns error, not crash."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_kb.search.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_resp,
        )
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_search", {
            "query": "test", "collection": "docs",
        })

        assert result["status"] == "error"
        assert "401" in result["error"]

    def test_search_httpx_500(self, mock_kb):
        """Embedding API server error returns error, not crash."""
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_kb.search.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=MagicMock(), response=mock_resp,
        )
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_search", {
            "query": "test", "collection": "docs",
        })

        assert result["status"] == "error"
        assert "500" in result["error"]

    def test_search_runtime_error(self, mock_kb):
        """Unexpected RuntimeError during search returns error, not crash."""
        mock_kb.search.side_effect = RuntimeError("FAISS segfault simulation")
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_search", {
            "query": "test", "collection": "docs",
        })

        assert result["status"] == "error"
        assert "FAISS segfault" in result["error"]

    def test_search_os_error(self, mock_kb):
        """OS-level error (disk, permissions) returns error, not crash."""
        mock_kb.search.side_effect = OSError("Permission denied: index.faiss")
        server = create_knowledge_server(mock_kb)

        result = call_tool(server, "kb_search", {
            "query": "test", "collection": "docs",
        })

        assert result["status"] == "error"
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# setup_runtime_kb
# ---------------------------------------------------------------------------

class TestSetupRuntimeKb:

    def test_symlinks_collections(self, tmp_path):
        """Symlinks collection directories from base to runtime."""
        base = tmp_path / "base_index"
        coll_dir = base / "my_collection"
        coll_dir.mkdir(parents=True)
        (coll_dir / "index.faiss").write_text("fake index")
        (coll_dir / "chunks.jsonl").write_text('{"id": "1"}\n')
        (coll_dir / "DESCRIPTION.md").write_text("Test collection")

        runtime = tmp_path / "runtime"
        result = setup_runtime_kb(base, runtime)

        assert result == runtime / "kb_index"
        link = result / "my_collection"
        assert link.exists()
        assert link.is_symlink()
        assert (link / "index.faiss").exists()
        assert (link / "chunks.jsonl").exists()
        assert (link / "DESCRIPTION.md").exists()

    def test_skips_non_collection_dirs(self, tmp_path):
        """Directories without index.faiss are not symlinked."""
        base = tmp_path / "base_index"
        (base / "random_dir").mkdir(parents=True)
        (base / "random_dir" / "notes.txt").write_text("not a collection")

        runtime = tmp_path / "runtime"
        result = setup_runtime_kb(base, runtime)

        assert not (result / "random_dir").exists()

    def test_nonexistent_base_dir(self, tmp_path):
        """Non-existent base directory creates empty runtime."""
        runtime = tmp_path / "runtime"
        result = setup_runtime_kb(tmp_path / "missing", runtime)

        assert result.exists()
        assert list(result.iterdir()) == []

    def test_does_not_overwrite_existing(self, tmp_path):
        """Existing runtime collections are not overwritten."""
        base = tmp_path / "base_index"
        coll = base / "docs"
        coll.mkdir(parents=True)
        (coll / "index.faiss").write_text("base version")

        runtime = tmp_path / "runtime"
        runtime_coll = runtime / "kb_index" / "docs"
        runtime_coll.mkdir(parents=True)
        (runtime_coll / "index.faiss").write_text("runtime version")

        setup_runtime_kb(base, runtime)

        assert (runtime_coll / "index.faiss").read_text() == "runtime version"


# ---------------------------------------------------------------------------
# Regression: OpenMP duplicate library crash (transport closed)
# ---------------------------------------------------------------------------

class TestOpenMPWorkaround:
    """Importing knowledge_server must set KMP_DUPLICATE_LIB_OK to prevent
    a fatal OpenMP crash when FAISS and sentence-transformers (PyTorch)
    both bundle libomp.

    Without this, kb_search with rerank=true kills
    the server process, producing 'transport closed' in MCP clients."""

    def test_kmp_duplicate_lib_ok_is_set(self):
        """KMP_DUPLICATE_LIB_OK is set after importing the knowledge server."""
        import os
        import dsagt.commands.knowledge_server  # noqa: F401

        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


# ---------------------------------------------------------------------------
# Regression: rerank schema default must match server config
# ---------------------------------------------------------------------------

class TestRerankSchemaDefault:
    """The kb_search schema previously hardcoded 'default': True for the
    rerank parameter, causing agents to request reranking even when the
    server wasn't started with --rerank. This triggered the OpenMP crash."""

    def _get_rerank_default(self, server):
        """Extract the rerank default from the kb_search tool schema."""
        req = types.ListToolsRequest(method="tools/list")
        handler = server.request_handlers[types.ListToolsRequest]
        result = asyncio.run(handler(req))
        for tool in result.root.tools:
            if tool.name == "kb_search":
                return tool.inputSchema["properties"]["rerank"]["default"]
        raise AssertionError("kb_search tool not found")

    def test_rerank_default_from_kb(self, mock_kb):
        """Schema advertises rerank default matching kb.default_rerank."""
        mock_kb.default_rerank = False
        server = create_knowledge_server(mock_kb)
        assert self._get_rerank_default(server) is False

        mock_kb.default_rerank = True
        server = create_knowledge_server(mock_kb)
        assert self._get_rerank_default(server) is True

    def test_search_omitted_rerank_passes_none(self, mock_kb):
        """Omitting rerank passes None to kb.search, which resolves to
        kb.default_rerank internally."""
        server = create_knowledge_server(mock_kb)
        call_tool(server, "kb_search", {
            "query": "test",
            "collection": "docs",
        })
        mock_kb.search.assert_called_once_with(
            query="test",
            collection="docs",
            top_k=5,
            rerank=None,
        )
