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
import threading
from unittest.mock import MagicMock

import pytest
import mcp.types as types

from dsagt.mcp.knowledge_tools import create_knowledge_server, setup_runtime_kb
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


async def call_tool_and_await_job(
    server, name: str, arguments: dict
) -> tuple[dict, dict]:
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


def make_search_result(
    text: str, source_file: str, chunk_index: int = 0, score: float = 0.9
):
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
    kb.append.return_value = {
        "collection": "docs",
        "files": 2,
        "chunks_added": 10,
        "total_chunks": 50,
    }
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
        result = call_tool(
            server,
            "kb_search",
            {
                "query": "how to install",
                "collection": "docs",
            },
        )

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
        call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
                "top_k": 10,
                "rerank": False,
            },
        )

        mock_kb.search.assert_called_once_with(
            query="test",
            collection="docs",
            collections=None,
            top_k=10,
            rerank=False,
            where=None,
            where_document=None,
        )

    def test_search_defaults(self, server, mock_kb):
        """Search uses default top_k=5 and server's use_rerank setting."""
        call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        mock_kb.search.assert_called_once_with(
            query="test",
            collection="docs",
            collections=None,
            top_k=5,
            rerank=None,  # agent didn't specify → kb.default_rerank resolves it
            where=None,
            where_document=None,
        )

    def test_search_nonexistent_collection(self, server, mock_kb):
        """Searching a missing collection returns an error."""
        mock_kb.search.side_effect = ValueError("Collection 'missing' not found")

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "missing",
            },
        )

        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_search_rerank_score_forwarded(self, server, mock_kb):
        """Rerank scores from the KB are included in the response."""
        mock_kb.search.return_value = [
            {**make_search_result("text", "file.md"), "rerank_score": 0.99},
        ]

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["results"][0]["rerank_score"] == 0.99


# ---------------------------------------------------------------------------
# kb_ingest (background job pattern)
# ---------------------------------------------------------------------------


class TestIngest:

    def test_ingest_returns_started(self, server, mock_kb, tmp_path):
        """Ingesting a folder returns immediately with a job_id."""
        folder = tmp_path / "new_docs"
        folder.mkdir()

        result = call_tool(
            server,
            "kb_ingest",
            {
                "folder_path": str(folder),
            },
        )

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
                server,
                "kb_ingest",
                {
                    "folder_path": str(folder),
                    "file_types": ["md", "txt"],
                },
            )
            # New server always passes collection_name to kb.ingest
            mock_kb.ingest.assert_called_once_with(
                folder,
                collection_name="docs2",
                file_types=["md", "txt"],
            )

        asyncio.run(run())

    def test_ingest_folder_not_found(self, server):
        """Ingesting a nonexistent folder returns an error immediately."""
        result = call_tool(
            server,
            "kb_ingest",
            {
                "folder_path": "/nonexistent/folder",
            },
        )

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_ingest_not_a_directory(self, server, tmp_path):
        """Ingesting a file (not a directory) returns an error immediately."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("I'm a file")

        result = call_tool(
            server,
            "kb_ingest",
            {
                "folder_path": str(file_path),
            },
        )

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

        # Simulate "docs" already exists with a Chroma index from a different source.
        # _collection_exists() requires a marker file, and deconflict only triggers
        # when source.txt records a different folder than the one being ingested.
        existing = mock_kb.index_dir / "docs"
        existing.mkdir()
        (existing / "chroma_ids.json").write_bytes(b"fake")
        (existing / "source.txt").write_text("/some/other/folder")

        mock_kb.ingest.return_value = {"collection": "docs1", "files": 3, "chunks": 10}

        result = call_tool(
            server,
            "kb_ingest",
            {
                "folder_path": str(folder),
            },
        )

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
        (base_dir / "chroma_ids.json").write_bytes(b"fake")
        (base_dir / "source.txt").write_text("/some/other/folder")

        (mock_kb.index_dir / "docs").symlink_to(base_dir)
        mock_kb.ingest.return_value = {"collection": "docs1", "files": 3, "chunks": 10}

        result = call_tool(
            server,
            "kb_ingest",
            {
                "folder_path": str(folder),
            },
        )

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

        # Hold the job in "running" until the test has observed it.
        # asyncio.run joins the worker thread at loop shutdown, so the
        # event must be set before ``run()`` returns; the wait timeout
        # only bounds a failing test.
        release = threading.Event()

        def blocking_ingest(*args, **kwargs):
            release.wait(timeout=10)
            return {"collection": "slow_docs", "files": 1, "chunks": 5}

        mock_kb.ingest.side_effect = blocking_ingest

        async def run():
            try:
                initial = await _call_tool_async(
                    server,
                    "kb_ingest",
                    {
                        "folder_path": str(folder),
                    },
                )
                assert initial["status"] == "started"
                job_id = initial["job_id"]

                # Immediately check — should still be running
                status = await _call_tool_async(
                    server, "kb_job_status", {"job_id": job_id}
                )
                assert status["status"] == "running"
            finally:
                release.set()

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
        (coll_dir / "chroma_ids.json").write_text("fake")

        result = call_tool(
            server,
            "kb_append",
            {
                "collection": "docs",
                "paths": [str(tmp_path)],
            },
        )

        assert result["status"] == "started"
        assert "job_id" in result
        assert result["collection"] == "docs"

    def test_append_job_completes(self, server, mock_kb, tmp_path):
        """Background append job completes successfully."""
        coll_dir = mock_kb.index_dir / "docs"
        coll_dir.mkdir(exist_ok=True)
        (coll_dir / "chroma_ids.json").write_text("fake")

        async def run():
            initial, final = await call_tool_and_await_job(
                server,
                "kb_append",
                {
                    "collection": "docs",
                    "paths": [str(tmp_path)],
                },
            )
            assert final["status"] == "complete"
            assert final["result"]["chunks_added"] == 10

        asyncio.run(run())

    def test_append_collection_not_found(self, server, mock_kb):
        """Appending to a nonexistent collection returns an error immediately."""
        result = call_tool(
            server,
            "kb_append",
            {
                "collection": "nonexistent",
                "paths": ["/some/path"],
            },
        )

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

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["status"] == "error"
        assert "Connection refused" in result["error"]

    def test_search_httpx_timeout(self, mock_kb):
        """Embedding API timeout during search returns error, not crash."""
        import httpx

        mock_kb.search.side_effect = httpx.ReadTimeout("Read timed out")
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["status"] == "error"
        assert "timed out" in result["error"].lower()

    def test_search_httpx_401(self, mock_kb):
        """Expired/invalid API key during search returns error, not crash."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_kb.search.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_resp,
        )
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["status"] == "error"
        assert "401" in result["error"]

    def test_search_httpx_500(self, mock_kb):
        """Embedding API server error returns error, not crash."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_kb.search.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=mock_resp,
        )
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["status"] == "error"
        assert "500" in result["error"]

    def test_search_runtime_error(self, mock_kb):
        """Unexpected RuntimeError during search returns error, not crash."""
        mock_kb.search.side_effect = RuntimeError("index segfault simulation")
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["status"] == "error"
        assert "index segfault" in result["error"]

    def test_search_os_error(self, mock_kb):
        """OS-level error (disk, permissions) returns error, not crash."""
        mock_kb.search.side_effect = OSError("Permission denied: chroma.sqlite3")
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["status"] == "error"
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# setup_runtime_kb
# ---------------------------------------------------------------------------


class TestSetupRuntimeKb:

    def test_copies_collections(self, tmp_path):
        """Copies (not symlinks) collection directories from base to runtime.

        Copy semantics pin each project to whatever bundled content was
        current at first start — different projects on the same machine
        may run different dsagt versions, and a symlink would let one
        project's ``setup-kb --rebuild`` mutate every project's view.
        """
        base = tmp_path / "base_index"
        coll_dir = base / "my_collection"
        coll_dir.mkdir(parents=True)
        (coll_dir / "chroma_ids.json").write_text("fake index")
        (coll_dir / "chunks.jsonl").write_text('{"id": "1"}\n')
        (coll_dir / "DESCRIPTION.md").write_text("Test collection")

        runtime = tmp_path / "runtime"
        result = setup_runtime_kb(base, runtime)

        assert result == runtime / "kb_index"
        copied = result / "my_collection"
        assert copied.exists()
        assert not copied.is_symlink()  # copy not symlink
        assert (copied / "chroma_ids.json").exists()
        assert (copied / "chroma_ids.json").read_text() == "fake index"
        assert (copied / "chunks.jsonl").exists()
        assert (copied / "DESCRIPTION.md").exists()

    def test_copy_is_independent(self, tmp_path):
        """Mutating the base after copy does not affect the project copy."""
        base = tmp_path / "base_index"
        coll = base / "codes"
        coll.mkdir(parents=True)
        (coll / "chroma_ids.json").write_text("v1")

        runtime = tmp_path / "runtime"
        setup_runtime_kb(base, runtime)

        # Mutate the base — simulating ``dsagt setup-kb --rebuild``.
        (coll / "chroma_ids.json").write_text("v2 newer")

        # Project copy stays at v1.
        project_copy = runtime / "kb_index" / "codes" / "chroma_ids.json"
        assert project_copy.read_text() == "v1"

    def test_skips_non_collection_dirs(self, tmp_path):
        """Directories without chroma_ids.json are not copied."""
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
        (coll / "chroma_ids.json").write_text("base version")

        runtime = tmp_path / "runtime"
        runtime_coll = runtime / "kb_index" / "docs"
        runtime_coll.mkdir(parents=True)
        (runtime_coll / "chroma_ids.json").write_text("runtime version")

        setup_runtime_kb(base, runtime)

        assert (runtime_coll / "chroma_ids.json").read_text() == "runtime version"


# ---------------------------------------------------------------------------
# Regression: OpenMP duplicate library crash (transport closed)
# ---------------------------------------------------------------------------


class TestOpenMPWorkaround:
    """Importing the knowledge tools module must set KMP_DUPLICATE_LIB_OK to
    prevent a fatal OpenMP crash when multiple native deps (e.g. ChromaDB and
    sentence-transformers / PyTorch) both bundle libomp.

    Without this, kb_search with rerank=true kills
    the server process, producing 'transport closed' in MCP clients."""

    def test_kmp_duplicate_lib_ok_is_set(self):
        """KMP_DUPLICATE_LIB_OK is set after importing dsagt.mcp.knowledge_tools."""
        import os
        import dsagt.mcp.knowledge_tools  # noqa: F401

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


# ---------------------------------------------------------------------------
# kb_search — multi-collection fan-out (moved from the former memory test file)
# ---------------------------------------------------------------------------


class TestKbSearchMultiCollection:

    def test_multi_collection_fanout(self, server, mock_kb):
        """Multi-collection search delegates once to kb.search with collections=.

        Fan-out + fusion across collections is kb.search's job (covered by
        TestFederatedSearch in test_knowledge_base.py); the handler just forwards.
        """
        mock_kb.search.return_value = [
            make_search_result("result", "/file.md", 0, 0.9),
        ]

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collections": ["docs", "papers"],
            },
        )

        assert result["status"] == "ok"
        mock_kb.search.assert_called_once_with(
            query="test",
            collection=None,
            collections=["docs", "papers"],
            top_k=5,
            rerank=None,
            where=None,
            where_document=None,
        )

    def test_no_collection_returns_error(self, server):
        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
            },
        )

        assert result["status"] == "error"

    def test_multi_collection_merges_results(self, server, mock_kb):
        """The handler returns kb.search's already-fused, sorted results."""
        # kb.search owns fusion now; it returns one merged, descending list.
        mock_kb.search.return_value = [
            make_search_result("result_1", "/file_1.md", score=0.9),
            make_search_result("result_2", "/file_2.md", score=0.7),
        ]

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collections": ["docs", "papers"],
                "top_k": 5,
            },
        )

        assert result["result_count"] == 2
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)


class TestKbSearchSchema:

    def _get_tool(self, server, name):
        req = types.ListToolsRequest(method="tools/list")
        handler = server.request_handlers[types.ListToolsRequest]
        result = asyncio.run(handler(req))
        for tool in result.root.tools:
            if tool.name == name:
                return tool
        return None

    def test_kb_search_has_collections_param(self, server):
        tool = self._get_tool(server, "kb_search")
        assert "collections" in tool.inputSchema["properties"]

    def test_kb_search_query_is_only_required(self, server):
        tool = self._get_tool(server, "kb_search")
        assert tool.inputSchema["required"] == ["query"]
