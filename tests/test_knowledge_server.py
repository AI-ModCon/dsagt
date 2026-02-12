"""
Tests for the knowledge base MCP server.

Tests the tool handlers and setup_runtime_kb utility.
The KnowledgeBase is mocked — these tests verify the server's
handler logic, argument parsing, error handling, and response formatting.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import mcp.types as types

from dsagt.knowledge_server import create_knowledge_server, setup_runtime_kb


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
    kb.list_collections.return_value = [
        {"name": "docs", "description": "Project documentation"},
        {"name": "papers", "description": "Research papers"},
    ]
    kb.search.return_value = [
        make_search_result("First result text", "/path/to/file1.md", 0, 0.95),
        make_search_result("Second result text", "/path/to/file2.md", 1, 0.80),
    ]
    kb.ingest.return_value = {"collection": "new_docs", "files": 5, "chunks": 42}
    return kb


@pytest.fixture
def server(mock_kb):
    """Knowledge server with mocked KB and reranking enabled."""
    return create_knowledge_server(mock_kb, use_rerank=True)


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
        server = create_knowledge_server(mock_kb, use_rerank=True)

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
            rerank=True,  # server was created with use_rerank=True
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
# kb_ingest
# ---------------------------------------------------------------------------

class TestIngest:

    def test_ingest_success(self, server, mock_kb, tmp_path):
        """Successful ingestion returns collection stats."""
        folder = tmp_path / "new_docs"
        folder.mkdir()

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "ok"
        assert result["collection"] == "new_docs"
        assert result["files_indexed"] == 5
        assert result["chunks_created"] == 42

    def test_ingest_with_file_types(self, server, mock_kb, tmp_path):
        """File types are forwarded to kb.ingest."""
        folder = tmp_path / "docs"
        folder.mkdir()

        call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
            "file_types": ["md", "txt"],
        })

        mock_kb.ingest.assert_called_once_with(
            folder, file_types=["md", "txt"],
        )

    def test_ingest_folder_not_found(self, server):
        """Ingesting a nonexistent folder returns an error."""
        result = call_tool(server, "kb_ingest", {
            "folder_path": "/nonexistent/folder",
        })

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_ingest_not_a_directory(self, server, tmp_path):
        """Ingesting a file (not a directory) returns an error."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("I'm a file")

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(file_path),
        })

        assert result["status"] == "error"
        assert "Not a directory" in result["error"]

    def test_ingest_unexpected_error(self, server, mock_kb, tmp_path):
        """Unexpected errors during ingestion are caught and reported."""
        folder = tmp_path / "bad_docs"
        folder.mkdir()
        mock_kb.ingest.side_effect = RuntimeError("disk full")

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "error"
        assert "disk full" in result["error"]

    def test_ingest_deconflicts_existing_collection(self, server, mock_kb, tmp_path):
        """Ingesting into an existing collection name creates a numbered variant."""
        folder = tmp_path / "docs"
        folder.mkdir()

        # Simulate "docs" already exists in the index dir
        (mock_kb.index_dir / "docs").mkdir()
        mock_kb.ingest.return_value = {"collection": "docs1", "files": 3, "chunks": 10}

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "ok"
        assert result["collection"] == "docs1"
        assert result["warning"] is not None
        assert "docs1" in result["warning"]
        # collection_name should be passed when deconflicted
        mock_kb.ingest.assert_called_once_with(folder, collection_name="docs1")

    def test_ingest_deconflicts_symlinked_collection(self, server, mock_kb, tmp_path):
        """Ingesting when collection is a symlink deconflicts without corrupting base."""
        folder = tmp_path / "docs"
        folder.mkdir()

        # Simulate "docs" is a symlink to a base collection
        base_dir = tmp_path / "base_docs"
        base_dir.mkdir()
        (mock_kb.index_dir / "docs").symlink_to(base_dir)
        mock_kb.ingest.return_value = {"collection": "docs1", "files": 3, "chunks": 10}

        result = call_tool(server, "kb_ingest", {
            "folder_path": str(folder),
        })

        assert result["status"] == "ok"
        assert "docs1" in result["warning"]
        # Base symlink should still exist untouched
        assert (mock_kb.index_dir / "docs").is_symlink()


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

class TestUnknownTool:

    def test_unknown_tool(self, server):
        """Calling an unregistered tool returns an error."""
        result = call_tool(server, "nonexistent_tool", {})

        assert result["status"] == "error"
        assert "Unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# setup_runtime_kb
# ---------------------------------------------------------------------------

class TestSetupRuntimeKb:

    def test_copies_collections(self, tmp_path):
        """Copies collection directories from base to runtime."""
        base = tmp_path / "base_index"
        coll_dir = base / "my_collection"
        coll_dir.mkdir(parents=True)
        (coll_dir / "index.faiss").write_text("fake index")
        (coll_dir / "chunks.jsonl").write_text('{"id": "1"}\n')
        (coll_dir / "DESCRIPTION.md").write_text("Test collection")

        runtime = tmp_path / "runtime"
        result = setup_runtime_kb(base, runtime)

        assert result == runtime / "kb_index"
        assert (result / "my_collection" / "index.faiss").exists()
        assert (result / "my_collection" / "chunks.jsonl").exists()
        assert (result / "my_collection" / "DESCRIPTION.md").exists()

    def test_skips_non_collection_dirs(self, tmp_path):
        """Directories without index.faiss are not copied."""
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
