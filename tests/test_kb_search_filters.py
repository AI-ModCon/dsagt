"""
Tests for kb_search metadata filtering.

Tests the _build_where_clause helper and the handler's threading of
filter arguments through to kb.search(where=...).
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from mcp_helpers import call_tool_json as call_tool

from dsagt.mcp.knowledge_tools import create_knowledge_server


def make_search_result(text, source_file, chunk_index=0, score=0.9, extra_meta=None):
    meta = {
        "source_file": source_file,
        "collection": "test",
        "chunk_index": chunk_index,
        "file_type": ".md",
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"chunk": {"text": text, "metadata": meta}, "score": score}


@pytest.fixture
def mock_kb(tmp_path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.search.return_value = [
        make_search_result("result text", "/file.md"),
    ]
    return kb


@pytest.fixture
def server(mock_kb):
    return create_knowledge_server(mock_kb)


# ---------------------------------------------------------------------------
# Handler: filter threading
# ---------------------------------------------------------------------------


class TestSearchFilterThreading:

    def test_tool_name_filter_passed(self, server, mock_kb):
        """tool_name filter is threaded through as where clause."""
        call_tool(
            server,
            "kb_search",
            {
                "query": "quality filtering",
                "collection": "tool_executions",
                "tool_name": "fastp",
            },
        )

        mock_kb.search.assert_called_once_with(
            query="quality filtering",
            collection="tool_executions",
            collections=None,
            top_k=5,
            rerank=None,
            where={"tool_name": "fastp"},
            where_document=None,
        )

    def test_session_filter_passed(self, server, mock_kb):
        """session_id filter is threaded through."""
        call_tool(
            server,
            "kb_search",
            {
                "query": "pipeline",
                "collection": "tool_executions",
                "session_id": "s3",
            },
        )

        mock_kb.search.assert_called_once_with(
            query="pipeline",
            collection="tool_executions",
            collections=None,
            top_k=5,
            rerank=None,
            where={"session_id": "s3"},
            where_document=None,
        )

    def test_multiple_filters_combined(self, server, mock_kb):
        """Multiple filters produce a compound $and where clause."""
        call_tool(
            server,
            "kb_search",
            {
                "query": "parameters",
                "collection": "tool_executions",
                "tool_name": "fastp",
                "session_id": "s1",
            },
        )

        # The handler emits one single-key dict per filter, in the fixed
        # source key order (category, session_id, source_type, tool_name) —
        # so session_id precedes tool_name. Pin the exact payload: a dropped
        # or duplicated filter must fail here.
        where = mock_kb.search.call_args[1]["where"]
        assert where == {"$and": [{"session_id": "s1"}, {"tool_name": "fastp"}]}

    def test_return_code_filter_passed(self, server, mock_kb):
        """return_code filter is threaded as integer."""
        call_tool(
            server,
            "kb_search",
            {
                "query": "failures",
                "collection": "tool_executions",
                "return_code": 1,
            },
        )

        mock_kb.search.assert_called_once_with(
            query="failures",
            collection="tool_executions",
            collections=None,
            top_k=5,
            rerank=None,
            where={"return_code": 1},
            where_document=None,
        )

    def test_filters_with_multi_collection(self, server, mock_kb):
        """Multi-collection search delegates to kb.search once with collections=."""
        mock_kb.search.return_value = [
            make_search_result("result", "/f.md"),
        ]

        call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collections": ["tool_executions", "episodic_memory"],
                "tool_name": "fastp",
            },
        )

        # Fan-out lives in kb.search now; the handler makes a single call.
        mock_kb.search.assert_called_once_with(
            query="test",
            collection=None,
            collections=["tool_executions", "episodic_memory"],
            top_k=5,
            rerank=None,
            where={"tool_name": "fastp"},
            where_document=None,
        )


class TestDocumentFilterThreading:
    """The 'regex' / 'contains' args build a where_document passed to kb.search."""

    def test_regex_threaded_as_where_document(self, server, mock_kb):
        call_tool(
            server,
            "kb_search",
            {"query": "q", "collection": "session_memory", "regex": "(?i)parser"},
        )
        assert mock_kb.search.call_args[1]["where_document"] == {"$regex": "(?i)parser"}

    def test_contains_threaded_as_where_document(self, server, mock_kb):
        call_tool(
            server,
            "kb_search",
            {"query": "q", "collection": "session_memory", "contains": "Q30"},
        )
        assert mock_kb.search.call_args[1]["where_document"] == {"$contains": "Q30"}

    def test_regex_and_contains_combined_with_and(self, server, mock_kb):
        call_tool(
            server,
            "kb_search",
            {
                "query": "q",
                "collection": "session_memory",
                "regex": "fastp",
                "contains": "Q30",
            },
        )
        assert mock_kb.search.call_args[1]["where_document"] == {
            "$and": [{"$regex": "fastp"}, {"$contains": "Q30"}]
        }

    def test_no_document_filter_is_none(self, server, mock_kb):
        call_tool(server, "kb_search", {"query": "q", "collection": "session_memory"})
        assert mock_kb.search.call_args[1]["where_document"] is None


# ---------------------------------------------------------------------------
# Handler: metadata in results
# ---------------------------------------------------------------------------


class TestSearchResultMetadata:

    def test_metadata_included_in_results(self, mock_kb):
        """Search results include extra metadata fields."""
        mock_kb.search.return_value = [
            make_search_result(
                "fastp ran",
                "/file.md",
                extra_meta={"tool_name": "fastp", "session_id": "s1", "return_code": 0},
            ),
        ]
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "tool_executions",
            },
        )

        hit = result["results"][0]
        assert hit["metadata"]["tool_name"] == "fastp"
        assert hit["metadata"]["session_id"] == "s1"
        assert hit["metadata"]["return_code"] == 0

    def test_metadata_excludes_standard_fields(self, mock_kb):
        """Standard fields (source_file, chunk_index, collection, file_type) are
        not duplicated in the metadata dict."""
        mock_kb.search.return_value = [
            make_search_result("text", "/file.md"),
        ]
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        meta = result["results"][0]["metadata"]
        assert "source_file" not in meta
        assert "chunk_index" not in meta
        assert "collection" not in meta
        assert "file_type" not in meta

    def test_empty_metadata_for_reference_collections(self, mock_kb):
        """Reference collections (no extra metadata) produce empty metadata dict."""
        mock_kb.search.return_value = [
            make_search_result("text", "/file.md"),
        ]
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collection": "docs",
            },
        )

        assert result["results"][0]["metadata"] == {}


# ---------------------------------------------------------------------------
# Schema: filter params visible to agent
# ---------------------------------------------------------------------------


class TestSearchSchemaFilters:

    def _get_kb_search_schema(self, server):
        handler = server.get_request_handler("tools/list").handler
        result = asyncio.run(handler(None, None))
        for tool in result.tools:
            if tool.name == "kb_search":
                return tool.input_schema
        raise AssertionError("kb_search not found")

    def test_filter_params_in_schema(self, server):
        """All filter parameters are advertised in the kb_search schema."""
        schema = self._get_kb_search_schema(server)
        props = schema["properties"]
        for param in (
            "category",
            "session_id",
            "tool_name",
            "source_type",
            "return_code",
            "regex",
            "contains",
        ):
            assert param in props, f"Missing filter param: {param}"

    def test_filter_params_not_required(self, server):
        """Filter params are optional — only query is required."""
        schema = self._get_kb_search_schema(server)
        assert schema["required"] == ["query"]


# ---------------------------------------------------------------------------
# Handler: error behavior for missing collections
# ---------------------------------------------------------------------------


class TestSearchCollectionErrors:

    def test_single_missing_collection_returns_error(self, mock_kb):
        """Searching a single nonexistent collection returns error."""
        mock_kb.search.side_effect = ValueError("Collection 'missing' not found")
        server = create_knowledge_server(mock_kb)

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

    def test_all_collections_missing_passes_through_error(self, mock_kb):
        """kb.search owns the all-failed aggregation; the handler passes it through."""
        mock_kb.search.side_effect = ValueError(
            "All collections failed: Collection 'missing1' not found"
        )
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collections": ["missing1", "missing2"],
            },
        )

        assert result["status"] == "error"
        assert "All collections failed" in result["error"]

    def test_partial_multi_collection_returns_ok(self, mock_kb):
        """Partial-skip happens inside kb.search; the handler just returns its
        (already-fused) results as ok."""
        mock_kb.search.return_value = [make_search_result("found it", "/file.md")]
        server = create_knowledge_server(mock_kb)

        result = call_tool(
            server,
            "kb_search",
            {
                "query": "test",
                "collections": ["docs", "missing"],
            },
        )

        assert result["status"] == "ok"
        assert result["result_count"] == 1
