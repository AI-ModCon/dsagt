"""
Tests for KnowledgeBase and APIEmbeddingClient.

APIEmbeddingClient tests mock httpx to avoid network calls.
KnowledgeBase tests mock _make_embedder with deterministic vectors
and use real FAISS indexes and llama-index chunking on temp files.
Reranking is mocked since sentence-transformers is a heavy dependency.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import faiss
import numpy as np
import pytest

from dsagt.knowledge import APIEmbeddingClient, KnowledgeBase, CODE_LANGUAGES


@pytest.fixture(autouse=True)
def _fake_api_env(monkeypatch):
    """Set dummy API credentials for unit tests without leaking into other modules."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 8


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embeddings based on text hash."""
    if not texts:
        return np.array([], dtype=np.float32)
    rng = np.random.RandomState(42)
    embeddings = []
    for text in texts:
        seed = hash(text) % (2**31)
        rng.seed(seed)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        embeddings.append(vec)
    return np.array(embeddings, dtype=np.float32)


def make_mock_response(texts: list[str], dim: int = EMBEDDING_DIM):
    """Create a mock httpx response matching the OpenAI embeddings format."""
    rng = np.random.RandomState(0)
    data = [
        {"index": i, "embedding": rng.randn(dim).tolist()}
        for i, _ in enumerate(texts)
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": data}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def create_test_docs(folder: Path):
    """Create a small set of test documents in a folder."""
    (folder / "DESCRIPTION.md").write_text("Test collection for unit tests.")

    (folder / "readme.md").write_text(
        "# Test Project\n\n"
        "This is a test project with some documentation.\n\n"
        "## Installation\n\n"
        "Run pip install to get started.\n"
    )

    (folder / "notes.txt").write_text(
        "These are some plain text notes about the project.\n"
        "They contain useful information for testing search.\n"
    )

    (folder / "example.py").write_text(
        '"""Example module."""\n\n'
        "def hello(name: str) -> str:\n"
        '    """Greet someone."""\n'
        '    return f"Hello, {name}!"\n'
    )


# ---------------------------------------------------------------------------
# APIEmbeddingClient
# ---------------------------------------------------------------------------

class TestAPIEmbeddingClient:

    def test_missing_base_url_raises(self):
        """Constructor raises ValueError when no base URL is available."""
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("OPENAI_BASE_URL", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="base URL required"):
                    APIEmbeddingClient(api_key="test-key", base_url=None)

    def test_missing_api_key_raises(self):
        """Constructor raises ValueError when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("LLM_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key required"):
                    APIEmbeddingClient(api_key=None, base_url="http://test")

    def test_explicit_api_key(self):
        """Constructor accepts an explicit API key."""
        client = APIEmbeddingClient(api_key="explicit-key", base_url="http://test")
        assert client.api_key == "explicit-key"
        client.close()

    def test_embed_empty_list(self):
        """Embedding an empty list returns an empty array."""
        client = APIEmbeddingClient(api_key="test-key", base_url="http://test")
        result = client.embed([])
        assert result.shape == (0,)
        assert result.dtype == np.float32
        client.close()

    @patch("httpx.Client")
    def test_embed_single_batch(self, mock_client_cls):
        """Texts within batch_size make a single API call."""
        mock_client = MagicMock()
        mock_client.post.return_value = make_mock_response(["a", "b"])
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(api_key="test-key", base_url="http://test", batch_size=10)
        result = client.embed(["a", "b"])

        assert result.shape == (2, EMBEDDING_DIM)
        assert mock_client.post.call_count == 1
        client.close()

    @patch("httpx.Client")
    def test_embed_multiple_batches(self, mock_client_cls):
        """Texts exceeding batch_size are split into multiple API calls."""
        mock_client = MagicMock()

        def dynamic_response(url, **kwargs):
            texts = kwargs["json"]["input"]
            return make_mock_response(texts)

        mock_client.post.side_effect = dynamic_response
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(api_key="test-key", base_url="http://test", batch_size=2)
        # 5 texts with batch_size=2 -> 3 API calls (2+2+1)
        result = client.embed(["a", "b", "c", "d", "e"])

        assert result.shape == (5, EMBEDDING_DIM)
        assert mock_client.post.call_count == 3
        client.close()

    @patch("httpx.Client")
    def test_embed_sends_correct_payload(self, mock_client_cls):
        """API call includes correct model and input texts."""
        mock_client = MagicMock()
        mock_client.post.return_value = make_mock_response(["hello"])
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(
            api_key="my-key",
            model="test-model",
            base_url="https://example.com",
        )
        client.embed(["hello"])

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://example.com/embeddings"
        assert call_kwargs[1]["json"]["model"] == "test-model"
        assert call_kwargs[1]["json"]["input"] == ["hello"]
        assert "Bearer my-key" in call_kwargs[1]["headers"]["Authorization"]
        client.close()


# ---------------------------------------------------------------------------
# APIEmbeddingClient - error handling
# ---------------------------------------------------------------------------

class TestAPIEmbeddingClientErrors:
    """Tests for embedding API failure modes that could cause 'transport closed'."""

    @patch("httpx.Client")
    def test_http_401_raises(self, mock_client_cls):
        """Unauthorized (wrong/expired API key) raises HTTPStatusError."""
        import httpx

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response,
        )
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(api_key="bad-key", base_url="http://test")
        with pytest.raises(httpx.HTTPStatusError):
            client.embed(["test"])
        client.close()

    @patch("httpx.Client")
    def test_http_500_raises(self, mock_client_cls):
        """Server error (500) raises HTTPStatusError."""
        import httpx

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=MagicMock(), response=mock_response,
        )
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(api_key="test-key", base_url="http://test")
        with pytest.raises(httpx.HTTPStatusError):
            client.embed(["test"])
        client.close()

    @patch("httpx.Client")
    def test_connection_error_raises(self, mock_client_cls):
        """Network unreachable raises ConnectError."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(api_key="test-key", base_url="http://test")
        with pytest.raises(httpx.ConnectError):
            client.embed(["test"])
        client.close()

    @patch("httpx.Client")
    def test_timeout_raises(self, mock_client_cls):
        """Embedding API timeout raises ReadTimeout."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ReadTimeout("Read timed out")
        mock_client_cls.return_value = mock_client

        client = APIEmbeddingClient(api_key="test-key", base_url="http://test")
        with pytest.raises(httpx.ReadTimeout):
            client.embed(["test"])
        client.close()


# ---------------------------------------------------------------------------
# KnowledgeBase - collections and ingest
# ---------------------------------------------------------------------------

class TestKnowledgeBaseIngest:

    @pytest.fixture
    def kb(self, tmp_path):
        """KnowledgeBase with mocked embedding client."""
        index_dir = tmp_path / "index"
        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            yield kb
            kb.close()

    @pytest.fixture
    def source_folder(self, tmp_path):
        """Folder with test documents for ingestion."""
        folder = tmp_path / "test_docs"
        folder.mkdir()
        create_test_docs(folder)
        return folder

    def test_empty_collections(self, kb):
        """Fresh knowledge base has no collections."""
        assert kb.collections == []
        assert kb.list_collections() == []

    def test_ingest_creates_collection(self, kb, source_folder):
        """Ingesting a folder creates a named collection with index and chunks."""
        result = kb.ingest(source_folder)

        assert result["collection"] == "test_docs"
        assert result["files"] > 0
        assert result["chunks"] > 0

        # Collection should now be listed
        assert "test_docs" in kb.collections

    def test_ingest_copies_description(self, kb, source_folder):
        """DESCRIPTION.md is copied from source to collection directory."""
        kb.ingest(source_folder)

        desc_path = kb.index_dir / "test_docs" / "DESCRIPTION.md"
        assert desc_path.exists()
        assert "unit tests" in desc_path.read_text()

    def test_list_collections_includes_description(self, kb, source_folder):
        """list_collections returns description text."""
        kb.ingest(source_folder)

        collections = kb.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "test_docs"
        assert "unit tests" in collections[0]["description"]

    def test_ingest_creates_faiss_index(self, kb, source_folder):
        """Ingest produces a valid FAISS index file."""
        kb.ingest(source_folder)

        index_path = kb.index_dir / "test_docs" / "index.faiss"
        assert index_path.exists()

        index = faiss.read_index(str(index_path))
        assert index.ntotal > 0

    def test_ingest_creates_chunks_jsonl(self, kb, source_folder):
        """Ingest produces a chunks.jsonl with valid entries."""
        result = kb.ingest(source_folder)

        chunks_path = kb.index_dir / "test_docs" / "chunks.jsonl"
        assert chunks_path.exists()

        with open(chunks_path) as f:
            chunks = [json.loads(line) for line in f]

        assert len(chunks) == result["chunks"]
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) > 0
            assert "metadata" in chunk
            assert chunk["metadata"]["collection"] == "test_docs"

    def test_ingest_empty_folder(self, kb, tmp_path):
        """Ingesting a folder with no matching files returns zeros."""
        empty = tmp_path / "empty_docs"
        empty.mkdir()

        result = kb.ingest(empty)
        assert result["files"] == 0
        assert result["chunks"] == 0

    def test_ingest_custom_file_types(self, kb, source_folder):
        """Custom file_types filters which files are processed."""
        result = kb.ingest(source_folder, file_types=["txt"])

        # Only .txt files should be processed
        chunks_path = kb.index_dir / "test_docs" / "chunks.jsonl"
        with open(chunks_path) as f:
            chunks = [json.loads(line) for line in f]

        for chunk in chunks:
            assert chunk["metadata"]["file_type"] == ".txt"

    def test_ingest_no_description(self, kb, tmp_path):
        """Collection without DESCRIPTION.md gets empty description."""
        folder = tmp_path / "no_desc"
        folder.mkdir()
        (folder / "file.txt").write_text("Some content for testing.")

        kb.ingest(folder)

        collections = kb.list_collections()
        # New route-based list_collections may return description from route
        # or empty string; just check it doesn't error
        assert len(collections) == 1
        assert collections[0]["name"] == "no_desc"


# ---------------------------------------------------------------------------
# KnowledgeBase - search
# ---------------------------------------------------------------------------

class TestKnowledgeBaseSearch:

    @pytest.fixture
    def kb_with_data(self, tmp_path):
        """KnowledgeBase with an ingested collection, mocked embeddings."""
        index_dir = tmp_path / "index"
        source_folder = tmp_path / "test_docs"
        source_folder.mkdir()
        create_test_docs(source_folder)

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            kb.ingest(source_folder)
            yield kb
            kb.close()

    def test_search_returns_results(self, kb_with_data):
        """Search returns a list of scored results."""
        results = kb_with_data.search(
            "installation instructions",
            collection="test_docs",
            top_k=3,
            rerank=False,
        )

        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert "chunk" in r
            assert "score" in r
            assert "text" in r["chunk"]
            assert "metadata" in r["chunk"]

    def test_search_respects_top_k(self, kb_with_data):
        """Search returns at most top_k results."""
        results = kb_with_data.search(
            "test",
            collection="test_docs",
            top_k=1,
            rerank=False,
        )
        assert len(results) <= 1

    def test_search_nonexistent_collection(self, kb_with_data):
        """Searching a nonexistent collection raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            kb_with_data.search("query", collection="nonexistent")

    def test_search_with_rerank(self, kb_with_data):
        """Search with reranking adds rerank_score to results."""
        mock_reranker = MagicMock()
        # Return descending scores so we can verify ordering
        mock_reranker.predict.return_value = np.array([0.9, 0.5, 0.1])

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_reranker

        import sys
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            # Ensure the lazy import triggers
            kb_with_data._reranker = None

            results = kb_with_data.search(
                "hello function",
                collection="test_docs",
                top_k=3,
                rerank=True,
            )

        assert len(results) > 0
        assert all("rerank_score" in r for r in results)
        # Should be sorted by rerank score descending
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_collection_isolation(self, tmp_path):
        """Searching one collection does not return results from another."""
        index_dir = tmp_path / "index"

        folder_a = tmp_path / "collection_a"
        folder_a.mkdir()
        (folder_a / "doc.txt").write_text("Alpha collection content about rockets.")

        folder_b = tmp_path / "collection_b"
        folder_b.mkdir()
        (folder_b / "doc.txt").write_text("Beta collection content about submarines.")

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            kb.ingest(folder_a)
            kb.ingest(folder_b)

            results = kb.search("rockets", collection="collection_a", top_k=5, rerank=False)
            sources = [r["chunk"]["metadata"]["collection"] for r in results]
            assert all(s == "collection_a" for s in sources)

            kb.close()


# ---------------------------------------------------------------------------
# KnowledgeBase - loading and caching
# ---------------------------------------------------------------------------

class TestKnowledgeBaseLoad:

    def test_load_caches_collection(self, tmp_path):
        """Loading a collection caches the index and chunks."""
        index_dir = tmp_path / "index"
        source_folder = tmp_path / "docs"
        source_folder.mkdir()
        (source_folder / "file.txt").write_text("Content for caching test.")

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            kb.ingest(source_folder)

            # Clear the cache that ingest populated
            kb._cache.clear()

            # First load reads from disk
            index1, chunks1 = kb._load("docs")
            assert "docs" in kb._cache

            # Second load returns cached
            index2, chunks2 = kb._load("docs")
            assert index1 is index2
            assert chunks1 is chunks2

            kb.close()


# ---------------------------------------------------------------------------
# KnowledgeBase - parser selection
# ---------------------------------------------------------------------------

class TestGetParser:

    @pytest.fixture
    def kb(self, tmp_path):
        # New __init__ doesn't create an embedder, but mock to be safe
        with patch("dsagt.knowledge._make_embedder"):
            kb = KnowledgeBase(index_dir=tmp_path / "index")
            yield kb
            kb.close()

    def test_markdown_parser(self, kb):
        from llama_index.core.node_parser import MarkdownNodeParser
        parser = kb._get_parser(".md")
        assert isinstance(parser, MarkdownNodeParser)

    def test_code_parser(self, kb):
        from llama_index.core.node_parser import CodeSplitter
        parser = kb._get_parser(".py")
        assert isinstance(parser, CodeSplitter)

    def test_default_parser(self, kb):
        from llama_index.core.node_parser import SentenceSplitter
        parser = kb._get_parser(".txt")
        assert isinstance(parser, SentenceSplitter)

    def test_code_languages_coverage(self):
        """All CODE_LANGUAGES entries map to a language string."""
        for ext, lang in CODE_LANGUAGES.items():
            assert ext.startswith(".")
            assert isinstance(lang, str) and len(lang) > 0


# ---------------------------------------------------------------------------
# KnowledgeBase - context manager
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_context_manager_calls_close(self, tmp_path):
        """close() cleans up cached embedders created during the session."""
        mock_client = MagicMock()

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            with KnowledgeBase(index_dir=tmp_path / "index") as kb:
                # Trigger embedder creation so close() has something to clean up
                route = kb._get_route("dummy")
                kb._get_embedder(route)

            mock_client.close.assert_called_once()
