"""
Tests for KnowledgeBase and APIEmbedder.

APIEmbedder tests mock the client's httpx POST to avoid network
calls.  KnowledgeBase tests mock Embedder.create with deterministic vectors
and use real ChromaDB indexes and llama-index chunking on temp files.
Reranking is mocked since sentence-transformers is a heavy dependency.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
import numpy as np
import pytest

from dsagt.knowledge import APIEmbedder, KnowledgeBase, CODE_LANGUAGES


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


def make_http_response(texts: list[str], dim: int = EMBEDDING_DIM, status: int = 200):
    """Build a real ``httpx.Response`` mimicking an OpenAI ``/v1/embeddings``
    reply for the given texts.

    Returning a real ``httpx.Response`` (not a MagicMock) lets the client's
    own ``raise_for_status()`` / ``json()`` calls run for real, so tests
    exercise the actual parsing + error-classification paths.
    """
    rng = np.random.RandomState(0)
    data = [
        {"index": i, "embedding": rng.randn(dim).tolist()} for i in range(len(texts))
    ]
    return httpx.Response(
        status,
        json={"data": data},
        request=httpx.Request("POST", "http://test/embeddings"),
    )


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
# APIEmbedder
# ---------------------------------------------------------------------------


class TestAPIEmbedder:

    def test_missing_base_url_raises(self):
        """Constructor raises ValueError when no base URL is available."""
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("OPENAI_BASE_URL", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="base URL required"):
                    APIEmbedder(api_key="test-key", base_url=None)

    def test_missing_api_key_raises(self):
        """Constructor raises ValueError when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("LLM_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key required"):
                    APIEmbedder(api_key=None, base_url="http://test")

    def test_explicit_api_key(self):
        """Constructor accepts an explicit API key."""
        client = APIEmbedder(api_key="explicit-key", base_url="http://test")
        assert client.api_key == "explicit-key"
        client.close()

    def test_model_name_sent_verbatim(self):
        """The model string is sent unchanged — no provider prefix.

        Gateways route by alias, so lab-specific suffixes
        (``text-embedding-3-small-project``) and HuggingFace-style names
        with slashes (``lbl/nomic-embed-text``) must reach the endpoint
        exactly as configured.
        """
        for name in (
            "my-embed",
            "text-embedding-3-small-project",
            "lbl/nomic-embed-text",
        ):
            client = APIEmbedder(api_key="k", base_url="http://test", model=name)
            assert client.model == name
            client.close()

    def test_embeddings_url_built_from_base_url(self):
        """The embeddings route hangs off the (``/v1``) base URL root."""
        client = APIEmbedder(
            api_key="k",
            base_url="https://gw.example.com/v1/",
            model="m",
        )
        assert client._embeddings_url == "https://gw.example.com/v1/embeddings"
        client.close()

    def test_embed_posts_correct_request(self):
        """The POST carries the right URL, model/input body, and auth header."""
        client = APIEmbedder(
            api_key="my-key",
            model="test-model",
            base_url="https://example.com/v1",
        )
        with patch.object(
            client._client,
            "post",
            return_value=make_http_response(["hello"]),
        ) as mock_post:
            result = client.embed(["hello"])

        assert result.shape == (1, EMBEDDING_DIM)
        assert mock_post.call_count == 1
        url = mock_post.call_args.args[0]
        kwargs = mock_post.call_args.kwargs
        assert url == "https://example.com/v1/embeddings"
        assert kwargs["json"] == {"model": "test-model", "input": ["hello"]}
        assert kwargs["headers"]["Authorization"] == "Bearer my-key"
        client.close()

    def test_embed_returns_vectors_in_index_order(self):
        """Out-of-order response data is sorted back to input order."""
        rng = np.random.RandomState(7)
        out_of_order = [
            {"index": 1, "embedding": rng.randn(EMBEDDING_DIM).tolist()},
            {"index": 0, "embedding": rng.randn(EMBEDDING_DIM).tolist()},
        ]
        resp = httpx.Response(
            200,
            json={"data": out_of_order},
            request=httpx.Request("POST", "http://test/embeddings"),
        )

        client = APIEmbedder(api_key="k", base_url="http://test")
        with patch.object(client._client, "post", return_value=resp):
            result = client.embed(["first", "second"])
        assert result.shape == (2, EMBEDDING_DIM)
        # First-row vector matches the data entry with index=0 (the second list element)
        assert np.allclose(
            result[0], np.array(out_of_order[1]["embedding"], dtype=np.float32)
        )
        client.close()


# ---------------------------------------------------------------------------
# APIEmbedder - retry and error propagation
# ---------------------------------------------------------------------------


class TestAPIEmbedderErrors:
    """Verify the explicit rate-limit retry layer in APIEmbedder.

    The retry layer exists because lab gateways enforce Azure-style 60s
    quota windows that a generic exponential backoff would undershoot.
    These tests pin the contract:

    * Authentication / bad-request errors are NOT retried (fail fast on
      misconfiguration).
    * Rate-limit and transient errors ARE retried up to max_attempts.
    * The ``Retry-After`` header / body hint is honored.
    """

    def _err_response(self, status, headers=None, json_body=None):
        return httpx.Response(
            status,
            headers=headers or {},
            json=json_body if json_body is not None else {"error": "x"},
            request=httpx.Request("POST", "http://test/embeddings"),
        )

    @patch("dsagt.knowledge.time.sleep")
    def test_authentication_error_propagates_immediately(self, mock_sleep):
        """A 401 must NOT be retried — this is a misconfiguration, not transient."""
        client = APIEmbedder(api_key="bad-key", base_url="http://test")
        with patch.object(
            client._client,
            "post",
            return_value=self._err_response(401),
        ) as mock_post:
            with pytest.raises(httpx.HTTPStatusError):
                client.embed(["test"])
        # No retries: one call, no sleeps.
        assert mock_post.call_count == 1
        assert mock_sleep.call_count == 0
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    def test_rate_limit_retries_then_propagates(self, mock_sleep):
        """A persistent rate limit retries up to max_attempts then raises."""
        resp = self._err_response(429, headers={"retry-after": "60"})
        client = APIEmbedder(api_key="k", base_url="http://test")
        with patch.object(client._client, "post", return_value=resp) as mock_post:
            with pytest.raises(httpx.HTTPStatusError):
                client.embed(["test"])

        # max_attempts is 6 — that's 6 calls and 5 sleeps between them.
        assert mock_post.call_count == 6
        assert mock_sleep.call_count == 5
        # Each sleep should respect the Retry-After 60s hint.
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 60.0
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    def test_rate_limit_retries_then_succeeds(self, mock_sleep):
        """If the rate limit clears on a retry, embed() returns successfully."""
        client = APIEmbedder(api_key="k", base_url="http://test")
        # Two rate-limit failures, then success.
        side = [
            self._err_response(429, headers={"retry-after": "60"}),
            self._err_response(429, headers={"retry-after": "60"}),
            make_http_response(["one"]),
        ]
        with patch.object(client._client, "post", side_effect=side) as mock_post:
            result = client.embed(["one"])

        assert result.shape == (1, EMBEDDING_DIM)
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    def test_transient_connection_error_retries(self, mock_sleep):
        """Transport errors (connection reset) are retryable."""
        client = APIEmbedder(api_key="k", base_url="http://test")
        err = httpx.ConnectError(
            "connection reset",
            request=httpx.Request("POST", "http://test/embeddings"),
        )
        with patch.object(client._client, "post", side_effect=err) as mock_post:
            with pytest.raises(httpx.ConnectError):
                client.embed(["test"])
        assert mock_post.call_count == 6
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    def test_rate_limit_body_hint_honored_without_header(self, mock_sleep):
        """When there's no Retry-After header, a 429 body hint is parsed."""
        body = {
            "error": {
                "message": (
                    "rate limit exceeded. Please retry after 90 seconds. "
                    "To increase your default rate limit, visit ..."
                )
            }
        }
        resp = self._err_response(429, json_body=body)
        client = APIEmbedder(api_key="k", base_url="http://test")
        with patch.object(client._client, "post", return_value=resp) as mock_post:
            with pytest.raises(httpx.HTTPStatusError):
                client.embed(["test"])

        assert mock_post.call_count == 6
        # The 90s hint from the body must be honored, not the 60s default.
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 90.0
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    def test_timeout_retries(self, mock_sleep):
        """Timeouts are transient and should be retried with exponential backoff."""
        client = APIEmbedder(api_key="k", base_url="http://test")
        err = httpx.ReadTimeout(
            "Request timed out",
            request=httpx.Request("POST", "http://test/embeddings"),
        )
        with patch.object(client._client, "post", side_effect=err) as mock_post:
            with pytest.raises(httpx.ReadTimeout):
                client.embed(["test"])
        assert mock_post.call_count == 6
        # Exponential backoff capped at 30s: 2^1, 2^2, 2^3, 2^4, min(2^5, 30).
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == [2.0, 4.0, 8.0, 16.0, 30.0]
        client.close()


class TestAPIEmbedderBatching:
    """Long inputs are split into batch_size chunks for the embedding API."""

    def test_batches_when_over_batch_size(self):
        """A 250-text input with batch_size=100 produces 3 API calls."""
        client = APIEmbedder(
            api_key="k",
            base_url="http://test",
            batch_size=100,
        )

        def make_response(url, *, json, headers):
            return make_http_response(json["input"])

        with patch.object(
            client._client,
            "post",
            side_effect=make_response,
        ) as mock_post:
            texts = [f"chunk {i}" for i in range(250)]
            result = client.embed(texts)

        assert result.shape == (250, EMBEDDING_DIM)
        assert mock_post.call_count == 3
        # Verify the batch sizes were 100, 100, 50.
        batch_sizes = [
            len(call.kwargs["json"]["input"]) for call in mock_post.call_args_list
        ]
        assert batch_sizes == [100, 100, 50]
        client.close()

    def test_single_call_when_under_batch_size(self):
        """Inputs within batch_size make a single call (no batching loop)."""
        client = APIEmbedder(
            api_key="k",
            base_url="http://test",
            batch_size=100,
        )
        with patch.object(
            client._client,
            "post",
            return_value=make_http_response(["a", "b", "c", "d", "e"]),
        ) as mock_post:
            result = client.embed(["a", "b", "c", "d", "e"])

        assert result.shape == (5, EMBEDDING_DIM)
        assert mock_post.call_count == 1
        client.close()


class TestRetryAfterParsing:
    """Standalone tests for the retry-after parser used by the retry layer."""

    def test_extracts_seconds_from_message(self):
        from dsagt.knowledge import _extract_retry_after_seconds

        msg = "Please retry after 60 seconds"
        assert _extract_retry_after_seconds(msg) == 60.0

    def test_extracts_seconds_with_decimal(self):
        from dsagt.knowledge import _extract_retry_after_seconds

        assert _extract_retry_after_seconds("retry after 12.5 seconds") == 12.5

    def test_case_insensitive(self):
        from dsagt.knowledge import _extract_retry_after_seconds

        assert _extract_retry_after_seconds("RETRY AFTER 30 SECONDS") == 30.0

    def test_returns_default_when_no_hint(self):
        from dsagt.knowledge import _extract_retry_after_seconds

        assert _extract_retry_after_seconds("rate limit", default=42.0) == 42.0

    def test_finds_hint_in_long_message(self):
        from dsagt.knowledge import _extract_retry_after_seconds

        # The real lab error nests the hint deep inside a JSON body.
        msg = (
            'Openai_likeException - {"error":{"message":'
            '"AzureException RateLimitError - exceeded quota. '
            'Please retry after 75 seconds. To increase ..."}}'
        )
        assert _extract_retry_after_seconds(msg) == 75.0


# ---------------------------------------------------------------------------
# KnowledgeBase.ingest exclude_patterns
# ---------------------------------------------------------------------------


class TestIngestExcludePatterns:
    """The exclude_patterns parameter on ingest() filters out files whose
    relative path matches any of the supplied glob patterns.  Used by
    dsagt-setup-kb to skip tests, build artifacts, and private modules
    when embedding upstream library source for the core knowledge base.
    """

    @pytest.fixture
    def kb(self, tmp_path):
        index_dir = tmp_path / "index"
        mock_client = MagicMock()
        mock_client.embed = fake_embed
        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            yield kb
            kb.close()

    @pytest.fixture
    def repo_layout(self, tmp_path):
        """Mini repo with a realistic mix of source, tests, examples, and
        build artifacts."""
        root = tmp_path / "mylib_repo"
        root.mkdir()

        # Public source
        (root / "mylib").mkdir()
        (root / "mylib" / "__init__.py").write_text('"""Mylib package."""\n')
        (root / "mylib" / "core.py").write_text("def public_fn():\n    return 1\n")
        (root / "mylib" / "_internal.py").write_text("def _hidden():\n    return 2\n")

        # Tests in a subdirectory
        (root / "mylib" / "tests").mkdir()
        (root / "mylib" / "tests" / "__init__.py").write_text("")
        (root / "mylib" / "tests" / "test_core.py").write_text(
            "def test_public_fn():\n    assert True\n"
        )

        # Top-level tests dir as well
        (root / "tests").mkdir()
        (root / "tests" / "test_integration.py").write_text(
            "def test_smoke():\n    assert True\n"
        )
        (root / "tests" / "conftest.py").write_text("import pytest\n")

        # Examples (kept on purpose for the agent)
        (root / "examples").mkdir()
        (root / "examples" / "quickstart.py").write_text(
            "from mylib import public_fn\nprint(public_fn())\n"
        )

        # Docs
        (root / "docs").mkdir()
        (root / "docs" / "guide.md").write_text("# Guide\n\nUse mylib.\n")

        # Build cruft
        (root / "mylib" / "__pycache__").mkdir()
        (root / "mylib" / "__pycache__" / "core.cpython-312.pyc").write_text("")

        return root

    def test_no_exclude_keeps_everything(self, kb, repo_layout):
        result = kb.ingest(
            repo_layout,
            collection_name="full",
            file_types=["py", "md"],
        )
        # All py + md files except the .pyc which isn't in file_types.
        # 8 .py files (mylib/__init__.py, mylib/core.py, mylib/_internal.py,
        # mylib/tests/__init__.py, mylib/tests/test_core.py,
        # tests/test_integration.py, tests/conftest.py, examples/quickstart.py)
        # + 1 .md file = 9 total.
        assert result["files"] == 9

    def test_exclude_tests_directory(self, kb, repo_layout):
        """Pattern 'tests' should match the tests segment in any path."""
        result = kb.ingest(
            repo_layout,
            collection_name="no_tests",
            file_types=["py", "md"],
            exclude_patterns=["tests"],
        )
        # Excludes: mylib/tests/__init__.py, mylib/tests/test_core.py,
        # tests/test_integration.py, tests/conftest.py = 4 files
        # Keeps: mylib/__init__.py, mylib/core.py, mylib/_internal.py,
        # examples/quickstart.py, docs/guide.md = 5 files
        assert result["files"] == 5

    def test_exclude_test_files_by_basename(self, kb, repo_layout):
        """Pattern 'test_*.py' matches the basename anywhere in the tree."""
        result = kb.ingest(
            repo_layout,
            collection_name="no_test_files",
            file_types=["py", "md"],
            exclude_patterns=["test_*.py"],
        )
        # Excludes: mylib/tests/test_core.py, tests/test_integration.py
        # (the conftest and __init__ in tests/ are kept by this pattern alone)
        # 9 - 2 = 7 files
        assert result["files"] == 7

    def test_exclude_private_modules(self, kb, repo_layout):
        """Pattern '_*.py' excludes private modules.

        Note: _*.py also matches __init__.py because the basename starts
        with underscore.  This is the documented behavior — callers who
        want to keep __init__.py should use a more specific pattern.
        """
        result = kb.ingest(
            repo_layout,
            collection_name="no_private",
            file_types=["py", "md"],
            exclude_patterns=["_*.py"],
        )
        # Excludes: mylib/_internal.py, mylib/__init__.py,
        # mylib/tests/__init__.py = 3 files
        # 9 - 3 = 6 files
        assert result["files"] == 6

    def test_exclude_pycache_dir(self, kb, repo_layout):
        """__pycache__ should be excluded by directory-segment match."""
        # Pre-populate the cache with a parseable .py file so it actually
        # shows up in the file list when we DON'T filter.
        (repo_layout / "mylib" / "__pycache__" / "fake.py").write_text("x = 1\n")

        result_unfiltered = kb.ingest(
            repo_layout,
            collection_name="with_cache",
            file_types=["py"],
        )
        result_filtered = kb.ingest(
            repo_layout,
            collection_name="no_cache",
            file_types=["py"],
            exclude_patterns=["__pycache__"],
        )
        assert result_filtered["files"] == result_unfiltered["files"] - 1

    def test_combined_default_patterns(self, kb, repo_layout):
        """The setup_core_kb default set: tests + private + cache."""
        from dsagt.commands.setup_core_kb import DEFAULT_EXCLUDE_PATTERNS

        result = kb.ingest(
            repo_layout,
            collection_name="defaults",
            file_types=["py", "md"],
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
        )
        # Should keep: mylib/core.py, examples/quickstart.py, docs/guide.md
        # Should exclude: tests dirs, conftest, test_*.py, _*.py, __init__.py
        assert result["files"] == 3

    def test_default_patterns_keep_packaging_metadata(self, kb, tmp_path):
        """pyproject.toml / setup.py / setup.cfg must NOT be excluded by
        the default set — the agent uses them to install dependencies
        when registering tools that depend on the library.
        """
        from dsagt.commands.setup_core_kb import DEFAULT_EXCLUDE_PATTERNS

        root = tmp_path / "pkg_repo"
        root.mkdir()
        (root / "pyproject.toml").write_text(
            '[project]\nname = "mylib"\nversion = "1.2.3"\n'
        )
        (root / "setup.py").write_text(
            'from setuptools import setup\nsetup(name="mylib")\n'
        )
        (root / "setup.cfg").write_text("[metadata]\nname = mylib\n")
        (root / "mylib").mkdir()
        (root / "mylib" / "core.py").write_text("def f():\n    return 1\n")

        result = kb.ingest(
            root,
            collection_name="pkg_meta",
            file_types=["py", "toml", "cfg"],
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
        )
        # core.py + pyproject.toml + setup.py + setup.cfg = 4 files
        # (mylib has no __init__.py in this fixture)
        assert result["files"] == 4


class TestCollectFilesDirectly:
    """Unit tests for KnowledgeBase._collect_files extracted from ingest().

    The whole point of pulling this helper out of ingest() was to make
    file-discovery and exclude-pattern logic testable WITHOUT spinning up
    an embedder, an index, or a chunker.  These tests exercise the helper
    directly: no mocked Embedder.create context, no add_entries call, no
    cleanup of cached collections.  If a regression in the file-walk or
    fnmatch logic ever lands, these tests fail in milliseconds and point
    at the exact problem instead of being buried under ingest() setup.
    """

    @pytest.fixture
    def kb(self, tmp_path):
        # Build a minimal KB.  We don't call any method that would touch
        # the embedder, so the embedder kwargs don't matter.
        return KnowledgeBase(
            index_dir=tmp_path / "index",
            default_embedder="local",
            model="unused",
        )

    @pytest.fixture
    def repo(self, tmp_path):
        """Tiny repo with public source, tests, examples, and a __pycache__."""
        root = tmp_path / "minirepo"
        root.mkdir()
        (root / "core.py").write_text("def f(): return 1\n")
        (root / "_priv.py").write_text("def _g(): return 2\n")
        (root / "tests").mkdir()
        (root / "tests" / "test_core.py").write_text("def test_f(): assert True\n")
        (root / "examples").mkdir()
        (root / "examples" / "demo.py").write_text("from minirepo import f\n")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "core.cpython-312.pyc").write_text("binary")
        (root / "README.md").write_text("# Mini\n")
        return root

    def test_no_exclude_returns_all_matching_extensions(self, kb, repo):
        files = kb._collect_files(repo, ["py", "md"], exclude_patterns=None)
        names = sorted(f.name for f in files)
        # 4 .py + 1 .md (the .pyc is not in file_types)
        assert names == ["README.md", "_priv.py", "core.py", "demo.py", "test_core.py"]

    def test_exclude_directory_segment(self, kb, repo):
        """Pattern 'tests' must match the tests/ directory anywhere."""
        files = kb._collect_files(repo, ["py"], exclude_patterns=["tests"])
        names = sorted(f.name for f in files)
        assert "test_core.py" not in names
        assert "core.py" in names

    def test_exclude_basename_glob(self, kb, repo):
        """Pattern '_*.py' must match private modules by basename."""
        files = kb._collect_files(repo, ["py"], exclude_patterns=["_*.py"])
        names = sorted(f.name for f in files)
        assert "_priv.py" not in names
        assert "core.py" in names

    def test_empty_pattern_list_is_noop(self, kb, repo):
        """An empty exclude list returns the same files as None."""
        files_none = kb._collect_files(repo, ["py"], exclude_patterns=None)
        files_empty = kb._collect_files(repo, ["py"], exclude_patterns=[])
        assert sorted(files_none) == sorted(files_empty)

    def test_returns_paths_not_strings(self, kb, repo):
        """Result is list[Path], not list[str] — _chunk_file expects Path."""
        files = kb._collect_files(repo, ["py"], exclude_patterns=None)
        assert all(isinstance(f, Path) for f in files)


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

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
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

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
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

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            kb.ingest(folder_a)
            kb.ingest(folder_b)

            results = kb.search(
                "rockets", collection="collection_a", top_k=5, rerank=False
            )
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

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            kb.ingest(source_folder)

            # Clear the store cache that ingest populated
            store = kb._store
            store._cache.clear()

            # First load reads from disk
            index1, chunks1 = store._load("docs")
            assert "docs" in store._cache

            # Second load returns cached
            index2, chunks2 = store._load("docs")
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
        with patch("dsagt.knowledge.Embedder.create"):
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

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            with KnowledgeBase(index_dir=tmp_path / "index") as kb:
                # Trigger lazy embedder construction so close() has something
                # to clean up.
                kb._store.embedder

            mock_client.close.assert_called_once()


class TestStoreEmbedderConstruction:
    """One embedder per store, built lazily from explicit args.

    Per-collection embedder routing was removed: the store fixes a single
    embedder at construction, so the explicit args the KB was given flow
    straight through to ``Embedder.create`` (named, no kwargs dict) on first use.
    """

    def test_store_builds_embedder_from_args(self, tmp_path):
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            kb = KnowledgeBase(
                index_dir=tmp_path / "index",
                default_embedder="api",
                api_key="sk-real-key",
                base_url="https://embed.example.com",
                model="text-embedding-3-small",
            )
            kb._store.embedder  # trigger lazy construction

            mock_make.assert_called_once_with(
                "api",
                model="text-embedding-3-small",
                base_url="https://embed.example.com",
                api_key="sk-real-key",
                device=None,
            )

    def test_embedder_built_once_and_cached(self, tmp_path):
        """Repeated access builds the embedder exactly once."""
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            kb = KnowledgeBase(
                index_dir=tmp_path / "index",
                default_embedder="api",
                api_key="sk-real-key",
                base_url="https://embed.example.com",
                model="default-model",
            )
            kb._store.embedder
            kb._store.embedder
            assert mock_make.call_count == 1


# ---------------------------------------------------------------------------
# BM25 sparse index + RRF hybrid retrieval
# ---------------------------------------------------------------------------


class TestBM25Tokenize:
    """The tokenizer drives recall — verify identifier-style splits land."""

    def test_lowercases(self):
        from dsagt.knowledge import _bm25_tokenize

        assert _bm25_tokenize("Hello WORLD") == ["hello", "world"]

    def test_splits_on_underscore(self):
        from dsagt.knowledge import _bm25_tokenize

        # snake_case must fan out for BM25 to score "user_id" queries against
        # surrounding code.
        assert _bm25_tokenize("get_user_id") == ["get", "user", "id"]

    def test_splits_on_hyphen(self):
        from dsagt.knowledge import _bm25_tokenize

        assert _bm25_tokenize("kb-ingest") == ["kb", "ingest"]

    def test_drops_punctuation(self):
        from dsagt.knowledge import _bm25_tokenize

        assert _bm25_tokenize("foo.bar(baz)") == ["foo", "bar", "baz"]

    def test_keeps_numbers(self):
        from dsagt.knowledge import _bm25_tokenize

        assert _bm25_tokenize("v1.2.3 model") == ["v1", "2", "3", "model"]


class TestBM25Index:

    def test_build_then_search_finds_exact_match(self, tmp_path):
        from dsagt.knowledge import BM25Index

        idx = BM25Index()
        idx.build(
            [
                "Alpha rocket fuel composition.",
                "Beta submarine pressure hull.",
                "Gamma rocket telemetry parser.",
            ]
        )
        scores, indices = idx.search("rocket", k=3)
        assert len(indices) == 3
        # Docs 0 and 2 mention "rocket", should outrank doc 1.
        top_two = set(indices[:2].tolist())
        assert top_two == {0, 2}

    def test_empty_corpus_returns_empty(self, tmp_path):
        from dsagt.knowledge import BM25Index

        idx = BM25Index()
        idx.build([])
        scores, indices = idx.search("anything", k=5)
        assert len(scores) == 0 and len(indices) == 0

    def test_empty_query_returns_empty(self, tmp_path):
        from dsagt.knowledge import BM25Index

        idx = BM25Index()
        idx.build(["alpha", "beta"])
        # Punctuation-only query tokenizes to []; should not crash.
        scores, indices = idx.search("...!!!", k=5)
        assert len(scores) == 0 and len(indices) == 0

    def test_save_load_roundtrip(self, tmp_path):
        from dsagt.knowledge import BM25Index

        idx = BM25Index()
        idx.build(["foo bar", "baz qux", "foo qux"])
        idx.save(tmp_path)

        loaded = BM25Index.load(tmp_path)
        assert loaded.size == 3
        scores_a, idx_a = idx.search("foo", k=2)
        scores_b, idx_b = loaded.search("foo", k=2)
        assert idx_a.tolist() == idx_b.tolist()
        assert scores_a.tolist() == scores_b.tolist()

    def test_load_missing_file_returns_empty(self, tmp_path):
        from dsagt.knowledge import BM25Index

        loaded = BM25Index.load(tmp_path)
        assert loaded.size == 0


class TestRRFMerge:

    def test_single_ranker_passes_through_order(self):
        from dsagt.knowledge import _rrf_merge

        merged = _rrf_merge([[5, 2, 7]])
        assert [idx for idx, _ in merged] == [5, 2, 7]

    def test_two_rankers_promote_shared_top(self):
        from dsagt.knowledge import _rrf_merge

        # Doc 1 is rank-1 in both rankings; doc 2 only in one.  RRF must
        # rank doc 1 above all unique docs.
        merged = _rrf_merge([[1, 2, 3], [1, 4, 5]])
        idxs = [idx for idx, _ in merged]
        assert idxs[0] == 1

    def test_skips_negative_indices(self):
        from dsagt.knowledge import _rrf_merge

        # Backends pad with -1 when fewer than k results; must not pollute scores.
        merged = _rrf_merge([[0, -1, -1], [0, 1, 2]])
        idxs = [idx for idx, _ in merged]
        assert -1 not in idxs


class TestFederatedSearch:
    """KnowledgeBase.search owns collection→store routing + cross-collection RRF."""

    @pytest.fixture
    def kb(self, tmp_path):
        mock_client = MagicMock()
        mock_client.embed = fake_embed
        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            kb = KnowledgeBase(index_dir=tmp_path / "index")
            kb.add_entries(
                texts=["alpha quality filtering", "alpha assembly step"],
                collection="coll_a",
            )
            kb.add_entries(
                texts=["beta quality filtering", "beta assembly step"],
                collection="coll_b",
            )
            yield kb
            kb.close()

    def test_single_collection_routes_to_store(self, kb):
        results = kb.search("quality", collection="coll_a", top_k=5, rerank=False)
        assert results
        assert all(r["chunk"]["metadata"]["collection"] == "coll_a" for r in results)

    def test_multi_collection_fuses_both(self, kb):
        results = kb.search(
            "quality filtering",
            collections=["coll_a", "coll_b"],
            top_k=10,
            rerank=False,
        )
        seen = {r["chunk"]["metadata"]["collection"] for r in results}
        assert seen == {"coll_a", "coll_b"}
        # Fused scores are descending RRF scores.
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_missing_single_collection_raises(self, kb):
        with pytest.raises(ValueError, match="not found"):
            kb.search("x", collection="nope", top_k=5)

    def test_partial_missing_skips_and_returns_found(self, kb):
        results = kb.search(
            "quality", collections=["coll_a", "nope"], top_k=5, rerank=False
        )
        assert results
        assert all(r["chunk"]["metadata"]["collection"] == "coll_a" for r in results)

    def test_all_missing_raises_all_failed(self, kb):
        with pytest.raises(ValueError, match="All collections failed"):
            kb.search("x", collections=["nope1", "nope2"], top_k=5)

    def test_no_target_raises(self, kb):
        with pytest.raises(ValueError, match="Provide"):
            kb.search("x", top_k=5)


class TestHybridSearch:
    """End-to-end hybrid search behavior on a real KnowledgeBase."""

    @pytest.fixture
    def kb_with_data(self, tmp_path):
        index_dir = tmp_path / "index"
        source_folder = tmp_path / "test_docs"
        source_folder.mkdir()
        create_test_docs(source_folder)

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir)
            kb.ingest(source_folder)
            yield kb
            kb.close()

    def test_ingest_writes_bm25_when_hybrid(self, kb_with_data, tmp_path):
        """Hybrid is on by default; ingest must produce bm25.pkl."""
        bm25_path = tmp_path / "index" / "test_docs" / "bm25.pkl"
        assert bm25_path.exists()

    def test_search_succeeds_with_hybrid_default(self, kb_with_data):
        """Hybrid path returns results without errors."""
        results = kb_with_data.search(
            "installation",
            collection="test_docs",
            top_k=3,
            rerank=False,
        )
        assert len(results) > 0

    def test_add_entries_rebuilds_bm25(self, tmp_path):
        """add_entries must rebuild bm25.pkl with the new entry texts."""
        from dsagt.knowledge import KnowledgeBase

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
            kb = KnowledgeBase(index_dir=tmp_path / "index")
            kb.add_entries(
                texts=["alpha document", "beta document"],
                collection="memory",
            )
            bm25_path = tmp_path / "index" / "memory" / "bm25.pkl"
            assert bm25_path.exists()

            kb.add_entries(texts=["gamma document"], collection="memory")
            # BM25 must now know about all three docs.
            bm25 = kb._store._get_bm25("memory")
            assert bm25.size == 3
            kb.close()
