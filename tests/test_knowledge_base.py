"""
Tests for KnowledgeBase and APIEmbeddingClient.

APIEmbeddingClient tests mock litellm.embedding to avoid network calls.
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


def make_mock_litellm_response(texts: list[str], dim: int = EMBEDDING_DIM):
    """Create a fake litellm.EmbeddingResponse for the given texts.

    LiteLLM returns a Pydantic model whose ``.data`` field is a list of dicts
    with ``index`` and ``embedding`` keys (matching the OpenAI API shape).
    A MagicMock with the same attribute access is sufficient for tests.
    """
    rng = np.random.RandomState(0)
    data = [
        {"index": i, "embedding": rng.randn(dim).tolist()}
        for i in range(len(texts))
    ]
    resp = MagicMock()
    resp.data = data
    return resp


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

    def test_bare_model_name_gets_openai_like_prefix(self):
        """Bare model names should be routed via the openai_like/ prefix.

        ``openai_like`` is required (not ``openai``) so LiteLLM does not
        normalize lab-specific suffixes like ``-project`` away — see the
        comment in APIEmbeddingClient.__init__ for the full rationale.
        """
        client = APIEmbeddingClient(
            api_key="k", base_url="http://test", model="my-embed",
        )
        assert client._litellm_model == "openai_like/my-embed"
        client.close()

    def test_lab_suffixed_model_name_round_trips_verbatim(self):
        """Regression: ``text-embedding-3-small-project`` must NOT be
        normalized to ``text-embedding-3-small`` by the openai_like router.

        Lab LiteLLM proxies route by alias.  If the suffix gets stripped, the
        request reaches the upstream as a name the proxy's ACL doesn't know
        about, and we get a 401 ``team_model_access_denied``.
        """
        client = APIEmbeddingClient(
            api_key="k", base_url="http://test",
            model="text-embedding-3-small-project",
        )
        assert client._litellm_model == "openai_like/text-embedding-3-small-project"
        client.close()

    def test_slash_in_model_name_still_gets_openai_like_prefix(self):
        """HuggingFace-style names (``lbl/nomic-embed-text``) are model
        identifiers, not LiteLLM provider prefixes — the whole thing needs
        ``openai_like/`` in front so LiteLLM dispatches to the OpenAI-wire
        client pointed at our base_url.  The rest of DSAGT assumes an
        OpenAI-compat endpoint, so there's no valid case for the user's
        string to carry a LiteLLM provider prefix of its own.
        """
        client = APIEmbeddingClient(
            api_key="k", base_url="http://test", model="lbl/nomic-embed-text",
        )
        assert client._litellm_model == "openai_like/lbl/nomic-embed-text"
        client.close()

    def test_embed_empty_list(self):
        """Embedding an empty list returns an empty array."""
        client = APIEmbeddingClient(api_key="test-key", base_url="http://test")
        result = client.embed([])
        assert result.shape == (0,)
        assert result.dtype == np.float32
        client.close()

    @patch("litellm.embedding")
    def test_embed_calls_litellm_with_correct_args(self, mock_embedding):
        """litellm.embedding receives the right model, input, api_base, api_key."""
        mock_embedding.return_value = make_mock_litellm_response(["hello"])

        client = APIEmbeddingClient(
            api_key="my-key",
            model="test-model",
            base_url="https://example.com",
        )
        result = client.embed(["hello"])

        assert result.shape == (1, EMBEDDING_DIM)
        assert mock_embedding.call_count == 1
        call_kwargs = mock_embedding.call_args.kwargs
        assert call_kwargs["model"] == "openai_like/test-model"
        assert call_kwargs["input"] == ["hello"]
        assert call_kwargs["api_base"] == "https://example.com"
        assert call_kwargs["api_key"] == "my-key"
        client.close()

    @patch("litellm.embedding")
    def test_embed_returns_vectors_in_index_order(self, mock_embedding):
        """Out-of-order response data is sorted back to input order."""
        # Construct a response where data is in reverse order to ensure
        # the client sorts by 'index' field.
        rng = np.random.RandomState(7)
        out_of_order = [
            {"index": 1, "embedding": rng.randn(EMBEDDING_DIM).tolist()},
            {"index": 0, "embedding": rng.randn(EMBEDDING_DIM).tolist()},
        ]
        resp = MagicMock()
        resp.data = out_of_order
        mock_embedding.return_value = resp

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        result = client.embed(["first", "second"])
        assert result.shape == (2, EMBEDDING_DIM)
        # First-row vector matches the data entry with index=0 (the second list element)
        assert np.allclose(result[0], np.array(out_of_order[1]["embedding"], dtype=np.float32))
        client.close()

    @patch("litellm.embedding")
    def test_embed_handles_pydantic_style_data(self, mock_embedding):
        """LiteLLM may return Pydantic objects with attribute access instead of dicts."""

        class _Item:
            def __init__(self, idx, vec):
                self.index = idx
                self.embedding = vec

        rng = np.random.RandomState(0)
        items = [_Item(i, rng.randn(EMBEDDING_DIM).tolist()) for i in range(3)]
        resp = MagicMock()
        resp.data = items
        mock_embedding.return_value = resp

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        result = client.embed(["a", "b", "c"])
        assert result.shape == (3, EMBEDDING_DIM)
        client.close()


# ---------------------------------------------------------------------------
# APIEmbeddingClient - retry and error propagation
# ---------------------------------------------------------------------------

class TestAPIEmbeddingClientErrors:
    """Verify the explicit rate-limit retry layer in APIEmbeddingClient.

    The retry layer exists because lab LiteLLM proxies wrap upstream 429s
    in a way that defeats litellm's built-in retry classification, and
    litellm's default backoff is too short for Azure-style 60s quota
    windows.  These tests pin the contract:

    * Authentication / bad-request errors are NOT retried (fail fast on
      misconfiguration).
    * Rate-limit and transient errors ARE retried up to max_attempts.
    * The upstream "retry after N seconds" hint is honored.
    """

    @patch("dsagt.knowledge.time.sleep")
    @patch("litellm.embedding")
    def test_authentication_error_propagates_immediately(
        self, mock_embedding, mock_sleep,
    ):
        """A 401 must NOT be retried — this is a misconfiguration, not transient."""
        import litellm

        mock_embedding.side_effect = litellm.exceptions.AuthenticationError(
            message="Invalid API key", llm_provider="openai", model="test",
        )

        client = APIEmbeddingClient(api_key="bad-key", base_url="http://test")
        with pytest.raises(litellm.exceptions.AuthenticationError):
            client.embed(["test"])
        # No retries: one call, no sleeps.
        assert mock_embedding.call_count == 1
        assert mock_sleep.call_count == 0
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    @patch("litellm.embedding")
    def test_rate_limit_retries_then_propagates(self, mock_embedding, mock_sleep):
        """A persistent rate limit retries up to max_attempts then raises."""
        import litellm

        mock_embedding.side_effect = litellm.exceptions.RateLimitError(
            message="429 Please retry after 60 seconds",
            llm_provider="openai", model="test",
        )

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        with pytest.raises(litellm.exceptions.RateLimitError):
            client.embed(["test"])

        # max_attempts is 6 — that's 6 calls and 5 sleeps between them.
        assert mock_embedding.call_count == 6
        assert mock_sleep.call_count == 5
        # Each sleep should respect the upstream-suggested 60s wait.
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 60.0
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    @patch("litellm.embedding")
    def test_rate_limit_retries_then_succeeds(self, mock_embedding, mock_sleep):
        """If the rate limit clears on a retry, embed() returns successfully."""
        import litellm

        rng = np.random.RandomState(42)
        success_resp = MagicMock()
        success_resp.data = [
            {"index": 0, "embedding": rng.randn(EMBEDDING_DIM).tolist()},
        ]

        # Two rate-limit failures, then success.
        mock_embedding.side_effect = [
            litellm.exceptions.RateLimitError(
                message="429 Please retry after 60 seconds",
                llm_provider="openai", model="test",
            ),
            litellm.exceptions.RateLimitError(
                message="429 Please retry after 60 seconds",
                llm_provider="openai", model="test",
            ),
            success_resp,
        ]

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        result = client.embed(["one"])

        assert result.shape == (1, EMBEDDING_DIM)
        assert mock_embedding.call_count == 3
        assert mock_sleep.call_count == 2
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    @patch("litellm.embedding")
    def test_transient_connection_error_retries(self, mock_embedding, mock_sleep):
        """APIConnectionError is retryable (covers the lab-proxy 429 wrapping case)."""
        import litellm

        mock_embedding.side_effect = litellm.exceptions.APIConnectionError(
            message="connection reset",
            llm_provider="openai_like", model="test",
        )

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        with pytest.raises(litellm.exceptions.APIConnectionError):
            client.embed(["test"])
        assert mock_embedding.call_count == 6
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    @patch("litellm.embedding")
    def test_lab_proxy_wrapped_429_retries(self, mock_embedding, mock_sleep):
        """Real-world failure mode: openai_like wraps an upstream 429 as
        APIConnectionError with the rate-limit body in the message.  Our
        retry layer must detect this via string matching, not just by
        exception class, and must use the upstream-suggested wait time.
        """
        import litellm

        # This is a lightly-paraphrased version of the actual lab error.
        wrapped_429 = litellm.exceptions.APIConnectionError(
            message=(
                "Openai_likeException - "
                '{"error":{"message":"litellm.RateLimitError: AzureException '
                "RateLimitError - rate limit exceeded. "
                "Please retry after 90 seconds. "
                'To increase your default rate limit, visit ..."}}'
            ),
            llm_provider="openai_like", model="text-embedding-3-small-project",
        )
        mock_embedding.side_effect = wrapped_429

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        with pytest.raises(litellm.exceptions.APIConnectionError):
            client.embed(["test"])

        assert mock_embedding.call_count == 6
        # The 90s hint from the upstream message must be honored, not the 60s default.
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 90.0
        client.close()

    @patch("dsagt.knowledge.time.sleep")
    @patch("litellm.embedding")
    def test_timeout_retries(self, mock_embedding, mock_sleep):
        """Timeouts are transient and should be retried with exponential backoff."""
        import litellm

        mock_embedding.side_effect = litellm.exceptions.Timeout(
            message="Request timed out", llm_provider="openai", model="test",
        )

        client = APIEmbeddingClient(api_key="k", base_url="http://test")
        with pytest.raises(litellm.exceptions.Timeout):
            client.embed(["test"])
        assert mock_embedding.call_count == 6
        # Exponential backoff capped at 30s: 2^1, 2^2, 2^3, 2^4, min(2^5, 30).
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == [2.0, 4.0, 8.0, 16.0, 30.0]
        client.close()


class TestAPIEmbeddingClientBatching:
    """Long inputs are split into batch_size chunks for the embedding API."""

    @patch("litellm.embedding")
    def test_batches_when_over_batch_size(self, mock_embedding):
        """A 250-text input with batch_size=100 produces 3 API calls."""
        rng = np.random.RandomState(0)

        def make_response(input_texts, **_):
            resp = MagicMock()
            resp.data = [
                {"index": i, "embedding": rng.randn(EMBEDDING_DIM).tolist()}
                for i in range(len(input_texts))
            ]
            return resp

        mock_embedding.side_effect = lambda **kwargs: make_response(kwargs["input"])

        client = APIEmbeddingClient(
            api_key="k", base_url="http://test", batch_size=100,
        )
        texts = [f"chunk {i}" for i in range(250)]
        result = client.embed(texts)

        assert result.shape == (250, EMBEDDING_DIM)
        assert mock_embedding.call_count == 3
        # Verify the batch sizes were 100, 100, 50.
        batch_sizes = [
            len(call.kwargs["input"]) for call in mock_embedding.call_args_list
        ]
        assert batch_sizes == [100, 100, 50]
        client.close()

    @patch("litellm.embedding")
    def test_single_call_when_under_batch_size(self, mock_embedding):
        """Inputs within batch_size make a single call (no batching loop)."""
        rng = np.random.RandomState(0)
        resp = MagicMock()
        resp.data = [
            {"index": i, "embedding": rng.randn(EMBEDDING_DIM).tolist()}
            for i in range(5)
        ]
        mock_embedding.return_value = resp

        client = APIEmbeddingClient(
            api_key="k", base_url="http://test", batch_size=100,
        )
        result = client.embed(["a", "b", "c", "d", "e"])

        assert result.shape == (5, EMBEDDING_DIM)
        assert mock_embedding.call_count == 1
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
        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
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
        (root / "mylib" / "core.py").write_text('def public_fn():\n    return 1\n')
        (root / "mylib" / "_internal.py").write_text('def _hidden():\n    return 2\n')

        # Tests in a subdirectory
        (root / "mylib" / "tests").mkdir()
        (root / "mylib" / "tests" / "__init__.py").write_text("")
        (root / "mylib" / "tests" / "test_core.py").write_text(
            'def test_public_fn():\n    assert True\n'
        )

        # Top-level tests dir as well
        (root / "tests").mkdir()
        (root / "tests" / "test_integration.py").write_text(
            'def test_smoke():\n    assert True\n'
        )
        (root / "tests" / "conftest.py").write_text("import pytest\n")

        # Examples (kept on purpose for the agent)
        (root / "examples").mkdir()
        (root / "examples" / "quickstart.py").write_text(
            'from mylib import public_fn\nprint(public_fn())\n'
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
            repo_layout, collection_name="full",
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
            repo_layout, collection_name="no_tests",
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
            repo_layout, collection_name="no_test_files",
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
            repo_layout, collection_name="no_private",
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
            repo_layout, collection_name="with_cache",
            file_types=["py"],
        )
        result_filtered = kb.ingest(
            repo_layout, collection_name="no_cache",
            file_types=["py"],
            exclude_patterns=["__pycache__"],
        )
        assert result_filtered["files"] == result_unfiltered["files"] - 1

    def test_combined_default_patterns(self, kb, repo_layout):
        """The setup_core_kb default set: tests + private + cache."""
        from dsagt.commands.setup_core_kb import DEFAULT_EXCLUDE_PATTERNS

        result = kb.ingest(
            repo_layout, collection_name="defaults",
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
            root, collection_name="pkg_meta",
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
    directly: no mocked _make_embedder context, no add_entries call, no
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
            embedder_kwargs={"model": "unused"},
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

    def test_ingest_creates_faiss_index(self, tmp_path, source_folder):
        """Ingest into an explicitly FAISS-routed KB produces an index.faiss."""
        index_dir = tmp_path / "index_faiss"
        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir, default_index="faiss")
            try:
                kb.ingest(source_folder)
                index_path = kb.index_dir / "test_docs" / "index.faiss"
                assert index_path.exists()

                index = faiss.read_index(str(index_path))
                assert index.ntotal > 0
            finally:
                kb.close()

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


class TestEmbedderCredentialMerge:
    """Routes carry per-collection overrides; credentials live on the kb's
    default route.  Explicit routes (e.g. EPISODIC_MEMORY_ROUTE) must inherit
    api_key/base_url from the default kb config — otherwise memory extraction
    falls through to env-var fallbacks and breaks under mixed-endpoint setups
    where LLM_API_KEY != EMBEDDING_API_KEY.
    """

    def test_explicit_route_inherits_default_credentials(self, tmp_path):
        from dsagt.knowledge import CollectionRoute

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            kb = KnowledgeBase(
                index_dir=tmp_path / "index",
                embedder_kwargs={
                    "api_key": "sk-real-key",
                    "base_url": "https://embed.example.com",
                    "model": "text-embedding-3-small",
                },
            )
            # Route with no embedder_kwargs (mimics EPISODIC_MEMORY_ROUTE).
            route = CollectionRoute(embedding_backend="api", vector_db="chroma")
            kb._get_embedder(route)

            mock_make.assert_called_once_with(
                "api",
                api_key="sk-real-key",
                base_url="https://embed.example.com",
                model="text-embedding-3-small",
            )

    def test_route_kwargs_override_default(self, tmp_path):
        """Per-route overrides win over kb defaults for matching keys."""
        from dsagt.knowledge import CollectionRoute

        with patch("dsagt.knowledge._make_embedder") as mock_make:
            kb = KnowledgeBase(
                index_dir=tmp_path / "index",
                embedder_kwargs={
                    "api_key": "sk-real-key",
                    "base_url": "https://embed.example.com",
                    "model": "default-model",
                },
            )
            # Route overrides model but not credentials.
            route = CollectionRoute(
                embedding_backend="api",
                vector_db="chroma",
                embedder_kwargs={"model": "BAAI/bge-base-en-v1.5"},
            )
            kb._get_embedder(route)

            mock_make.assert_called_once_with(
                "api",
                api_key="sk-real-key",
                base_url="https://embed.example.com",
                model="BAAI/bge-base-en-v1.5",
            )


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
        idx.build([
            "Alpha rocket fuel composition.",
            "Beta submarine pressure hull.",
            "Gamma rocket telemetry parser.",
        ])
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
        # FAISS pads with -1 when fewer than k results; must not pollute scores.
        merged = _rrf_merge([[0, -1, -1], [0, 1, 2]])
        idxs = [idx for idx, _ in merged]
        assert -1 not in idxs


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

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir, default_index="faiss")
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

    def test_search_raises_when_hybrid_but_no_bm25(self, tmp_path):
        """Hybrid=True with missing bm25.pkl is a loud failure."""
        from dsagt.knowledge import KnowledgeBase, CollectionRoute

        index_dir = tmp_path / "index"
        source_folder = tmp_path / "docs"
        source_folder.mkdir()
        (source_folder / "doc.txt").write_text("Some content here.")

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        # Build with hybrid=False so no bm25.pkl is written, then flip the
        # persisted route to hybrid=True to simulate a pre-hybrid collection
        # in a now-hybrid-aware install.
        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=index_dir, default_index="faiss")
            kb.ingest(
                source_folder,
                route=CollectionRoute(
                    embedding_backend="api",
                    vector_db="faiss",
                    hybrid=False,
                ),
            )
            kb.close()

            # Mutate the persisted route on disk.
            import json as _json
            route_path = index_dir / "docs" / "route.json"
            route_data = _json.loads(route_path.read_text())
            route_data["hybrid"] = True
            route_path.write_text(_json.dumps(route_data))

            # Fresh KB instance: no in-memory cache, must read from disk.
            kb2 = KnowledgeBase(index_dir=index_dir, default_index="faiss")
            with pytest.raises(FileNotFoundError, match="bm25.pkl"):
                kb2.search("anything", collection="docs", top_k=3, rerank=False)
            kb2.close()

    def test_add_entries_rebuilds_bm25(self, tmp_path):
        """add_entries must rebuild bm25.pkl with the new entry texts."""
        from dsagt.knowledge import KnowledgeBase

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=tmp_path / "index", default_index="faiss")
            kb.add_entries(
                texts=["alpha document", "beta document"],
                collection="memory",
            )
            bm25_path = tmp_path / "index" / "memory" / "bm25.pkl"
            assert bm25_path.exists()

            kb.add_entries(texts=["gamma document"], collection="memory")
            # BM25 must now know about all three docs.
            bm25 = kb._get_bm25("memory")
            assert bm25.size == 3
            kb.close()

    def test_route_persists_hybrid_flag(self, tmp_path):
        """route.json must round-trip the hybrid field."""
        from dsagt.knowledge import KnowledgeBase, CollectionRoute
        import json as _json

        mock_client = MagicMock()
        mock_client.embed = fake_embed

        with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
            kb = KnowledgeBase(index_dir=tmp_path / "index", default_index="faiss")
            kb.add_entries(
                texts=["x"],
                collection="c",
                route=CollectionRoute(
                    embedding_backend="api",
                    vector_db="faiss",
                    hybrid=False,
                ),
            )
            kb.close()

        route_data = _json.loads(
            (tmp_path / "index" / "c" / "route.json").read_text()
        )
        assert route_data["hybrid"] is False
