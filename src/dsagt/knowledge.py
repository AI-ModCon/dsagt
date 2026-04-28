"""
Knowledge base: semantic search over document collections.

Pluggable embedding backends (local sentence-transformers or OpenAI-compatible API)
and vector-DB backends (FAISS or ChromaDB) with per-collection routing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

import numpy as np

# llama_index is intentionally NOT imported at module top.
# It pulls in ~400 transitive submodules and adds ~8s to cold start, which
# matters because dsagt-run imports this module on every tool invocation
# but only ever needs the embedding/search code paths, never the parsers.
# llama_index is lazy-imported inside _chunk_file() and _get_parser().

from dsagt.observability import (
    kb_embed_span,
    kb_index_search_span,
    kb_rerank_span,
    llm_source,
    obs,
    traced,
)


CODE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".rs": "rust",
    ".go": "go", ".java": "java", ".cpp": "cpp", ".c": "c",
    ".rb": "ruby", ".php": "php",
}


class BaseEmbeddingClient(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (n_texts, dim), L2-normalized."""

    def close(self) -> None:
        pass


class BaseVectorIndex(ABC):
    """Common interface for all vector-index backends."""

    @abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        """Append pre-normalized embeddings."""

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, int_indices) arrays of length k."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist index to *directory*."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "BaseVectorIndex":
        """Load index from *directory*."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors stored."""


class LocalEmbeddingClient(BaseEmbeddingClient):
    """sentence-transformers, runs fully offline."""

    def __init__(
        self,
        model: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 256,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer
        self.batch_size = batch_size
        self._model = SentenceTransformer(model, device=device)
        logger.info("Loaded local embedding model: %s (dim=%d)",
                    model, self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)


# --- rate-limit retry helpers (used by APIEmbeddingClient) ----------------
#
# We can't rely on litellm.num_retries alone for embedding rate limits.  Lab
# LiteLLM proxies (notably PNNL's Azure-fronted instance) wrap upstream 429s
# as APIConnectionError carrying the rate-limit error in a JSON message body,
# which defeats litellm's exception-class-based retry detection.  Even when
# litellm does retry, its default exponential backoff caps at ~30s total
# across 5 attempts, while Azure's per-minute quota window asks for 60s+
# between retries.  So we own this retry layer ourselves and disable
# litellm's by passing ``num_retries=0`` per call.

_RETRY_AFTER_RE = re.compile(
    r"retry after (\d+(?:\.\d+)?)\s*seconds?", re.IGNORECASE,
)


def _extract_retry_after_seconds(message: str, default: float = 60.0) -> float:
    """Parse 'Please retry after N seconds' from upstream error messages.

    Honors the upstream's own hint when present so we don't undershoot the
    quota window.  Falls back to *default* when the message doesn't contain
    a parseable hint.
    """
    m = _RETRY_AFTER_RE.search(message)
    if m:
        return float(m.group(1))
    return default


def _is_retryable_embedding_error(exc: Exception) -> bool:
    """Decide whether an embedding exception is worth retrying.

    Detects rate-limit and transient errors across the various shapes
    LiteLLM exposes them in.  Authentication / bad-request errors are
    explicitly NOT retryable so we fail fast on misconfiguration.
    """
    # litellm is a hard dependency — broken install if this fails.
    import litellm

    if isinstance(exc, (
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.BadRequestError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
    )):
        return False

    if isinstance(exc, (
        litellm.exceptions.RateLimitError,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.Timeout,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.InternalServerError,
    )):
        return True

    # Some lab proxies wrap rate limits as plain Exceptions with the upstream
    # body in str(exc).  Fall back to message inspection.
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "ratelimiterror", "429", "throttl"))


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    """How long to sleep before the next retry attempt.

    Rate-limit errors get the upstream-suggested wait (or 60s default).
    Other transient errors get exponential backoff capped at 30s.
    """
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg or "throttl" in msg:
        return _extract_retry_after_seconds(str(exc), default=60.0)
    return min(2.0 ** attempt, 30.0)


class APIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI-compatible embedding client backed by LiteLLM.

    LiteLLM normalizes a wide set of providers (OpenAI, Azure, Bedrock,
    Vertex, Cohere, Voyage, Ollama, Together, etc.) behind a single
    ``embedding(...)`` call.  This client adds two things on top:

    1. **Manual batching** of large inputs into ``batch_size``-sized
       requests so a single rate-limit hit only loses one batch and the
       user can see per-batch progress in the logs.

    2. **Explicit rate-limit retry** with retry-after-aware backoff
       (see ``_embed_batch_with_retry``).  This is the layer that keeps
       large ``kb_ingest`` and ``dsagt-setup-kb`` runs alive in the face
       of TPM quotas — litellm's built-in retry isn't sufficient because
       lab proxies wrap upstream 429s in a way that defeats its
       exception-class detection, and its backoff is too short for the
       60-second quota windows Azure enforces.

    The model string passed to LiteLLM determines provider routing.  Bare
    model names get an ``openai_like/`` prefix so the lab proxy receives
    the model alias unchanged.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        batch_size: int = 100,
    ):
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small-project")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        # EMBEDDING_API_KEY is the canonical name; LLM_API_KEY/OPENAI_API_KEY
        # are legacy fallbacks from when the embedding endpoint shared the
        # LLM endpoint's auth.  When embeddings route through dsagt's proxy
        # (the default), this is a sentinel — the proxy holds the real key.
        self.api_key = (
            api_key
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.timeout = timeout
        self.batch_size = batch_size

        if not self.base_url:
            raise ValueError("Embedding API base URL required via argument or OPENAI_BASE_URL env var")
        if not self.api_key:
            raise ValueError("API key required via argument or LLM_API_KEY env var")

        # Always prefix with ``openai_like`` so LiteLLM dispatches to the
        # OpenAI-wire-protocol client pointed at our ``base_url`` — the rest
        # of DSAGT already assumes the embedding endpoint is OpenAI-compat,
        # so there's no valid case for a different provider here.
        #
        # ``openai_like`` (not ``openai``) is deliberate on two counts: (a)
        # ``openai`` matches LiteLLM's canonical model registry and silently
        # normalizes aliases like ``text-embedding-3-small-project`` down to
        # ``text-embedding-3-small``, which lab proxies then reject; (b)
        # ``openai_like`` is the documented escape hatch for "endpoint
        # speaks OpenAI but is not OpenAI" — it forwards the model string
        # verbatim, preserving HuggingFace-style names (``lbl/nomic-embed-text``,
        # ``nomic-ai/nomic-embed-text-v1``) whose slashes are part of the
        # model identifier, not a provider prefix.
        self._litellm_model = f"openai_like/{self.model}"

    @llm_source("embedding")
    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        # Single small input: one call, no batch logging.
        if len(texts) <= self.batch_size:
            return self._embed_batch_with_retry(texts)

        # Large input: split into batch_size groups so a rate-limit hit
        # only loses one batch and the user gets visible progress.
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(
                "Embedding batch %d/%d (%d texts)",
                batch_num, n_batches, len(batch),
            )
            batches.append(self._embed_batch_with_retry(batch))
        return np.vstack(batches)

    def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_attempts: int = 6,
    ) -> np.ndarray:
        """Embed one batch with explicit rate-limit and transient-error retry.

        See the module-level rate-limit retry helpers for the rationale.
        We pass ``num_retries=0`` to litellm so its built-in retry doesn't
        race with ours and we don't end up double-retrying with conflicting
        backoff strategies.
        """
        import litellm

        attempt = 0
        while True:
            attempt += 1
            try:
                response = litellm.embedding(
                    model=self._litellm_model,
                    input=texts,
                    api_base=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    num_retries=0,
                )
                break  # success
            except Exception as exc:
                if not _is_retryable_embedding_error(exc) or attempt >= max_attempts:
                    raise
                wait = _retry_wait_seconds(exc, attempt)
                logger.warning(
                    "Embedding error (attempt %d/%d). "
                    "Waiting %.0fs then retrying. Cause: %s",
                    attempt, max_attempts, wait, str(exc)[:200],
                )
                time.sleep(wait)

        # Response shape mirrors OpenAI: response.data is a list of objects
        # with .embedding (or ['embedding'] for dict access).
        sorted_data = sorted(
            response.data,
            key=lambda d: d["index"] if isinstance(d, dict) else d.index,
        )
        vectors = [
            d["embedding"] if isinstance(d, dict) else d.embedding
            for d in sorted_data
        ]
        return np.array(vectors, dtype=np.float32)

    def close(self) -> None:
        # No client object to close anymore — LiteLLM owns its own pool.
        pass

class FAISSIndex(BaseVectorIndex):
    """Inner-product FAISS flat index (cosine similarity on L2-normed vecs)."""

    _FILENAME = "index.faiss"

    def __init__(self, dim: int | None = None):
        self._dim = dim
        self._index = None  # lazy-init on first add

    def _ensure_init(self, dim: int) -> None:
        if self._index is None:
            import faiss
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray) -> None:
        self._ensure_init(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec.reshape(1, -1), k)
        return scores[0], indices[0]

    def save(self, directory: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(directory / self._FILENAME))

    @classmethod
    def load(cls, directory: Path) -> "FAISSIndex":
        import faiss
        obj = cls()
        obj._index = faiss.read_index(str(directory / cls._FILENAME))
        obj._dim = obj._index.d
        return obj

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal


class ChromaIndex(BaseVectorIndex):
    """ChromaDB-backed index.  Requires ``chromadb`` package."""

    _META_FILE = "chroma_ids.json"  # maps int position → chroma id

    def __init__(self, collection_name: str, persist_dir: Path | None = None):
        import chromadb

        self._name = collection_name
        self._persist_dir = persist_dir
        if persist_dir:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            # chromadb.Client() was removed in v0.4+; use EphemeralClient for in-memory
            self._client = chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._ids: list[str] = []  # positional id list for int-index mapping

    # ChromaDB stores ids internally; we maintain a positional list so that
    # returned integer indices are consistent with the chunk list.
    def add(self, embeddings: np.ndarray, metadatas: list[dict] | None = None) -> None:
        start = len(self._ids)
        new_ids = [str(start + i) for i in range(len(embeddings))]
        self._ids.extend(new_ids)
        embeddings_list = embeddings.tolist()

        # ChromaDB caps a single add() at a sqlite-configuration-dependent
        # batch size (typically ~5461).  Ingesting a large collection in one
        # shot throws InternalError, so chunk the call ourselves.  Stay well
        # under the cap for portability across chroma versions.
        batch_size = 5000
        for i in range(0, len(new_ids), batch_size):
            kwargs: dict = {
                "ids": new_ids[i:i + batch_size],
                "embeddings": embeddings_list[i:i + batch_size],
            }
            if metadatas is not None:
                kwargs["metadatas"] = metadatas[i:i + batch_size]
            self._col.add(**kwargs)

    def search(self, query_vec: np.ndarray, k: int, where: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self._ids))
        if k == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        query_kwargs: dict = {"query_embeddings": [query_vec.tolist()], "n_results": k}
        if where is not None:
            query_kwargs["where"] = where
        results = self._col.query(**query_kwargs)
        chroma_ids = results["ids"][0]
        distances = results["distances"][0]  # cosine distance (0=identical)
        scores = np.array([1.0 - d for d in distances], dtype=np.float32)
        indices = np.array([int(cid) for cid in chroma_ids], dtype=np.int64)
        return scores, indices

    def save(self, directory: Path) -> None:
        # ChromaDB PersistentClient auto-saves; just write id list for rebuild.
        (directory / self._META_FILE).write_text(json.dumps(self._ids))

    @classmethod
    def load(cls, directory: Path) -> "ChromaIndex":
        meta_path = directory / cls._META_FILE
        name = directory.name
        obj = cls(collection_name=name, persist_dir=directory)
        if meta_path.exists():
            obj._ids = json.loads(meta_path.read_text())
        return obj

    @property
    def size(self) -> int:
        return len(self._ids)


# map short names to embedding client constructors
EMBEDDER_REGISTRY: dict[str, type[BaseEmbeddingClient]] = {
    "local": LocalEmbeddingClient,
    "api": APIEmbeddingClient,
}

# map short names to vector-index constructors
VECTORINDEX_REGISTRY: dict[str, type[BaseVectorIndex]] = {
    "faiss": FAISSIndex,
    "chroma": ChromaIndex,
}


def _make_embedder(backend: str, **kwargs) -> BaseEmbeddingClient:
    backend = backend.lower()
    if backend not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedding backend '{backend}'. "
                         f"Choose from: {list(EMBEDDER_REGISTRY)}")
    return EMBEDDER_REGISTRY[backend](**kwargs)


def _make_index(backend: str, **kwargs) -> BaseVectorIndex:
    backend = backend.lower()
    if backend not in VECTORINDEX_REGISTRY:
        raise ValueError(f"Unknown vector-index backend '{backend}'. "
                         f"Choose from: {list(VECTORINDEX_REGISTRY)}")
    return VECTORINDEX_REGISTRY[backend](**kwargs)


@dataclass
class CollectionRoute:
    """
    Describes which embedding client and vector-index backend to use for a
    named collection.

    Parameters
    ----------
    embedding_backend : str
        Key in EMBEDDER_REGISTRY, e.g. ``"local"`` or ``"api"``.
    vector_db : str
        Key in VECTORINDEX_REGISTRY, e.g. ``"faiss"`` or ``"chroma"``.
    embedder_kwargs : dict
        Extra kwargs forwarded to the embedding client constructor.
    index_kwargs : dict
        Extra kwargs forwarded to the vector-index constructor.
    description : str
        Human-readable note shown in ``list_collections()`` when no
        DESCRIPTION.md file exists.
    """

    embedding_backend: str = "api"
    vector_db: str = "chroma"
    embedder_kwargs: dict = field(default_factory=dict)
    index_kwargs: dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CollectionRoute":
        return cls(**d)


class KnowledgeBase:
    """
    Collection-based document retrieval with pluggable embedder & vector-DB routing.

    Each collection lives in ``<index_dir>/<collection_name>/`` and may use a
    *different* embedding model and vector-index backend.  Routing is controlled
    by a dictionary of :class:`CollectionRoute` objects keyed on collection name.

    Quick-start
    -----------
    .. code-block:: python

        from knowledge_base import KnowledgeBase, CollectionRoute

        kb = KnowledgeBase(
            index_dir="./kb_store",
            routes={
                # GPU-heavy research docs → local model + FAISS
                "research": CollectionRoute(
                    embedding_backend="local",
                    vector_db="faiss",
                    embedder_kwargs={"model": "BAAI/bge-base-en-v1.5"},
                ),
                # Fast live ingestion → API + Chroma
                "support": CollectionRoute(
                    embedding_backend="api",
                    vector_db="chroma",
                    embedder_kwargs={"model": "text-embedding-3-small"},
                ),
            },
        )

        kb.ingest("./docs/research", "research")
        results = kb.search("transformer architecture", "research")

    Embedding backends
    ------------------
    ``"local"``
        sentence-transformers, no network required.
    ``"api"``
        OpenAI-compatible REST API.  Reads ``LLM_API_KEY`` env var.

    Vector-index backends
    ---------------------
    ``"chroma"``
        ChromaDB HNSW index (default).  Supports metadata filtering
        and incremental updates.
    ``"faiss"``
        Flat inner-product index.  Faster for small static collections
        but no metadata filtering.
    """

    FILE_TYPES = [
        "pdf", "md", "rst", "txt", "py", "docx",
        "json", "yaml", "yml",
        # Packaging metadata: agents reading this index need to know which
        # version of a library to install when registering tools that
        # depend on it.  pyproject.toml is the modern standard; setup.cfg
        # is still common in older codebases.
        "toml", "cfg",
    ]
    _ROUTE_FILE = "route.json"

    def __init__(
        self,
        index_dir: str | Path,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        default_rerank: bool = False,
        # Global default route (used when collection has no specific route)
        default_embedder: str | None = None,
        default_index: str | None = None,
        embedder_kwargs: dict | None = None,
        # Per-collection routing registry
        routes: dict[str, CollectionRoute] | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model
        self.default_rerank = default_rerank

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Build default route from explicit kwargs (config-driven).
        # No env var reads — callers pass config values through.
        self._default_route = CollectionRoute(
            embedding_backend=default_embedder or "api",
            vector_db=default_index or "chroma",
            embedder_kwargs=embedder_kwargs or {},
        )

        # Per-collection route registry
        self._routes: dict[str, CollectionRoute] = routes or {}

        # Shared embedder cache: embedder_key → client instance
        # Key is "<embedding_backend>|<sorted embedder_kwargs>" so identical configs share one client.
        self._embedder_cache: dict[str, BaseEmbeddingClient] = {}

        # Collection runtime cache: name → (BaseVectorIndex, list[dict])
        self._cache: dict[str, tuple[BaseVectorIndex, list[dict]]] = {}

        # Per-file-type parser cache. Constructing a CodeSplitter loads the
        # tree-sitter language definition (~25ms each), so a 300-file ingest
        # without this cache pays 7-8 seconds in parser construction alone.
        # Parsers are stateless once built; safe to share across files.
        self._parsers: dict[str, Any] = {}

        # Per-ingest counter for files that _chunk_file skipped due to
        # read/parse errors.  ingest()/append() reset this before the loop
        # and surface it in the result dict so users notice when a non-zero
        # number of files are silently being dropped.
        self._chunk_skip_count: int = 0

        self._reranker = None

    # route management

    def register_route(self, collection: str, route: CollectionRoute) -> None:
        """Register (or update) a routing rule for *collection*.

        Updates the in-memory routing table AND persists ``route.json``
        to disk so the mapping survives server restarts.
        """
        self._routes[collection] = route
        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)
        (coll_dir / self._ROUTE_FILE).write_text(json.dumps(route.to_dict(), indent=2))
        logger.info("Registered route for '%s': embedder=%s index=%s",
                    collection, route.embedding_backend, route.vector_db)

    def _get_route(self, collection: str) -> CollectionRoute:
        return self._routes.get(collection, self._default_route)

    def _get_embedder(self, route: CollectionRoute) -> BaseEmbeddingClient:
        """Return (possibly cached) embedder matching *route*.

        Routes carry per-collection overrides (e.g. a specific model name);
        credentials (api_key, base_url) live on the kb's default route.
        Merge so explicit routes inherit defaults unless they override.
        """
        merged = {**self._default_route.embedder_kwargs, **route.embedder_kwargs}
        key = f"{route.embedding_backend}|{sorted(merged.items())}"
        if key not in self._embedder_cache:
            self._embedder_cache[key] = _make_embedder(
                route.embedding_backend, **merged
            )
        return self._embedder_cache[key]

    @property
    def collections(self) -> list[str]:
        return [p.name for p in self.index_dir.iterdir() if p.is_dir() and
                ((p / "index.faiss").exists() or (p / "chroma_ids.json").exists())]

    def list_collections(self) -> list[dict]:
        result = []
        for name in self.collections:
            coll_dir = self.index_dir / name
            desc_path = coll_dir / "DESCRIPTION.md"

            # Ensure the persisted route is loaded into _routes so we always
            # report the actual model/backend, not the server default.
            if name not in self._routes:
                persisted = self._load_route(name)
                if persisted:
                    self._routes[name] = persisted

            route = self._routes.get(name, self._default_route)
            description = (
                desc_path.read_text() if desc_path.exists()
                else route.description
            )
            result.append({
                "name": name,
                "description": description,
                "embedding_backend": route.embedding_backend,
                "embedding_model": route.embedder_kwargs.get("model", "default"),
                "vector_db": route.vector_db,
            })
        return result

    @traced(
        "kb.ingest",
        capture=["collection_name"],
        extract_return={
            "n_files": lambda r: r.get("files"),
            "n_chunks": lambda r: r.get("chunks"),
        },
    )
    def ingest(
        self,
        folder: str | Path,
        collection_name: str | None = None,
        file_types: list[str] | None = None,
        route: CollectionRoute | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict:
        """
        Ingest *folder* as a new collection.

        Parameters
        ----------
        folder : path
            Source directory.
        collection_name : str, optional
            Defaults to folder name.
        file_types : list[str], optional
            Extensions to include (without leading dot).
        route : CollectionRoute, optional
            Override routing for this collection.  Persisted to disk so
            subsequent ``search`` / ``append`` calls use the same backend.
        exclude_patterns : list[str], optional
            Glob-style patterns matched (via :func:`fnmatch.fnmatch`)
            against each file's path *relative to* ``folder``.  Any file
            whose relative path matches any pattern is skipped.  Patterns
            are checked against both the full relative path and the
            basename, so ``"tests/*"`` excludes a top-level tests dir and
            ``"test_*.py"`` excludes test files anywhere in the tree.
            Useful for skipping tests, build artifacts, and private modules
            without inflating the embed cost of large libraries.
        """
        folder = Path(folder)
        collection = collection_name or folder.name
        file_types = file_types or self.FILE_TYPES

        obs.set_inputs({
            "folder": str(folder),
            "collection": collection,
            "file_types": file_types,
        })

        # Set route in memory so _get_embedder resolves during the build.
        # Persisted to disk via register_route after the index is built.
        if route is not None:
            self._routes[collection] = route
        active_route = self._get_route(collection)

        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)

        # Record source folder so the MCP server can detect re-ingests vs. conflicts.
        (coll_dir / "source.txt").write_text(str(folder.resolve()))

        desc_src = folder / "DESCRIPTION.md"
        if desc_src.exists():
            (coll_dir / "DESCRIPTION.md").write_text(desc_src.read_text())

        files = self._collect_files(folder, file_types, exclude_patterns)
        logger.info("Found %d files to process", len(files))

        # Reset the per-ingest skip counter; _chunk_file bumps it on
        # read/parse failure.  We surface the result in the return dict
        # so callers (and the user looking at dsagt-setup-kb output)
        # notice if a non-trivial number of files were silently dropped.
        self._chunk_skip_count = 0
        chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        n_skipped = self._chunk_skip_count
        logger.info(
            "Created %d chunks (skipped %d files)", len(chunks), n_skipped,
        )

        if not chunks:
            return {
                "collection": collection,
                "files": 0,
                "chunks": 0,
                "skipped_files": n_skipped,
            }

        embedder = self._get_embedder(active_route)
        with kb_embed_span(active_route.embedding_backend,
                           active_route.embedder_kwargs.get("model"), len(chunks)):
            embeddings = self._normalize(embedder.embed([c["text"] for c in chunks]))

        index = _make_index(active_route.vector_db,
                            **self._index_init_kwargs(active_route, coll_dir))
        index.add(embeddings)
        index.save(coll_dir)

        # Persist route AFTER index is built — guarantees route.json always
        # reflects what was actually used, never overwritten by a racing job.
        self.register_route(collection, active_route)

        with open(coll_dir / "chunks.jsonl", "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        self._cache[collection] = (index, chunks)
        return {
            "collection": collection,
            "files": len(files),
            "chunks": len(chunks),
            "skipped_files": n_skipped,
        }

    @traced(
        "kb.append",
        capture=["collection"],
        extract_return={
            "n_files": lambda r: r.get("files"),
            "n_chunks": lambda r: r.get("chunks_added"),
        },
    )
    def append(
        self,
        collection: str,
        paths: list[str | Path],
        file_types: list[str] | None = None,
    ) -> dict:
        """Append documents to an existing collection."""
        file_types = file_types or self.FILE_TYPES

        obs.set_inputs({
            "collection": collection,
            "n_paths": len(paths),
            "paths_preview": [str(p) for p in paths[:5]],
        })

        index, existing_chunks = self._load(collection)
        active_route = self._get_route(collection)
        coll_dir = self.index_dir / collection

        files: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(f for ext in file_types for f in p.glob(f"**/*.{ext}"))
            else:
                logger.warning("Path not found, skipping: %s", p)

        if not files:
            return {"collection": collection, "files": 0, "chunks_added": 0,
                    "total_chunks": len(existing_chunks), "skipped_files": 0}

        # Reset and capture per-file skip count, same pattern as ingest().
        self._chunk_skip_count = 0
        new_chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        n_skipped = self._chunk_skip_count
        if not new_chunks:
            return {"collection": collection, "files": len(files),
                    "chunks_added": 0, "total_chunks": len(existing_chunks),
                    "skipped_files": n_skipped}

        embedder = self._get_embedder(active_route)
        with kb_embed_span(active_route.embedding_backend,
                           active_route.embedder_kwargs.get("model"), len(new_chunks)):
            embeddings = self._normalize(embedder.embed([c["text"] for c in new_chunks]))
        index.add(embeddings)
        index.save(coll_dir)

        with open(coll_dir / "chunks.jsonl", "a") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk) + "\n")

        all_chunks = existing_chunks + new_chunks
        self._cache[collection] = (index, all_chunks)
        return {
            "collection": collection,
            "files": len(files),
            "chunks_added": len(new_chunks),
            "total_chunks": len(all_chunks),
            "skipped_files": n_skipped,
        }

    @traced(
        "kb.add_entries",
        capture=["collection"],
        extract_return={"n_entries": lambda r: r.get("entries_added")},
    )
    def add_entries(
        self,
        texts: list[str],
        collection: str,
        metadatas: list[dict] | None = None,
        route: CollectionRoute | None = None,
        return_embeddings: bool = False,
    ) -> dict:
        """Add pre-formed text entries with optional metadata to a collection.

        Unlike ``ingest``/``append``, this skips document parsing and chunking.
        Each text is embedded and stored as-is.  Used by episodic memory,
        tool_executions, and other structured entry types that produce their
        own text representations rather than ingesting raw documents.

        Parameters
        ----------
        texts : list[str]
            Text content for each entry (will be embedded).
        collection : str
            Target collection name (created if it doesn't exist).
        metadatas : list[dict], optional
            Per-entry metadata dicts.  On ChromaDB-backed collections these
            are stored as native metadata for ``where`` filtering.  On all
            backends they are merged into the chunk metadata in chunks.jsonl.
        route : CollectionRoute, optional
            Override routing for this collection (persisted to disk).
        return_embeddings : bool, optional
            If True, include the freshly-computed embeddings in the result
            dict under the ``"embeddings"`` key.  Callers that need the
            embeddings for downstream work (e.g. centroid-based outlier
            detection in memory extraction) should set this so they can
            avoid a second round-trip to the embedding API.

        Returns
        -------
        dict
            ``{collection, entries_added, total_entries}`` plus
            ``"embeddings"`` (numpy ndarray) if ``return_embeddings=True``.
        """
        obs.set_inputs({
            "collection": collection,
            "n_entries": len(texts),
            "texts_preview": [t[:200] for t in texts[:3]],
        })

        if not texts:
            result = {"collection": collection, "entries_added": 0, "total_entries": 0}
            if return_embeddings:
                result["embeddings"] = np.array([], dtype=np.float32)
            return result

        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)

        if route is not None:
            self._routes[collection] = route
        active_route = self._get_route(collection)

        embedder = self._get_embedder(active_route)
        with kb_embed_span(active_route.embedding_backend,
                           active_route.embedder_kwargs.get("model"), len(texts)):
            embeddings = self._normalize(embedder.embed(texts))

        # Load existing or create new index
        if (coll_dir / "chunks.jsonl").exists():
            index, existing_chunks = self._load(collection)
        else:
            index = _make_index(
                active_route.vector_db,
                **self._index_init_kwargs(active_route, coll_dir),
            )
            existing_chunks = []

        # Build chunk dicts (consistent with chunks.jsonl format)
        new_chunks = []
        for i, text in enumerate(texts):
            entry_meta = metadatas[i] if metadatas else {}
            chunk = {
                "text": text,
                "metadata": {
                    "collection": collection,
                    "source_file": "entry",
                    "chunk_index": len(existing_chunks) + i,
                    **entry_meta,
                },
            }
            new_chunks.append(chunk)

        # Store embeddings (with metadata on ChromaDB)
        if active_route.vector_db == "chroma" and metadatas is not None:
            index.add(embeddings, metadatas=metadatas)
        else:
            index.add(embeddings)

        index.save(coll_dir)
        self.register_route(collection, active_route)

        # Append to chunks.jsonl
        with open(coll_dir / "chunks.jsonl", "a") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk) + "\n")

        all_chunks = existing_chunks + new_chunks
        self._cache[collection] = (index, all_chunks)

        result = {
            "collection": collection,
            "entries_added": len(texts),
            "total_entries": len(all_chunks),
        }
        if return_embeddings:
            result["embeddings"] = embeddings
        return result

    def embed_texts(self, texts: list[str], collection: str) -> np.ndarray:
        """Embed texts using the embedder configured for a collection.

        Returns L2-normalized float32 array of shape (n_texts, dim).
        """
        route = self._get_route(collection)
        embedder = self._get_embedder(route)
        return self._normalize(embedder.embed(texts))

    @traced("kb.search", capture=["collection", "top_k", "rerank"])
    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        rerank: bool | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Search *collection* with *query*.

        Parameters
        ----------
        rerank : bool, optional
            Use cross-encoder reranking.  Defaults to ``self.default_rerank``
            (set from ``knowledge.rerank`` in dsagt_config.yaml).
        where : dict, optional
            ChromaDB ``where`` filter clause.  Only effective on ChromaDB-backed
            collections; silently ignored for FAISS collections.
        """
        if rerank is None:
            rerank = self.default_rerank

        obs.set_inputs({"query": query, "collection": collection, "top_k": top_k})

        index, chunks = self._load(collection)
        active_route = self._get_route(collection)
        embedder = self._get_embedder(active_route)

        with kb_embed_span(active_route.embedding_backend,
                           active_route.embedder_kwargs.get("model"), 1):
            query_emb = self._normalize(embedder.embed([query]))[0]

        search_k = min(top_k * 10 if rerank else top_k, len(chunks))
        filtered = where is not None and active_route.vector_db == "chroma"
        with kb_index_search_span(active_route.vector_db, search_k, filtered):
            if filtered:
                scores, indices = index.search(query_emb, search_k, where=where)
            else:
                scores, indices = index.search(query_emb, search_k)

        results = [
            {"chunk": chunks[i], "score": float(scores[j])}
            for j, i in enumerate(indices) if i >= 0
        ]

        if rerank and results:
            with kb_rerank_span(self.rerank_model, len(results)):
                final = self._rerank(query, results, top_k)
            obs.set("hits", len(final))
            obs.set_outputs({"hits": len(final), "top_texts": [r["chunk"].get("text", "")[:200] for r in final[:3]]})
            return final

        final = results[:top_k]
        obs.set("hits", len(final))
        obs.set_outputs({"hits": len(final), "top_texts": [r["chunk"].get("text", "")[:200] for r in final[:3]]})
        return final

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.where(norms > 0, norms, 1)

    def _index_init_kwargs(self, route: CollectionRoute, coll_dir: Path) -> dict:
        """Extra kwargs needed by some index constructors at init time."""
        if route.vector_db == "chroma":
            return {"collection_name": coll_dir.name,
                    "persist_dir": coll_dir, **route.index_kwargs}
        return dict(route.index_kwargs)

    def _load_route(self, collection: str) -> CollectionRoute | None:
        route_path = self.index_dir / collection / self._ROUTE_FILE
        if route_path.exists():
            return CollectionRoute.from_dict(json.loads(route_path.read_text()))
        return None

    def _load(self, name: str) -> tuple[BaseVectorIndex, list[dict]]:
        """Load collection index + chunks (cached in memory)."""
        if name in self._cache:
            return self._cache[name]

        coll_dir = self.index_dir / name
        if not coll_dir.exists():
            raise ValueError(f"Collection '{name}' not found")

        # Restore persisted route if not already registered
        if name not in self._routes:
            persisted = self._load_route(name)
            if persisted:
                self._routes[name] = persisted

        active_route = self._get_route(name)

        # Pick correct loader
        if active_route.vector_db == "chroma":
            index = ChromaIndex.load(coll_dir)
        else:
            index = FAISSIndex.load(coll_dir)

        with open(coll_dir / "chunks.jsonl") as f:
            chunks = [json.loads(line) for line in f]

        self._cache[name] = (index, chunks)
        return index, chunks

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.rerank_model, max_length=512)
        pairs = [[query, r["chunk"]["text"]] for r in results]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [{**r, "rerank_score": float(s)} for r, s in ranked[:top_k]]

    def _collect_files(
        self,
        folder: Path,
        file_types: list[str],
        exclude_patterns: list[str] | None,
    ) -> list[Path]:
        """Walk *folder* for files matching *file_types*, applying optional
        glob exclusions.

        Pure function relative to filesystem state — no caching, no
        side-effecting attributes.  Extracted from ``ingest()`` so file
        discovery and exclusion logic can be tested in isolation without
        spinning up an embedder, an index, or a chunker.

        Patterns are checked against the relative path, the basename, and
        each individual path segment, so ``"tests"`` excludes any file
        whose path contains a ``tests/`` directory regardless of depth.
        """
        from fnmatch import fnmatch

        all_files = [
            f for ext in file_types for f in folder.glob(f"**/*.{ext}")
        ]

        if not exclude_patterns:
            return all_files

        def _excluded(f: Path) -> bool:
            rel = str(f.relative_to(folder))
            name = f.name
            return any(
                fnmatch(rel, pat) or fnmatch(name, pat)
                or any(fnmatch(part, pat) for part in Path(rel).parts)
                for pat in exclude_patterns
            )

        kept = [f for f in all_files if not _excluded(f)]
        n_excluded = len(all_files) - len(kept)
        if n_excluded:
            logger.info(
                "Excluded %d/%d files via %d pattern(s)",
                n_excluded, len(all_files), len(exclude_patterns),
            )
        return kept

    def _chunk_file(self, path: Path, collection: str) -> Iterator[dict]:
        # Lazy import — see the module-top comment about cold-start cost.
        from llama_index.core import SimpleDirectoryReader

        # Per-file read/parse failures are kept as soft failures (count
        # surfaced in ingest()'s return dict) rather than aborting the
        # whole ingest, because real-world directories like cloned
        # upstream repos contain occasional unreadable files (binary
        # blobs misnamed .txt, malformed UTF-8) that shouldn't kill an
        # ingest of thousands of files.  Callers see the skip count
        # and can investigate if it's suspiciously high.
        try:
            docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        except (FileNotFoundError, IOError, ValueError) as e:
            logger.warning("Could not read %s: %s", path, e)
            self._chunk_skip_count += 1
            return
        if not docs:
            return
        file_type = path.suffix.lower()
        parser = self._get_parser(file_type)
        try:
            nodes = parser.get_nodes_from_documents(docs)
        except (ValueError, RuntimeError) as e:
            logger.warning("Could not parse %s: %s", path, e)
            self._chunk_skip_count += 1
            return
        for i, node in enumerate(nodes):
            text = node.get_content().strip()
            if not text:
                continue
            yield {
                "id": hashlib.sha256(f"{path}:{i}:{text[:100]}".encode()).hexdigest()[:16],
                "text": text,
                "metadata": {
                    "source_file": str(path),
                    "collection": collection,
                    "chunk_index": i,
                    "file_type": file_type,
                },
            }

    def _get_parser(self, file_type: str):
        """Return a cached parser for *file_type*, building it on first use.

        Parsers are stateless after construction.  ``CodeSplitter`` in
        particular loads a tree-sitter language definition on every
        construction (~25ms), so a large ingest used to pay this cost per
        file; now it pays once per file type.
        """
        cached = self._parsers.get(file_type)
        if cached is not None:
            return cached

        # Lazy import — see the module-top comment about cold-start cost.
        from llama_index.core.node_parser import (
            CodeSplitter,
            MarkdownNodeParser,
            SentenceSplitter,
        )

        if file_type in CODE_LANGUAGES:
            parser = CodeSplitter(
                language=CODE_LANGUAGES[file_type],
                chunk_lines=40,
                chunk_lines_overlap=10,
                max_chars=self.chunk_size * 4,
            )
        elif file_type == ".md":
            parser = MarkdownNodeParser()
        else:
            parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        self._parsers[file_type] = parser
        return parser

    def close(self) -> None:
        for client in self._embedder_cache.values():
            client.close()
        self._embedder_cache.clear()
        self._parsers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
