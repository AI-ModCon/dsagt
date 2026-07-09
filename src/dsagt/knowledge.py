"""
Knowledge base — hybrid semantic retrieval over document collections, and the
shared substrate every other DSAGT capability searches against.

A :class:`KnowledgeBase` holds one or more vector stores (each a single embedder
over many collections) and fuses their results by Reciprocal Rank Fusion (RRF).
Fusion runs at two levels: *within* a collection, dense vector similarity is
blended with a BM25 sparse leg, so exact identifiers and error strings that dense
embeddings under-rank still surface; *across* collections and stores, per-
collection rankings are fused by rank alone — letting results from different
embedding spaces combine without any cross-space score normalization.  The same
surface backs document/domain-knowledge retrieval, explicit and episodic memory,
tool-usage provenance, and skills discovery.  Design-wise it keeps maintenance
bounded: one embedder per store, a bring-your-own external store is just another
:class:`VectorStore` subclass added to the list, and the heavy document parsers
are imported lazily so only actual ingestion pays for them, not server startup.

The niche it fills
------------------
General coding agents have settled retrieval into two modes this KB deliberately
does not compete with — agentic ``grep`` / tool search over *code* (Claude Code,
Cursor, Windsurf et al. dropped codebase vector indexing outright) and *web
search* for open-ended questions.  Neither serves **bounded, domain-scoped,
vetted corpora of highly technical prose** — protocols, API references, domain
knowledge, tool-use provenance — which is exactly where hybrid dense+sparse
retrieval still wins and where general agents otherwise fall back to untrusted
web search.  That curated, tagged, auditable, hit-it-*first* retrieval is the
niche DSAGT fills to supplement general agents on specialized, highly technical
task loads.  The design above is the 2026 best-practice answer to the two
documented failure modes: hybrid search (BM25 recovers the exact identifiers
dense vectors miss) and per-collection domain scoping (the antidote to
"vector-search dilution", where pure-vector accuracy collapses on large
heterogeneous corpora).

References:
  - Code agents drop vector indexing for agentic grep —
    https://www.mindstudio.ai/blog/is-rag-dead-what-ai-agents-use-instead ;
    https://vadim.blog/claude-code-no-indexing/
  - Hybrid BM25 + dense + reranking, the highest-impact RAG upgrade —
    https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026
  - "When More Documents Hurt RAG": vector-search dilution, fixed by
    domain-scoped retrieval — https://arxiv.org/abs/2606.11350
  - 2026 enterprise knowledge management — trust / traceability favor curated
    KBs over web search —
    https://windowsforum.com/threads/2026-enterprise-ai-knowledge-management-from-search-to-governed-agent-workflows.410816/

Class map — every edge is ``<branch>─<rel> Class``, where ``<rel>`` is one of
``◇`` holds · ``◆`` owns · ``▷`` inherits  (``*`` = one per collection)::

    KnowledgeBase                   federation infra: ingest pipeline +
    │                               cross-collection RRF (_rrf_across)
    └─◇ VectorStore «abstract» 1..*   BYO adapter port: one embedder, many
        │                             collections; single-collection add/search
        └─▷ ChromaVectorStore         local Chroma; hybrid dense+BM25 per
            │                         collection, fused by _rrf_merge
            ├─◇ Embedder «abstract» 1   text → vectors; .create() factory
            │   │                       (one per store; may be shared)
            │   ├─▷ LocalEmbedder       sentence-transformers, offline
            │   └─▷ APIEmbedder         OpenAI /v1/embeddings + rate-limit retry
            ├─◆ ChromaIndex  *          dense leg   (add/search/save/load)
            └─◆ BM25Index    *          sparse leg  (build/search/save/load)

    free fns: _rrf_merge · _rrf_across (rank fusion, used by store + KB)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# Silence noisy warnings, progress bars from a host of libraries
import warnings as _warnings

for _noisy in ("pypdf", "pypdf2", "PyPDF2", "pdfminer", "fontTools"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

for _noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "transformers_modules",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_warnings.filterwarnings(
    "ignore",
    message="The `get_sentence_embedding_dimension` method has been renamed",
    category=FutureWarning,
)

_warnings.filterwarnings(
    "ignore",
    message=r".*unauthenticated requests to the HF Hub.*",
)

import numpy as np

# llama_index is intentionally NOT imported at module top: it pulls in ~400
# transitive submodules and adds ~8s to import.  Only document *ingest*
# (kb_ingest / the `dsagt init` KB build) needs the parsers — search, memory,
# provenance indexing, and skills never do — so importing it at module top
# would add ~8s of blocking load to every MCP-server startup (and to
# post-session extraction) for a parser most sessions never invoke.
# Lazy-imported inside _chunk_file() / _get_parser() so only ingest pays for it.

from dsagt.observability import (
    kb_embed_span,
    kb_index_search_span,
    kb_rerank_span,
    obs,
    traced,
)

CODE_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".php": "php",
}


# ===========================================================================
# Embedder — one per VectorStore
# ===========================================================================


class Embedder(ABC):
    """Common interface for all embedding backends.

    A stateless-after-construction leaf: a single :class:`VectorStore` has-a one
    ``Embedder``, and the *same* instance can be shared by several stores (e.g. a
    Chroma store and a FAISS store on one embedding space).
    """

    #: Short backend tag used for tracing spans ("local" / "api").
    backend: str = "unknown"
    #: Model identifier, for span labelling / collection listing.  Subclasses
    #: set this in ``__init__``.
    model: str | None = None

    @classmethod
    def create(
        cls,
        backend: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ) -> "Embedder":
        """Factory: construct the one embedder for a store, with explicit args.

        The api-vs-local selector that stays after per-collection embedder
        routing was removed: a store fixes a single embedder at construction.
        Each backend pulls only the parameters it uses — no ``**kwargs`` splat.
        """
        backend = (backend or "api").lower()
        if backend == "local":
            return LocalEmbedder(model=model, device=device)
        if backend == "api":
            return APIEmbedder(model=model, base_url=base_url, api_key=api_key)
        raise ValueError(
            f"Unknown embedding backend {backend!r}. Choose from: local, api"
        )

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (n_texts, dim), L2-normalized."""

    def close(self) -> None:
        pass


class LocalEmbedder(Embedder):
    """sentence-transformers, runs fully offline."""

    backend = "local"

    #: Default local model.  ``bge-small-en-v1.5`` (33M params, ~130 MB
    #: on disk, ~250 MB resident) is ~3× faster and ~3× smaller than
    #: ``bge-base`` for ~2 nDCG@10 points lower MTEB retrieval score — a
    #: hard-to-notice difference for typical DSAGT KB sizes (single-digit
    #: thousands of chunks).
    #: Override via ``embedding.model`` in ``.dsagt/config.yaml`` (e.g.
    #: ``BAAI/bge-large-en-v1.5`` for higher quality at the cost of
    #: ~10× memory and ~5× CPU).
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 256,
        device: str | None = None,
    ):
        import sys
        from sentence_transformers import SentenceTransformer

        model = model or self.DEFAULT_MODEL
        self.model = model
        self.batch_size = batch_size
        # Probe the HF cache for the model's config.  If hit, load with
        # ``local_files_only=True`` so SentenceTransformer skips the
        # ETag-validation HEAD requests it would otherwise issue against
        # huggingface.co — those round-trips are anonymous (HF_TOKEN
        # isn't propagated into MCP-server env blocks for cline /
        # codex), so they trigger an "unauthenticated requests" warning
        # surfaced under the agent's debug stream.  Cache miss path stays
        # online so first-run downloads still work.
        try:
            from huggingface_hub import try_to_load_from_cache

            cache_hit = try_to_load_from_cache(model, "config.json") is not None
        except Exception:
            cache_hit = False
        if cache_hit:
            self._model = SentenceTransformer(
                model,
                device=device,
                local_files_only=True,
            )
        else:
            print(
                f"  Downloading {model} from HuggingFace "
                "(set HF_TOKEN for faster throughput)...",
                file=sys.stderr,
                flush=True,
            )
            self._model = SentenceTransformer(model, device=device)
        # Belt-and-suspenders: dsagt/__init__.py sets OMP_NUM_THREADS /
        # MKL_NUM_THREADS env vars before heavy imports, but PyTorch
        # also has its own intra-op thread count that ignores those env
        # vars in some configurations.  Cap explicitly here.
        try:
            import torch

            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        except Exception:  # noqa: BLE001 — best-effort cap, never fatal
            pass
        logger.info(
            "Loaded local embedding model: %s (dim=%d)",
            model,
            self._model.get_embedding_dimension(),
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        # Show the progress bar only for non-trivial inputs so single-query
        # search calls (kb.search → embed([query])) stay silent while large
        # ingest runs surface tqdm progress.
        show_bar = len(texts) > self.batch_size
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_bar,
            normalize_embeddings=True,
        ).astype(np.float32)


# --- rate-limit retry helpers (used by APIEmbedder) ----------------
#
# The API embedder talks to an OpenAI-compatible ``/v1/embeddings`` endpoint
# over httpx.  We own the retry layer because lab gateways (notably PNNL's
# Azure-fronted instance) return upstream 429s with the quota window in the
# body and ask for 60s+ between retries — longer than a generic exponential
# backoff would wait.  Rate-limit / transient errors retry with a
# retry-after-aware wait; auth / bad-request errors fail fast.

_RETRY_AFTER_RE = re.compile(
    r"retry after (\d+(?:\.\d+)?)\s*seconds?",
    re.IGNORECASE,
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

    Retries rate-limit (429) and transient server / network errors;
    authentication and bad-request errors are explicitly NOT retryable so
    we fail fast on misconfiguration.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        # 408 request timeout, 425 too early, 429 rate limit, 5xx server.
        return code in (408, 425, 429, 500, 502, 503, 504)

    # Connect / read timeouts and other transport failures are transient.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    # Some gateways wrap rate limits in a plain Exception with the upstream
    # body in str(exc).  Fall back to message inspection.
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "ratelimiterror", "429", "throttl"))


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    """How long to sleep before the next retry attempt.

    Honors a ``Retry-After`` header (or a retry-after hint in the body) on
    rate-limit responses; other transient errors get exponential backoff
    capped at 30s.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        if exc.response.status_code == 429:
            return _extract_retry_after_seconds(exc.response.text, default=60.0)
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg or "throttl" in msg:
        return _extract_retry_after_seconds(str(exc), default=60.0)
    return min(2.0**attempt, 30.0)


class APIEmbedder(Embedder):
    """Embedding client for an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Talks to the endpoint directly over httpx — no provider-abstraction
    layer.  The model string is sent verbatim in the request body, so
    gateway aliases (``text-embedding-3-small-project``) and HuggingFace-
    style names with slashes (``nomic-ai/nomic-embed-text-v1``) pass
    through unchanged.  Two things on top of the raw call:

    1. **Manual batching** of large inputs into ``batch_size``-sized
       requests so a single rate-limit hit only loses one batch and the
       user can see per-batch progress in the logs.

    2. **Explicit rate-limit retry** with retry-after-aware backoff
       (see ``_embed_batch_with_retry``) — what keeps large ``kb_ingest``
       and ``dsagt init`` KB builds alive against the 60-second quota
       windows lab gateways enforce.
    """

    backend = "api"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        batch_size: int = 100,
    ):
        import httpx

        self.model = model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small-project"
        )
        self.base_url = (
            base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        )
        # EMBEDDING_API_KEY is the canonical name; LLM_API_KEY/OPENAI_API_KEY
        # are accepted as fallbacks for setups where the embedding endpoint
        # shares the LLM endpoint's auth.
        self.api_key = (
            api_key
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.timeout = timeout
        self.batch_size = batch_size

        if not self.base_url:
            raise ValueError(
                "Embedding API base URL required via argument or "
                "EMBEDDING_BASE_URL / OPENAI_BASE_URL env var"
            )
        if not self.api_key:
            raise ValueError(
                "API key required via argument or EMBEDDING_API_KEY env var"
            )

        # ``base_url`` is the OpenAI-style root (typically ending in ``/v1``);
        # the embeddings route hangs off it.
        self._embeddings_url = self.base_url.rstrip("/") + "/embeddings"
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        # Single small input: one call, no batch logging.
        if len(texts) <= self.batch_size:
            return self._embed_batch_with_retry(texts)

        # Large input: split into batch_size groups so a rate-limit hit
        # only loses one batch and the user gets visible progress.
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(
                "Embedding batch %d/%d (%d texts)",
                batch_num,
                n_batches,
                len(batch),
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
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._client.post(
                    self._embeddings_url, json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                break  # success
            except Exception as exc:
                if not _is_retryable_embedding_error(exc) or attempt >= max_attempts:
                    raise
                wait = _retry_wait_seconds(exc, attempt)
                logger.warning(
                    "Embedding error (attempt %d/%d). "
                    "Waiting %.0fs then retrying. Cause: %s",
                    attempt,
                    max_attempts,
                    wait,
                    str(exc)[:200],
                )
                time.sleep(wait)

        # Response shape is OpenAI's: ``data`` is a list of ``{"index": i,
        # "embedding": [...]}`` objects, not guaranteed to be in input order.
        sorted_data = sorted(data["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in sorted_data]
        return np.array(vectors, dtype=np.float32)

    def close(self) -> None:
        self._client.close()


# ===========================================================================
# ChromaIndex — single-collection dense vector wrapper (ChromaVectorStore's leg)
# ===========================================================================


class ChromaIndex:
    """ChromaDB-backed dense index for ONE collection.  Requires ``chromadb``.

    A plain helper owned by :class:`ChromaVectorStore` — not a pluggable
    "vector-DB backend".  Stores cosine-space HNSW vectors plus a positional
    id list so returned integer indices line up with the chunk list.
    """

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
    def add(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
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
                "ids": new_ids[i : i + batch_size],
                "embeddings": embeddings_list[i : i + batch_size],
            }
            if metadatas is not None:
                kwargs["metadatas"] = metadatas[i : i + batch_size]
            # Store the chunk text as the Chroma *document* too — that's what
            # ``where_document`` ($contains / $regex) matches against.
            if documents is not None:
                kwargs["documents"] = documents[i : i + batch_size]
            self._col.add(**kwargs)

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self._ids))
        if k == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        query_kwargs: dict = {"query_embeddings": [query_vec.tolist()], "n_results": k}
        if where is not None:
            query_kwargs["where"] = where
        if where_document is not None:
            query_kwargs["where_document"] = where_document
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


# ===========================================================================
# BM25 — the sparse leg fused with dense vectors per collection
# ===========================================================================

# BM25 sparse-retrieval token splitter.  Splits on every non-alphanumeric
# character so ``snake_case`` and ``kebab-case`` identifiers fan out into
# their parts; this matters because much of what an agent searches the KB
# for ("get_user_id", "kb-ingest") is identifier-shaped, and BM25 needs
# token-level matches to score them at all.  CamelCase still survives as
# a single token, which is fine — the dense embedding handles those.
_BM25_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _bm25_tokenize(text: str) -> list[str]:
    return [t.lower() for t in _BM25_TOKEN_RE.findall(text)]


class BM25Index:
    """Sparse BM25 keyword index over a collection's chunk texts.

    Maintained alongside the dense vector index so single-collection search
    can fuse the two via Reciprocal Rank Fusion.  Stored as a single pickle
    file (``bm25.pkl``) per collection — its presence is also what marks a
    collection as *hybrid*.

    Rebuilt from scratch on every write because BM25 IDF stats are
    corpus-global — there is no cheap incremental update.  For DSAGT corpus
    sizes (single-digit thousands of chunks) the rebuild is millisecond-scale.
    Watch the cost if a collection grows past tens of thousands of entries.
    """

    _FILENAME = "bm25.pkl"

    def __init__(self):
        self._bm25 = None
        self._n = 0

    def build(self, texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        if not texts:
            self._bm25 = None
            self._n = 0
            return
        tokenized = [_bm25_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._n = len(texts)

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, positional_indices) for the top *k* hits."""
        if self._bm25 is None or self._n == 0 or k <= 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )
        tokens = _bm25_tokenize(query)
        if not tokens:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )
        scores = self._bm25.get_scores(tokens)
        k = min(k, self._n)
        # argpartition is O(n); we only need the top-k unsorted, then sort
        # those k.  Faster than np.argsort on the full array for big k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return scores[top_idx].astype(np.float32), top_idx.astype(np.int64)

    def save(self, directory: Path) -> None:
        import pickle

        with open(directory / self._FILENAME, "wb") as f:
            pickle.dump({"bm25": self._bm25, "n": self._n}, f)

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        import pickle

        obj = cls()
        path = directory / cls._FILENAME
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            obj._bm25 = data["bm25"]
            obj._n = data["n"]
        return obj

    @property
    def size(self) -> int:
        return self._n


# ===========================================================================
# Reciprocal Rank Fusion — within a collection AND across collections/stores
# ===========================================================================


def _rrf_merge(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple ranked-index lists.

    Each ranking is a list of positional indices in descending relevance.
    Combined RRF score for doc ``i`` is ``sum_r 1 / (k + rank_r(i) + 1)``
    summed across rankers that included ``i``.  *k=60* is the standard
    constant from Cormack et al. — large enough to dampen the long tail
    of the per-ranker rank curve, small enough that the top few ranks
    still dominate.

    Returns a list of ``(positional_index, rrf_score)`` tuples sorted
    by descending score.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            if idx < 0:
                continue
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _chunk_key(chunk: dict) -> tuple:
    """Stable identity for a result chunk, for cross-collection RRF.

    Different collections (and, later, different stores/embedders) live in
    incomparable score spaces, so fusion is rank-only.  We key on the chunk's
    own metadata rather than score; ``id``-based fallback keeps distinct chunks
    distinct even when metadata is sparse.
    """
    meta = chunk.get("metadata", {})
    return (
        meta.get("collection"),
        meta.get("source_file"),
        meta.get("chunk_index"),
        chunk.get("text", "")[:64],
    )


def _rrf_across(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Rank-fuse per-collection result lists (each ``[{chunk, score}, ...]``).

    Rank-only RRF across the lists — no cross-space score normalization, which
    is exactly what lets results from different embedding spaces fuse correctly.
    Returns a single list of result dicts (``score`` replaced by the RRF score),
    sorted by descending fused score.
    """
    scores: dict[tuple, float] = {}
    rep: dict[tuple, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            key = _chunk_key(r["chunk"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            rep.setdefault(key, r)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{**rep[key], "score": float(score)} for key, score in ordered]


# ---------------------------------------------------------------------------
# Recency weighting (episodic session_memory only)
# ---------------------------------------------------------------------------

#: The one collection recency applies to.  Mirrors
#: ``memory.SESSION_MEMORY_COLLECTION`` as a literal to avoid a knowledge→memory
#: import (memory imports knowledge, not the reverse).
_RECENCY_COLLECTION = "session_memory"

#: Max fractional boost a brand-new fact gets over its raw relevance.  A *boost*,
#: never a penalty: recency only lifts recent facts, so a strongly-relevant old
#: fact (e.g. a day-1 threshold that never changed) is never buried — it keeps
#: its full relevance score while a same-relevance newer fact edges ahead.
_RECENCY_BOOST = 0.5


def _apply_recency(
    results: list[dict],
    half_life_days: float,
    now: float,
    boost: float = _RECENCY_BOOST,
) -> list[dict]:
    """Re-rank ``[{chunk, score}, ...]`` by relevance × a recency factor.

    ``factor = 1 + boost · 2^(-age / half_life)`` — newest facts get up to
    ``1+boost``, decaying toward ``1`` (no change) with the given half-life.
    Facts without a numeric ``ts_epoch`` get factor ``1.0`` (unweighted), never
    dropped.  Pure + side-effect-free so it's unit-testable without an embedder.
    """
    hl_seconds = max(1.0, half_life_days * 86400.0)
    weighted = []
    for r in results:
        ts = r["chunk"].get("metadata", {}).get("ts_epoch")
        if ts is None:
            factor = 1.0
        else:
            age = max(0.0, now - float(ts))
            factor = 1.0 + boost * (0.5 ** (age / hl_seconds))
        weighted.append({**r, "score": r["score"] * factor, "recency_factor": factor})
    weighted.sort(key=lambda r: r["score"], reverse=True)
    return weighted


# ===========================================================================
# VectorStore — one embedder, many collections (the BYO adapter port)
# ===========================================================================


class VectorStore(ABC):
    """One :class:`Embedder`, many collections: store + single-collection search.

    The adapter contract a BYO store implements: subclass this and wrap your
    own backend.  :class:`KnowledgeBase` holds a list of these and fuses their
    collections.  Heterogeneity (mixed embedding spaces) is expressed by having
    *several* stores in the list, never by routing within one store.

    TODO (deferred to the second concrete store — BYO/FAISS, see
    knowledge-base-plan.md): most of :class:`ChromaVectorStore` is backend-
    agnostic — the BM25 sparse leg, the dense+sparse hybrid fusion, the
    ``chunks.jsonl`` payload, ``_normalize``.  Only the dense index
    (add/query/persist over Chroma's HNSW) is Chroma-specific.  When a second
    store lands, lift that reusable machinery into a shared base that defers
    only the dense primitives to subclasses, so a FAISS/BYO store reuses it
    rather than copying it.  Not extracted now: with one impl the seam can't be
    validated.  Dense-only stores (external adapters with no local sparse leg)
    are modelled as a *separate store type*, not a per-instance flag.
    """

    embedder: Embedder

    @property
    @abstractmethod
    def collections(self) -> list[str]:
        """Names of the collections this store hosts."""

    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """Whether *name* is hosted by this store."""

    @abstractmethod
    def collection_info(self, name: str) -> dict:
        """Listing metadata for *name* (description, embedder, vector_db)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* with this store's embedder (L2-normalized)."""

    @abstractmethod
    def add_chunks(
        self,
        collection: str,
        chunks: list[dict],
        return_embeddings: bool = False,
    ) -> dict:
        """Embed + store pre-built ``{text, metadata}`` chunks into *collection*."""

    @abstractmethod
    def add_entries(
        self,
        texts: list[str],
        collection: str,
        metadatas: list[dict] | None = None,
        return_embeddings: bool = False,
    ) -> dict:
        """Embed + store raw *texts* (with optional metadata) into *collection*."""

    @abstractmethod
    def search(
        self,
        query: str,
        collection: str,
        top_k: int,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """Single-collection hybrid search → ``[{chunk, score}, ...]``."""

    def close(self) -> None:
        pass


class ChromaVectorStore(VectorStore):
    """Local-ChromaDB store: one embedder, many collections under *index_dir*.

    Each collection lives in ``<index_dir>/<name>/`` as a Chroma persistent
    collection (dense), a ``chunks.jsonl`` payload, and a ``bm25.pkl`` sparse
    leg.  The embedder is fixed for the whole store; it is built lazily (via
    :meth:`Embedder.create`) so server startup can background-load it, or injected
    directly for reuse across stores.

    The local store is **unconditionally hybrid**: it writes a BM25 sparse leg on
    every write and fuses dense + sparse on every (unfiltered) search.  Dense-only
    retrieval is not a toggle here — a metadata-``where`` filter falls back to
    dense-only for that one query (BM25 has no filter equivalent), and dense-only
    *stores* (external/BYO adapters with no local sparse leg) are a separate store
    type, not a flag on this one.
    """

    def __init__(
        self,
        index_dir: str | Path,
        embedder: Embedder | None = None,
        *,
        backend: str = "api",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Embedder is either injected (shareable across stores) or built lazily
        # from explicit args.  Lazy build keeps the ~5-10s sentence-transformers
        # load off the hot path (see preload via KnowledgeBase).  backend/model
        # are kept as lightweight labels so listing never forces a model load.
        self._embedder = embedder
        if embedder is not None:
            self._backend, self._model = embedder.backend, embedder.model
        else:
            self._backend, self._model = backend, model
        self._base_url, self._api_key, self._device = base_url, api_key, device
        # Serializes embedder construction so a background preload and a
        # foreground first-query call don't race and double-load the model.
        self._embedder_lock = threading.Lock()

        # Collection runtime caches: name → (ChromaIndex, chunks) and name → BM25.
        self._cache: dict[str, tuple[ChromaIndex, list[dict]]] = {}
        self._bm25_cache: dict[str, BM25Index] = {}

    # -- embedder ------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:  # type: ignore[override]
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = Embedder.create(
                    self._backend,
                    model=self._model,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    device=self._device,
                )
            return self._embedder

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model(self) -> str | None:
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        with kb_embed_span(self.backend, self.model, len(texts)):
            return self._normalize(self.embedder.embed(texts))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.where(norms > 0, norms, 1)

    # -- collection inventory ------------------------------------------------

    @property
    def collections(self) -> list[str]:
        return [
            p.name
            for p in self.index_dir.iterdir()
            if p.is_dir() and (p / "chroma_ids.json").exists()
        ]

    def has_collection(self, name: str) -> bool:
        return (self.index_dir / name / "chroma_ids.json").exists()

    def collection_info(self, name: str) -> dict:
        coll_dir = self.index_dir / name
        desc_path = coll_dir / "DESCRIPTION.md"
        description = desc_path.read_text() if desc_path.exists() else ""
        return {
            "name": name,
            "description": description,
            "embedding_backend": self.backend,
            "embedding_model": self.model or "default",
            "vector_db": "chroma",
        }

    # -- writes --------------------------------------------------------------

    def add_chunks(
        self,
        collection: str,
        chunks: list[dict],
        return_embeddings: bool = False,
    ) -> dict:
        """Embed + store pre-built ``{text, metadata}`` chunks.

        The shared write path for both document ingest (chunks carry
        ``source_file`` / ``file_type``) and structured entries.  Rebuilds the
        BM25 sparse leg on every write (the store is unconditionally hybrid).
        """
        coll_dir = self.index_dir / collection
        coll_dir.mkdir(parents=True, exist_ok=True)

        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        embeddings = self.embed(texts)

        if (coll_dir / "chunks.jsonl").exists():
            index, existing_chunks = self._load(collection)
        else:
            index = ChromaIndex(collection_name=coll_dir.name, persist_dir=coll_dir)
            existing_chunks = []

        try:
            index.add(embeddings, metadatas=metadatas, documents=texts)
        except Exception as e:
            hint = self._stale_index_message(collection, e)
            if hint:
                raise RuntimeError(hint) from e
            raise
        index.save(coll_dir)

        with open(coll_dir / "chunks.jsonl", "a") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        all_chunks = existing_chunks + chunks
        self._rebuild_bm25(collection, all_chunks)
        self._cache[collection] = (index, all_chunks)

        result = {
            "collection": collection,
            "entries_added": len(chunks),
            "total_entries": len(all_chunks),
        }
        if return_embeddings:
            result["embeddings"] = embeddings
        return result

    def add_entries(
        self,
        texts: list[str],
        collection: str,
        metadatas: list[dict] | None = None,
        return_embeddings: bool = False,
    ) -> dict:
        """Add raw *texts* (no parsing/chunking) as entries in *collection*."""
        coll_dir = self.index_dir / collection
        existing = 0
        if (coll_dir / "chunks.jsonl").exists():
            _, existing_chunks = self._load(collection)
            existing = len(existing_chunks)

        chunks = []
        for i, text in enumerate(texts):
            entry_meta = metadatas[i] if metadatas else {}
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "collection": collection,
                        "source_file": "entry",
                        "chunk_index": existing + i,
                        **entry_meta,
                    },
                }
            )
        return self.add_chunks(collection, chunks, return_embeddings=return_embeddings)

    # -- single-collection hybrid search ------------------------------------

    def search(
        self,
        query: str,
        collection: str,
        top_k: int,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        index, chunks = self._load(collection)
        query_emb = self.embed([query])[0]

        # A ``where`` (metadata) or ``where_document`` (text content) filter
        # disables the BM25 leg — BM25 has no filter equivalent, so a filtered
        # search is dense-only.
        filtered = where is not None or where_document is not None
        do_hybrid = not filtered
        # Oversample the per-ranker pools so RRF has depth on small top_k.
        candidate_k = min(
            max(top_k * 10, 50) if do_hybrid else top_k,
            len(chunks),
        )

        with kb_index_search_span("chroma", candidate_k, filtered):
            try:
                if filtered:
                    dense_scores, dense_indices = index.search(
                        query_emb,
                        candidate_k,
                        where=where,
                        where_document=where_document,
                    )
                else:
                    dense_scores, dense_indices = index.search(query_emb, candidate_k)
            except Exception as e:
                hint = self._stale_index_message(collection, e)
                if hint:
                    raise RuntimeError(hint) from e
                raise

        if do_hybrid:
            bm25 = self._get_bm25(collection)
            _, sparse_indices = bm25.search(query, candidate_k)
            dense_ranking = [int(i) for i in dense_indices if i >= 0]
            sparse_ranking = [int(i) for i in sparse_indices]
            merged = _rrf_merge([dense_ranking, sparse_ranking])
            results = [
                {"chunk": chunks[idx], "score": float(score)} for idx, score in merged
            ]
        else:
            results = [
                {"chunk": chunks[i], "score": float(dense_scores[j])}
                for j, i in enumerate(dense_indices)
                if i >= 0
            ]
        return results[:top_k]

    # -- persistence helpers -------------------------------------------------

    def _rebuild_bm25(self, collection: str, chunks: list[dict]) -> None:
        """Build (or rebuild) the BM25 index for *collection* from *chunks*."""
        bm25 = BM25Index()
        bm25.build([c["text"] for c in chunks])
        bm25.save(self.index_dir / collection)
        self._bm25_cache[collection] = bm25

    def _get_bm25(self, collection: str) -> BM25Index:
        cached = self._bm25_cache.get(collection)
        if cached is not None:
            return cached
        bm25 = BM25Index.load(self.index_dir / collection)
        self._bm25_cache[collection] = bm25
        return bm25

    def _load(self, name: str) -> tuple[ChromaIndex, list[dict]]:
        """Load a collection's index + chunks (cached in memory)."""
        if name in self._cache:
            return self._cache[name]

        coll_dir = self.index_dir / name
        if not coll_dir.exists():
            raise ValueError(f"Collection '{name}' not found")

        index = ChromaIndex.load(coll_dir)
        with open(coll_dir / "chunks.jsonl") as f:
            chunks = [json.loads(line) for line in f]

        self._cache[name] = (index, chunks)
        return index, chunks

    @staticmethod
    def _stale_index_message(collection: str, exc: Exception) -> str | None:
        """Return a user-friendly hint when *exc* looks like a dim mismatch.

        Chroma raises ``InvalidDimensionException`` with "dimension" in the
        message when a query vector's dimension doesn't match the index's —
        i.e. the embedder that BUILT the index was swapped for one with a
        different output dim (the user changed ``embedding.backend`` /
        ``embedding.model`` after init).  Detect on message-shape.
        """
        msg = str(exc).lower()
        if "dimension" in msg or "dim mismatch" in msg or "shape" in msg:
            return (
                f"Embedding dimension mismatch in collection {collection!r}. "
                f"The index was built with a different embedder than the one "
                f"currently configured.  Drop the stale collection "
                f"(`rm -rf <project>/kb_index/{collection}`) and re-create the "
                f"project with `dsagt init` so it rebuilds with the active "
                f"embedding settings."
            )
        return None

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
        self._embedder = None
        self._cache.clear()
        self._bm25_cache.clear()


# ===========================================================================
# KnowledgeBase — the federation infra: a list of stores + ingest pipeline
# ===========================================================================


class KnowledgeBase:
    """Collection-based document retrieval over one-or-more vector stores.

    Holds a **list** of :class:`VectorStore`s (today just the internal local
    Chroma store) and its job is to **fuse their collections**: collection→store
    routing plus rank-fusion across collections.  It also owns the document
    ingestion pipeline (collect / parse / chunk → ``VectorStore.add_chunks``) and
    cross-collection reranking.  This is the shared substrate every KB consumer
    (retrieval, memory, provenance, skills) calls into.

    Quick-start
    -----------
    .. code-block:: python

        kb = KnowledgeBase(index_dir="./kb_store", default_embedder="local")
        kb.ingest("./docs/research", "research")
        results = kb.search("transformer architecture", "research")
    """

    FILE_TYPES = [
        "pdf",
        "md",
        "rst",
        "txt",
        "py",
        "docx",
        "json",
        "yaml",
        "yml",
        # Packaging metadata: agents reading this index need to know which
        # version of a library to install when registering tools that
        # depend on it.  pyproject.toml is the modern standard; setup.cfg
        # is still common in older codebases.
        "toml",
        "cfg",
    ]

    def __init__(
        self,
        index_dir: str | Path,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        default_rerank: bool = False,
        recency_half_life_days: float | None = None,
        # Internal store's embedder (one per store, fixed at construction).
        # Explicit args — callers unpack their config here, no kwargs dict.
        default_embedder: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model
        self.default_rerank = default_rerank
        # Episodic recency weighting (session_memory only); None = off.
        self._recency_half_life_days = recency_half_life_days

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # The internal store: local Chroma, one embedder, many collections.
        # External BYO stores would be appended to ``self._stores`` (deferred).
        self._store = ChromaVectorStore(
            self.index_dir,
            backend=default_embedder or "api",
            model=model,
            base_url=base_url,
            api_key=api_key,
            device=device,
        )
        self._stores: list[VectorStore] = [self._store]

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

    # -- store routing -------------------------------------------------------

    def _store_for(self, collection: str) -> VectorStore:
        """Return the store hosting *collection*, else raise ``ValueError``."""
        for store in self._stores:
            if store.has_collection(collection):
                return store
        raise ValueError(f"Collection '{collection}' not found")

    @property
    def collections(self) -> list[str]:
        """Union of collection names across all stores."""
        seen: list[str] = []
        for store in self._stores:
            for name in store.collections:
                if name not in seen:
                    seen.append(name)
        return seen

    def list_collections(self) -> list[dict]:
        return [
            store.collection_info(name)
            for store in self._stores
            for name in store.collections
        ]

    def preload_default_embedder(self) -> None:
        """Kick off internal-store embedder construction in a daemon thread.

        Called at MCP server startup so the heavy load (sentence-transformers
        import + model load, ~5-10s) happens in parallel with the rest of
        bootstrap.  Failure is swallowed: it resurfaces with a full traceback
        on the first real embedding call.
        """

        def _load() -> None:
            try:
                self._store.embedder  # noqa: B018 — triggers lazy construction
            except Exception as e:
                logger.warning("Background embedder preload failed: %s", e)

        threading.Thread(
            target=_load, name="dsagt-embedder-preload", daemon=True
        ).start()

    def embed_texts(
        self, texts: list[str], collection: str | None = None
    ) -> np.ndarray:
        """Embed *texts* with the internal store's embedder (L2-normalized)."""
        return self._store.embed(texts)

    # -- structured entries (delegate to internal store) --------------------

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
        return_embeddings: bool = False,
    ) -> dict:
        """Add pre-formed text entries with optional metadata to a collection.

        Unlike ``ingest``/``append``, this skips document parsing and chunking.
        Used by episodic memory, tool_executions, and other structured entry
        types that produce their own text representations.  Metadata is stored
        as native Chroma metadata for ``where`` filtering.
        """
        obs.set_inputs(
            {
                "collection": collection,
                "n_entries": len(texts),
                "texts_preview": [t[:200] for t in texts[:3]],
            }
        )
        return self._store.add_entries(
            texts,
            collection,
            metadatas=metadatas,
            return_embeddings=return_embeddings,
        )

    # -- document ingestion pipeline ----------------------------------------

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
        exclude_patterns: list[str] | None = None,
    ) -> dict:
        """Ingest *folder* as a new collection in the internal store."""
        folder = Path(folder)
        collection = collection_name or folder.name
        file_types = file_types or self.FILE_TYPES

        obs.set_inputs(
            {
                "folder": str(folder),
                "collection": collection,
                "file_types": file_types,
            }
        )

        coll_dir = self.index_dir / collection
        coll_dir.mkdir(parents=True, exist_ok=True)

        # Record source folder so the MCP server can detect re-ingests vs. conflicts.
        (coll_dir / "source.txt").write_text(str(folder.resolve()))

        desc_src = folder / "DESCRIPTION.md"
        if desc_src.exists():
            (coll_dir / "DESCRIPTION.md").write_text(desc_src.read_text())

        files = self._collect_files(folder, file_types, exclude_patterns)
        logger.info("Found %d files to process", len(files))

        self._chunk_skip_count = 0
        chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        n_skipped = self._chunk_skip_count
        logger.info("Created %d chunks (skipped %d files)", len(chunks), n_skipped)

        if not chunks:
            return {
                "collection": collection,
                "files": 0,
                "chunks": 0,
                "skipped_files": n_skipped,
            }

        self._store.add_chunks(collection, chunks)
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

        obs.set_inputs(
            {
                "collection": collection,
                "n_paths": len(paths),
                "paths_preview": [str(p) for p in paths[:5]],
            }
        )

        store = self._store_for(collection)  # raises ValueError if missing

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
            return {
                "collection": collection,
                "files": 0,
                "chunks_added": 0,
                "skipped_files": 0,
            }

        self._chunk_skip_count = 0
        new_chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        n_skipped = self._chunk_skip_count
        if not new_chunks:
            return {
                "collection": collection,
                "files": len(files),
                "chunks_added": 0,
                "skipped_files": n_skipped,
            }

        result = store.add_chunks(collection, new_chunks)
        return {
            "collection": collection,
            "files": len(files),
            "chunks_added": len(new_chunks),
            "total_chunks": result.get("total_entries"),
            "skipped_files": n_skipped,
        }

    # -- federated search ----------------------------------------------------

    @traced("kb.search", capture=["collection", "top_k", "rerank"])
    def search(
        self,
        query: str,
        collection: str | None = None,
        collections: list[str] | None = None,
        top_k: int = 5,
        rerank: bool | None = None,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """Search one or many collections, fusing across them by rank.

        A single collection routes straight to its store's hybrid search.
        Multiple collections fan out — each store searched, then the per-
        collection rankings fused by Reciprocal Rank Fusion (rank-only, so
        different embedding spaces compose correctly).  Optional cross-encoder
        rerank runs over the fused candidates.

        Missing collections are skipped with a warning; the search fails only
        when *every* requested collection is absent.
        """
        if rerank is None:
            rerank = self.default_rerank

        targets = collections or ([collection] if collection else [])
        if not targets:
            raise ValueError("Provide 'collection' or 'collections'")

        obs.set_inputs({"query": query, "collections": targets, "top_k": top_k})

        # Oversample per-collection pools when reranking, fusing >1 collection,
        # or recency-weighting (so a recent fact can be lifted from deep in the
        # pool, not merely reordered within an already-cut top_k).
        recency_target = bool(
            self._recency_half_life_days and targets == [_RECENCY_COLLECTION]
        )
        oversample = rerank or len(targets) > 1 or recency_target
        candidate_k = max(top_k * 10, 50) if oversample else top_k

        per_coll: list[list[dict]] = []
        errors: list[str] = []
        for coll in targets:
            try:
                store = self._store_for(coll)
                per_coll.append(
                    store.search(
                        query,
                        coll,
                        candidate_k,
                        where=where,
                        where_document=where_document,
                    )
                )
            except (ValueError, FileNotFoundError, KeyError) as e:
                logger.warning("Search failed for '%s': %s", coll, e)
                errors.append(str(e))

        if not per_coll:
            if len(targets) == 1:
                raise ValueError(
                    errors[0] if errors else f"Collection '{targets[0]}' not found"
                )
            raise ValueError(f"All collections failed: {'; '.join(errors)}")

        fused = per_coll[0] if len(per_coll) == 1 else _rrf_across(per_coll)

        # Episodic recency: a recent corrected fact outranks a stale one without
        # any contradiction detection — recency is the ranker for session_memory
        # (mutually exclusive with the cross-encoder; the two are alternative
        # rerankers and recency is the one that matters for a time-ordered log).
        if recency_target and fused:
            final = _apply_recency(fused, self._recency_half_life_days, time.time())[
                :top_k
            ]
        elif rerank and fused:
            with kb_rerank_span(self.rerank_model, len(fused)):
                final = self._rerank(query, fused, top_k)
        else:
            final = fused[:top_k]

        obs.set("hits", len(final))
        obs.set_outputs(
            {
                "hits": len(final),
                "top_texts": [r["chunk"].get("text", "")[:200] for r in final[:3]],
            }
        )
        return final

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.rerank_model, max_length=512)
        pairs = [[query, r["chunk"]["text"]] for r in results]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [{**r, "rerank_score": float(s)} for r, s in ranked[:top_k]]

    # -- file discovery + chunking ------------------------------------------

    def _collect_files(
        self,
        folder: Path,
        file_types: list[str],
        exclude_patterns: list[str] | None,
    ) -> list[Path]:
        """Walk *folder* for files matching *file_types*, applying optional
        glob exclusions.

        Pure function relative to filesystem state — no caching, no
        side-effecting attributes.  Patterns are checked against the relative
        path, the basename, and each individual path segment, so ``"tests"``
        excludes any file whose path contains a ``tests/`` directory.
        """
        from fnmatch import fnmatch

        all_files = [f for ext in file_types for f in folder.glob(f"**/*.{ext}")]

        if not exclude_patterns:
            return all_files

        def _excluded(f: Path) -> bool:
            rel = str(f.relative_to(folder))
            name = f.name
            return any(
                fnmatch(rel, pat)
                or fnmatch(name, pat)
                or any(fnmatch(part, pat) for part in Path(rel).parts)
                for pat in exclude_patterns
            )

        kept = [f for f in all_files if not _excluded(f)]
        n_excluded = len(all_files) - len(kept)
        if n_excluded:
            logger.info(
                "Excluded %d/%d files via %d pattern(s)",
                n_excluded,
                len(all_files),
                len(exclude_patterns),
            )
        return kept

    def _chunk_file(self, path: Path, collection: str) -> Iterator[dict]:
        # Lazy import — see the module-top comment about cold-start cost.
        import contextlib
        import io as _io
        from llama_index.core import SimpleDirectoryReader

        # Per-file read/parse failures are kept as soft failures (count
        # surfaced in ingest()'s return dict) rather than aborting the
        # whole ingest
        try:
            with contextlib.redirect_stdout(_io.StringIO()):
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
        except Exception as e:
            # A single malformed file (or a tree-sitter ABI mismatch in the
            # code parser) must not abort the whole ingest — skip it.
            logger.warning("Could not parse %s: %s", path, e)
            self._chunk_skip_count += 1
            return
        for i, node in enumerate(nodes):
            text = node.get_content().strip()
            if not text:
                continue
            yield {
                "id": hashlib.sha256(f"{path}:{i}:{text[:100]}".encode()).hexdigest()[
                    :16
                ],
                "text": text,
                "metadata": {
                    "source_file": str(path),
                    "collection": collection,
                    "chunk_index": i,
                    "file_type": file_type,
                },
            }

    def _get_parser(self, file_type: str):
        """Return a cached parser for *file_type*, building it on first use."""
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
        for store in self._stores:
            store.close()
        self._parsers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
