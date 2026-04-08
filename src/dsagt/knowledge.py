"""Knowledge base for DSAGT agent — with pluggable embedder & vector-DB routing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

import numpy as np

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter


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


class APIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI-compatible REST API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        batch_size: int = 100,
    ):
        import httpx

        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small-project")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError("API key required via argument or LLM_API_KEY env var")

        self._client = httpx.Client(timeout=timeout)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        response = self._client.post(
            f"{self.base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda x: x["index"])
        return np.array([d["embedding"] for d in data], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        if len(texts) <= self.batch_size:
            return self._embed_batch(texts)
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info("Embedding batch %d/%d (%d texts)",
                        i // self.batch_size + 1,
                        (len(texts) + self.batch_size - 1) // self.batch_size,
                        len(batch))
            batches.append(self._embed_batch(batch))
        return np.vstack(batches)

    def close(self) -> None:
        self._client.close()

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
        kwargs: dict = {"ids": new_ids, "embeddings": embeddings.tolist()}
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
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

    def __init__(
        self,
        embedding_backend: str = "api",
        vector_db: str = "faiss",
        embedder_kwargs: dict | None = None,
        index_kwargs: dict | None = None,
        description: str = "",
    ):
        self.embedding_backend = embedding_backend
        self.vector_db = vector_db
        self.embedder_kwargs: dict = embedder_kwargs or {}
        self.index_kwargs: dict = index_kwargs or {}
        self.description = description

    def to_dict(self) -> dict:
        return {
            "embedding_backend": self.embedding_backend,
            "vector_db": self.vector_db,
            "embedder_kwargs": self.embedder_kwargs,
            "index_kwargs": self.index_kwargs,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CollectionRoute":
        return cls(**d)


# default route used when no collection-specific route is registered.
# If EMBEDDING_MODEL is set, propagate it to the embedder. Otherwise, let
# APIEmbeddingClient use its own default — avoids hardcoding a model name
# that may not exist at every institution's endpoint.
_env_embedding_model = os.getenv("EMBEDDING_MODEL")
DEFAULT_ROUTE = CollectionRoute(
    embedding_backend=os.getenv("DSAGT_EMBEDDING_BACKEND", "api"),
    vector_db="faiss",
    embedder_kwargs={"model": _env_embedding_model} if _env_embedding_model else {},
)


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
    ``"faiss"``
        Flat inner-product index (default).  No extra service needed.
    ``"chroma"``
        ChromaDB HNSW index.  ``pip install chromadb``.
    """

    FILE_TYPES = ["pdf", "md", "txt", "py", "docx", "json", "yaml", "yml"]
    _ROUTE_FILE = "route.json"

    def __init__(
        self,
        index_dir: str | Path,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        # Global default route (used when collection has no specific route)
        default_embedder: str | None = None,
        default_index: str | None = None,
        # Per-collection routing registry
        routes: dict[str, CollectionRoute] | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Build default route
        _model = os.getenv("EMBEDDING_MODEL")
        self._default_route = CollectionRoute(
            embedding_backend=default_embedder or os.getenv("DSAGT_EMBEDDING_BACKEND", "api"),
            vector_db=default_index or "faiss",
            embedder_kwargs={"model": _model} if _model else {},
        )

        # Per-collection route registry
        self._routes: dict[str, CollectionRoute] = routes or {}

        # Shared embedder cache: embedder_key → client instance
        # Key is "<embedding_backend>|<sorted embedder_kwargs>" so identical configs share one client.
        self._embedder_cache: dict[str, BaseEmbeddingClient] = {}

        # Collection runtime cache: name → (BaseVectorIndex, list[dict])
        self._cache: dict[str, tuple[BaseVectorIndex, list[dict]]] = {}

        self._reranker = None

    # route management

    def register_route(self, collection: str, route: CollectionRoute) -> None:
        """Register (or update) a routing rule for *collection*."""
        self._routes[collection] = route
        logger.info("Registered route for '%s': embedder=%s index=%s",
                    collection, route.embedding_backend, route.vector_db)

    def _get_route(self, collection: str) -> CollectionRoute:
        return self._routes.get(collection, self._default_route)

    def _get_embedder(self, route: CollectionRoute) -> BaseEmbeddingClient:
        """Return (possibly cached) embedder matching *route*."""
        key = f"{route.embedding_backend}|{sorted(route.embedder_kwargs.items())}"
        if key not in self._embedder_cache:
            self._embedder_cache[key] = _make_embedder(
                route.embedding_backend, **route.embedder_kwargs
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

    def ingest(
        self,
        folder: str | Path,
        collection_name: str | None = None,
        file_types: list[str] | None = None,
        route: CollectionRoute | None = None,
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
        """
        folder = Path(folder)
        collection = collection_name or folder.name
        file_types = file_types or self.FILE_TYPES

        # Register route (if provided) into memory
        if route is not None:
            self.register_route(collection, route)
        active_route = self._get_route(collection)

        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)

        # Record source folder so the MCP server can detect re-ingests vs. conflicts.
        (coll_dir / "source.txt").write_text(str(folder.resolve()))

        desc_src = folder / "DESCRIPTION.md"
        if desc_src.exists():
            (coll_dir / "DESCRIPTION.md").write_text(desc_src.read_text())

        files = [f for ext in file_types for f in folder.glob(f"**/*.{ext}")]
        logger.info("Found %d files to process", len(files))

        chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            return {"collection": collection, "files": 0, "chunks": 0}

        embedder = self._get_embedder(active_route)
        embeddings = self._normalize(embedder.embed([c["text"] for c in chunks]))

        index = _make_index(active_route.vector_db,
                            **self._index_init_kwargs(active_route, coll_dir))
        index.add(embeddings)
        index.save(coll_dir)

        # Persist route AFTER index is built — guarantees route.json always
        # reflects what was actually used, never overwritten by a racing job.
        self._save_route(collection, active_route)

        with open(coll_dir / "chunks.jsonl", "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        self._cache[collection] = (index, chunks)
        return {"collection": collection, "files": len(files), "chunks": len(chunks)}

    def append(
        self,
        collection: str,
        paths: list[str | Path],
        file_types: list[str] | None = None,
    ) -> dict:
        """Append documents to an existing collection."""
        file_types = file_types or self.FILE_TYPES

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
                    "total_chunks": len(existing_chunks)}

        new_chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        if not new_chunks:
            return {"collection": collection, "files": len(files),
                    "chunks_added": 0, "total_chunks": len(existing_chunks)}

        embedder = self._get_embedder(active_route)
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
        }

    def add_entries(
        self,
        texts: list[str],
        collection: str,
        metadatas: list[dict] | None = None,
        route: CollectionRoute | None = None,
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

        Returns
        -------
        dict
            ``{collection, entries_added, total_entries}``
        """
        if not texts:
            return {"collection": collection, "entries_added": 0, "total_entries": 0}

        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)

        if route is not None:
            self.register_route(collection, route)
        active_route = self._get_route(collection)

        embedder = self._get_embedder(active_route)
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
        self._save_route(collection, active_route)

        # Append to chunks.jsonl
        with open(coll_dir / "chunks.jsonl", "a") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk) + "\n")

        all_chunks = existing_chunks + new_chunks
        self._cache[collection] = (index, all_chunks)

        return {
            "collection": collection,
            "entries_added": len(texts),
            "total_entries": len(all_chunks),
        }

    def embed_texts(self, texts: list[str], collection: str) -> np.ndarray:
        """Embed texts using the embedder configured for a collection.

        Returns L2-normalized float32 array of shape (n_texts, dim).
        """
        route = self._get_route(collection)
        embedder = self._get_embedder(route)
        return self._normalize(embedder.embed(texts))

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        rerank: bool = False,
        where: dict | None = None,
    ) -> list[dict]:
        """Search *collection* with *query*.

        Parameters
        ----------
        where : dict, optional
            ChromaDB ``where`` filter clause.  Only effective on ChromaDB-backed
            collections; silently ignored for FAISS collections.
        """
        index, chunks = self._load(collection)
        active_route = self._get_route(collection)
        embedder = self._get_embedder(active_route)

        query_emb = self._normalize(embedder.embed([query]))[0]
        search_k = min(top_k * 10 if rerank else top_k, len(chunks))

        # Pass where clause only to ChromaDB indexes
        if where is not None and active_route.vector_db == "chroma":
            scores, indices = index.search(query_emb, search_k, where=where)
        else:
            scores, indices = index.search(query_emb, search_k)

        results = [
            {"chunk": chunks[i], "score": float(scores[j])}
            for j, i in enumerate(indices) if i >= 0
        ]

        if rerank and results:
            return self._rerank(query, results, top_k)
        return results[:top_k]

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

    def _save_route(self, collection: str, route: CollectionRoute) -> None:
        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)
        (coll_dir / self._ROUTE_FILE).write_text(json.dumps(route.to_dict(), indent=2))

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

    def _chunk_file(self, path: Path, collection: str) -> Iterator[dict]:
        try:
            docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return
        if not docs:
            return
        file_type = path.suffix.lower()
        parser = self._get_parser(file_type)
        try:
            nodes = parser.get_nodes_from_documents(docs)
        except Exception as e:
            logger.warning("Could not parse %s: %s", path, e)
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
        if file_type in CODE_LANGUAGES:
            return CodeSplitter(
                language=CODE_LANGUAGES[file_type],
                chunk_lines=40,
                chunk_lines_overlap=10,
                max_chars=self.chunk_size * 4,
            )
        if file_type == ".md":
            return MarkdownNodeParser()
        return SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def close(self) -> None:
        for client in self._embedder_cache.values():
            client.close()
        self._embedder_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
