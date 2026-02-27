"""Knowledge base for DSAGT agent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
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


class LocalEmbeddingClient:
    """Generate text embeddings using a local sentence-transformers model."""

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
        """Embed texts, returns array of shape (n_texts, embedding_dim)."""
        if not texts:
            return np.array([], dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    def close(self):
        pass


class APIEmbeddingClient:
    """Generate text embeddings via OpenAI-compatible API."""

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
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://ai-incubator-api.pnnl.gov")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError("API key required via argument or LLM_API_KEY env var")

        self._client = httpx.Client(timeout=timeout)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a single batch of texts."""
        response = self._client.post(
            f"{self.base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()

        data = sorted(response.json()["data"], key=lambda x: x["index"])
        return np.array([d["embedding"] for d in data], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts with batching, returns array of shape (n_texts, embedding_dim)."""
        if not texts:
            return np.array([], dtype=np.float32)

        if len(texts) <= self.batch_size:
            return self._embed_batch(texts)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info("Embedding batch %d/%d (%d texts)",
                        i // self.batch_size + 1,
                        (len(texts) + self.batch_size - 1) // self.batch_size,
                        len(batch))
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def close(self):
        self._client.close()


class KnowledgeBase:
    """
    Collection-based document retrieval.
    
    Each collection is a folder with its own FAISS index.
    Include DESCRIPTION.md in source folders for agent discovery.
    
    Embedding backend:
        - "local" (default): Uses sentence-transformers locally. Fast, no API needed.
        - "api": Uses OpenAI-compatible API. Requires LLM_API_KEY env var.
    """
    
    FILE_TYPES = ["pdf", "md", "txt", "py", "docx", "json", "yaml", "yml"]
    
    def __init__(
        self,
        index_dir: str | Path,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        embedding_backend: str = "api",
        embedding_model: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_timeout: float = 300.0,
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Select embedding backend
        backend = os.getenv("DSAGT_EMBEDDING_BACKEND", embedding_backend).lower()
        if backend == "local":
            self._client = LocalEmbeddingClient(
                model=embedding_model or "BAAI/bge-base-en-v1.5",
                batch_size=embedding_batch_size or 256,
            )
        else:
            self._client = APIEmbeddingClient(
                model=embedding_model,
                timeout=embedding_timeout,
                batch_size=embedding_batch_size or 100,
            )
        
        self._reranker = None
        self._cache: dict[str, tuple] = {}
    
    @property
    def collections(self) -> list[str]:
        """Available collection names."""
        return [p.name for p in self.index_dir.iterdir() 
                if p.is_dir() and (p / "index.faiss").exists()]
    
    def list_collections(self) -> list[dict]:
        """Collections with descriptions for agent discovery."""
        result = []
        for name in self.collections:
            desc_path = self.index_dir / name / "DESCRIPTION.md"
            description = desc_path.read_text() if desc_path.exists() else ""
            result.append({"name": name, "description": description})
        return result
    
    def ingest(self, folder: str | Path, collection_name: str | None = None, file_types: list[str] | None = None) -> dict:
        """
        Ingest folder as collection. Collection name = folder name.
        
        Copies DESCRIPTION.md from source folder if present.
        """
        folder = Path(folder)
        collection = collection_name or folder.name
        file_types = file_types or self.FILE_TYPES
        
        # Setup collection directory
        coll_dir = self.index_dir / collection
        coll_dir.mkdir(exist_ok=True)
        
        # Copy description if present
        desc_src = folder / "DESCRIPTION.md"
        if desc_src.exists():
            (coll_dir / "DESCRIPTION.md").write_text(desc_src.read_text())
        
        # Collect and chunk files
        files = [f for ext in file_types for f in folder.glob(f"**/*.{ext}")]
        logger.info("Found %d files to process", len(files))
        
        chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        logger.info("Created %d chunks", len(chunks))
        
        if not chunks:
            return {"collection": collection, "files": 0, "chunks": 0}
        
        # Embed and normalize (with batching)
        embeddings = self._client.embed([c["text"] for c in chunks])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)
        
        # Create and save index (infer dimension from embeddings)
        import faiss
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
        faiss.write_index(index, str(coll_dir / "index.faiss"))
        
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
        """
        Append documents to an existing collection.

        Accepts a list of file or folder paths. Folders are expanded
        recursively using *file_types*. New chunks are embedded, added
        to the existing FAISS index, and appended to chunks.jsonl.

        Returns counts of files processed and chunks added.
        """
        import faiss

        file_types = file_types or self.FILE_TYPES

        # Load existing index + chunks (creates cache entry)
        index, existing_chunks = self._load(collection)
        coll_dir = self.index_dir / collection

        # Resolve paths → individual files
        files: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(
                    f for ext in file_types for f in p.glob(f"**/*.{ext}")
                )
            else:
                logger.warning("Path not found, skipping: %s", p)

        if not files:
            return {"collection": collection, "files": 0, "chunks_added": 0,
                    "total_chunks": len(existing_chunks)}

        logger.info("Appending %d files to collection '%s'", len(files), collection)

        # Chunk new files
        new_chunks = [
            chunk for f in files for chunk in self._chunk_file(f, collection)
        ]
        logger.info("Created %d new chunks", len(new_chunks))

        if not new_chunks:
            return {"collection": collection, "files": len(files),
                    "chunks_added": 0, "total_chunks": len(existing_chunks)}

        # Embed and normalize
        embeddings = self._client.embed([c["text"] for c in new_chunks])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)

        # Add to existing FAISS index
        index.add(embeddings)
        faiss.write_index(index, str(coll_dir / "index.faiss"))

        # Append to chunks file
        with open(coll_dir / "chunks.jsonl", "a") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk) + "\n")

        # Update cache
        all_chunks = existing_chunks + new_chunks
        self._cache[collection] = (index, all_chunks)

        return {
            "collection": collection,
            "files": len(files),
            "chunks_added": len(new_chunks),
            "total_chunks": len(all_chunks),
        }
    
    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[dict]:
        """Search a collection."""
        index, chunks = self._load(collection)
        
        # Embed and normalize query
        query_emb = self._client.embed([query])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # FAISS search
        search_k = min(top_k * 10 if rerank else top_k, len(chunks))
        scores, indices = index.search(query_emb.reshape(1, -1), search_k)
        
        results = [
            {"chunk": chunks[i], "score": float(scores[0][j])}
            for j, i in enumerate(indices[0]) if i >= 0
        ]
        
        if rerank and results:
            return self._rerank(query, results, top_k)
        return results[:top_k]
    
    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Rerank results using cross-encoder."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.rerank_model, max_length=512)
        
        pairs = [[query, r["chunk"]["text"]] for r in results]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [{**r, "rerank_score": float(s)} for r, s in ranked[:top_k]]
    
    def _chunk_file(self, path: Path, collection: str) -> Iterator[dict]:
        """Chunk file using format-appropriate parser."""
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
        """Select parser based on file type."""
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
    
    def _load(self, name: str):
        """Load collection (cached)."""
        if name in self._cache:
            return self._cache[name]
        
        coll_dir = self.index_dir / name
        if not (coll_dir / "index.faiss").exists():
            raise ValueError(f"Collection '{name}' not found")
        
        import faiss
        index = faiss.read_index(str(coll_dir / "index.faiss"))
        with open(coll_dir / "chunks.jsonl") as f:
            chunks = [json.loads(line) for line in f]
        
        self._cache[name] = (index, chunks)
        return index, chunks
    
    def close(self):
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()