"""Knowledge base for DSAGT agent."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter


CODE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".rs": "rust",
    ".go": "go", ".java": "java", ".cpp": "cpp", ".c": "c",
    ".rb": "ruby", ".php": "php",
}


class EmbeddingClient:
    """Generate text embeddings via OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,  # FIX: Increased from 60s to 300s (5 min)
        batch_size: int = 100,   # FIX: Add batching to avoid overloading API
    ):
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
        
        # FIX: Process in batches to avoid timeouts and memory issues
        if len(texts) <= self.batch_size:
            return self._embed_batch(texts)
        
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"    Embedding batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size} ({len(batch)} texts)...")
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
    """
    
    FILE_TYPES = ["pdf", "md", "txt", "py", "docx", "json", "yaml", "yml"]
    
    def __init__(
        self,
        index_dir: str | Path,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_timeout: float = 300.0,  # FIX: Configurable timeout
        embedding_batch_size: int = 100,   # FIX: Configurable batch size
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self._client = EmbeddingClient(
            timeout=embedding_timeout,
            batch_size=embedding_batch_size,
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
    
    def ingest(self, folder: str | Path, file_types: list[str] | None = None) -> dict:
        """
        Ingest folder as collection. Collection name = folder name.
        
        Copies DESCRIPTION.md from source folder if present.
        """
        folder = Path(folder)
        collection = folder.name
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
        print(f"  Found {len(files)} files to process...")
        
        chunks = [chunk for f in files for chunk in self._chunk_file(f, collection)]
        print(f"  Created {len(chunks)} chunks...")
        
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
    
    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        rerank: bool = True,
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
            print(f"    Warning: Could not read {path}: {e}")
            return
        
        if not docs:
            return
        
        file_type = path.suffix.lower()
        parser = self._get_parser(file_type)
        
        try:
            nodes = parser.get_nodes_from_documents(docs)
        except Exception as e:
            print(f"    Warning: Could not parse {path}: {e}")
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
