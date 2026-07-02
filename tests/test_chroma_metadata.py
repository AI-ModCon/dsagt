"""
Tests for Phase 2 Step 1: ChromaIndex metadata support and KnowledgeBase.add_entries.

Tests the metadata-aware add/search on ChromaIndex, the where parameter
threading through KnowledgeBase.search, and the new add_entries method
for structured entry ingestion.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dsagt.knowledge import (
    ChromaIndex,
    KnowledgeBase,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embeddings: hash-based so different texts get
    different (but reproducible) vectors."""
    dim = 8
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hash(t) & 0xFFFFFFFF
        rng = np.random.RandomState(h)
        vecs[i] = rng.randn(dim).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


# ---------------------------------------------------------------------------
# ChromaIndex — metadata on add
# ---------------------------------------------------------------------------


class TestChromaIndexMetadataAdd:

    def test_add_without_metadata(self):
        """add() without metadatas works as before."""
        idx = ChromaIndex(collection_name="test_no_meta")
        emb = np.random.randn(3, 8).astype(np.float32)
        idx.add(emb)
        assert idx.size == 3

    def test_add_with_metadata(self):
        """add() stores metadata that can be retrieved."""
        idx = ChromaIndex(collection_name="test_meta")
        emb = np.random.randn(3, 8).astype(np.float32)
        metadatas = [
            {"tool_name": "fastp", "session_id": "s1"},
            {"tool_name": "megahit", "session_id": "s1"},
            {"tool_name": "fastp", "session_id": "s2"},
        ]
        idx.add(emb, metadatas=metadatas)
        assert idx.size == 3

        # Verify metadata was stored by querying with a where clause
        results = idx._col.get(ids=["0", "1", "2"], include=["metadatas"])
        stored = results["metadatas"]
        assert stored[0]["tool_name"] == "fastp"
        assert stored[1]["tool_name"] == "megahit"
        assert stored[2]["session_id"] == "s2"

    def test_add_metadata_incremental(self):
        """Multiple add() calls with metadata produce correct ids."""
        idx = ChromaIndex(collection_name="test_incr")
        emb1 = np.random.randn(2, 8).astype(np.float32)
        emb2 = np.random.randn(2, 8).astype(np.float32)

        idx.add(emb1, metadatas=[{"batch": "a"}, {"batch": "a"}])
        idx.add(emb2, metadatas=[{"batch": "b"}, {"batch": "b"}])
        assert idx.size == 4

        results = idx._col.get(ids=["2", "3"], include=["metadatas"])
        assert all(m["batch"] == "b" for m in results["metadatas"])


# ---------------------------------------------------------------------------
# ChromaIndex — where filter on search
# ---------------------------------------------------------------------------


class TestChromaIndexWhereSearch:

    @pytest.fixture
    def indexed(self):
        """ChromaIndex with 4 entries, 2 tools × 2 sessions."""
        idx = ChromaIndex(collection_name="test_where")
        # Use distinct vectors so search results are deterministic
        emb = np.eye(4, 8, dtype=np.float32)
        metadatas = [
            {"tool_name": "fastp", "session_id": "s1"},
            {"tool_name": "megahit", "session_id": "s1"},
            {"tool_name": "fastp", "session_id": "s2"},
            {"tool_name": "megahit", "session_id": "s2"},
        ]
        idx.add(emb, metadatas=metadatas)
        return idx

    def test_search_without_where(self, indexed):
        """search() without where returns results from all entries."""
        query = np.ones(8, dtype=np.float32)
        query /= np.linalg.norm(query)
        scores, indices = indexed.search(query, k=4)
        assert len(scores) == 4

    def test_search_with_where_filters(self, indexed):
        """search() with where restricts results to matching entries."""
        query = np.ones(8, dtype=np.float32)
        query /= np.linalg.norm(query)
        scores, indices = indexed.search(query, k=4, where={"tool_name": "fastp"})
        # Should only return indices 0 and 2 (the fastp entries)
        assert set(indices.tolist()) == {0, 2}

    def test_search_with_compound_where(self, indexed):
        """search() with compound $and where clause works."""
        query = np.ones(8, dtype=np.float32)
        query /= np.linalg.norm(query)
        scores, indices = indexed.search(
            query,
            k=4,
            where={"$and": [{"tool_name": "megahit"}, {"session_id": "s2"}]},
        )
        assert set(indices.tolist()) == {3}

    def test_search_where_no_matches(self, indexed):
        """search() with a where that matches nothing returns empty arrays."""
        query = np.ones(8, dtype=np.float32)
        query /= np.linalg.norm(query)
        scores, indices = indexed.search(
            query,
            k=4,
            where={"tool_name": "nonexistent"},
        )
        assert len(scores) == 0
        assert len(indices) == 0

    def test_search_empty_index(self):
        """search() on empty index returns empty arrays."""
        idx = ChromaIndex(collection_name="test_empty")
        query = np.ones(8, dtype=np.float32)
        scores, indices = idx.search(query, k=5)
        assert len(scores) == 0
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# ChromaIndex — persistence with metadata
# ---------------------------------------------------------------------------


class TestChromaIndexPersistence:

    def test_save_load_preserves_metadata(self, tmp_path):
        """Metadata survives save/load cycle."""
        coll_dir = tmp_path / "test_persist"
        coll_dir.mkdir()

        idx = ChromaIndex(collection_name="test_persist", persist_dir=coll_dir)
        emb = np.random.randn(2, 8).astype(np.float32)
        idx.add(emb, metadatas=[{"tool": "a"}, {"tool": "b"}])
        idx.save(coll_dir)

        loaded = ChromaIndex.load(coll_dir)
        assert loaded.size == 2

        # Verify metadata survived
        results = loaded._col.get(ids=["0", "1"], include=["metadatas"])
        assert results["metadatas"][0]["tool"] == "a"
        assert results["metadatas"][1]["tool"] == "b"


# ---------------------------------------------------------------------------
# KnowledgeBase.search — where parameter
# ---------------------------------------------------------------------------


class TestKnowledgeBaseSearchWhere:

    @pytest.fixture
    def kb_with_chroma(self, tmp_path):
        """KnowledgeBase with a ChromaDB-backed collection containing
        entries with metadata."""
        index_dir = tmp_path / "kb"

        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=index_dir)

            # Manually build the collection with metadata
            kb.add_entries(
                texts=[
                    "fastp quality filtering completed on sample1",
                    "megahit assembly completed on sample1",
                    "fastp quality filtering completed on sample2",
                    "megahit assembly completed on sample2",
                ],
                collection="mem",
                metadatas=[
                    {"tool_name": "fastp", "session_id": "s1"},
                    {"tool_name": "megahit", "session_id": "s1"},
                    {"tool_name": "fastp", "session_id": "s2"},
                    {"tool_name": "megahit", "session_id": "s2"},
                ],
            )

            yield kb
            kb.close()

    def test_search_without_where(self, kb_with_chroma):
        """Search without where returns results from all entries."""
        results = kb_with_chroma.search(
            "fastp quality",
            collection="mem",
            top_k=4,
            rerank=False,
        )
        assert len(results) > 0

    def test_search_with_where_filters(self, kb_with_chroma):
        """Search with where restricts to matching metadata."""
        results = kb_with_chroma.search(
            "quality filtering",
            collection="mem",
            top_k=4,
            rerank=False,
            where={"tool_name": "fastp"},
        )
        # All results should be fastp entries
        for r in results:
            assert r["chunk"]["metadata"]["tool_name"] == "fastp"

    def test_search_with_session_filter(self, kb_with_chroma):
        """Search with session_id filter returns only that session."""
        results = kb_with_chroma.search(
            "assembly",
            collection="mem",
            top_k=4,
            rerank=False,
            where={"session_id": "s2"},
        )
        for r in results:
            assert r["chunk"]["metadata"]["session_id"] == "s2"


# ---------------------------------------------------------------------------
# KnowledgeBase.add_entries
# ---------------------------------------------------------------------------


class TestAddEntries:

    @pytest.fixture
    def mock_kb(self, tmp_path):
        """KnowledgeBase with mocked embedder."""
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder
            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            yield kb
            kb.close()

    def test_add_entries_creates_collection(self, mock_kb):
        """add_entries creates a new collection if it doesn't exist."""
        result = mock_kb.add_entries(
            texts=["fact one", "fact two"],
            collection="new_coll",
        )
        assert result["collection"] == "new_coll"
        assert result["entries_added"] == 2
        assert result["total_entries"] == 2
        assert "new_coll" in mock_kb.collections

    def test_add_entries_appends(self, mock_kb):
        """Successive add_entries calls accumulate entries."""
        mock_kb.add_entries(texts=["first"], collection="coll")
        result = mock_kb.add_entries(texts=["second", "third"], collection="coll")
        assert result["entries_added"] == 2
        assert result["total_entries"] == 3

    def test_add_entries_empty_list(self, mock_kb):
        """add_entries with empty texts is a no-op."""
        result = mock_kb.add_entries(texts=[], collection="empty")
        assert result["entries_added"] == 0
        assert result["total_entries"] == 0

    def test_add_entries_with_metadata(self, tmp_path):
        """add_entries stores metadata in chunks.jsonl."""
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(
                index_dir=tmp_path / "kb",
            )

            kb.add_entries(
                texts=["ran fastp with Q20"],
                collection="mem",
                metadatas=[{"tool_name": "fastp", "session_id": "s1"}],
            )

            # Verify chunks.jsonl includes metadata
            chunks_path = tmp_path / "kb" / "mem" / "chunks.jsonl"
            assert chunks_path.exists()
            chunk = json.loads(chunks_path.read_text().strip())
            assert chunk["metadata"]["tool_name"] == "fastp"
            assert chunk["metadata"]["session_id"] == "s1"
            assert chunk["metadata"]["collection"] == "mem"

            kb.close()

    def test_add_entries_searchable(self, mock_kb):
        """Entries added via add_entries are findable via search."""
        mock_kb.add_entries(
            texts=["the quick brown fox", "the lazy dog sleeps"],
            collection="test_search",
        )
        results = mock_kb.search(
            "quick fox",
            collection="test_search",
            top_k=2,
            rerank=False,
        )
        assert len(results) > 0

    def test_add_entries_return_embeddings_default_off(self, mock_kb):
        """By default, the returned dict has no 'embeddings' key."""
        result = mock_kb.add_entries(texts=["one", "two"], collection="ret_off")
        assert "embeddings" not in result

    def test_add_entries_return_embeddings_on(self, mock_kb):
        """With return_embeddings=True, the result includes the L2-normalized
        embedding matrix used for indexing.  Memory extraction uses this to
        skip a redundant embedding round-trip.
        """
        result = mock_kb.add_entries(
            texts=["one", "two", "three"],
            collection="ret_on",
            return_embeddings=True,
        )
        assert "embeddings" in result
        embeddings = result["embeddings"]
        assert isinstance(embeddings, np.ndarray)
        # fake_embed produces 8-dimensional vectors.
        assert embeddings.shape == (3, 8)
        # Embeddings should be L2-normalized (unit vectors).
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_add_entries_uses_chroma(self, tmp_path):
        """add_entries writes a ChromaDB-backed collection (chroma_ids.json)."""
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(index_dir=tmp_path / "kb")
            kb.add_entries(
                texts=["test entry"],
                collection="chroma_coll",
            )

            # Verify it used ChromaDB (chroma_ids.json exists)
            coll_dir = tmp_path / "kb" / "chroma_coll"
            assert (coll_dir / "chroma_ids.json").exists()

            kb.close()


# ---------------------------------------------------------------------------
# End-to-end: add_entries + filtered search
# ---------------------------------------------------------------------------


class TestAddEntriesFilteredSearch:
    """Integration test: add entries with metadata, then search with where."""

    def test_filtered_search_returns_correct_subset(self, tmp_path):
        with patch("dsagt.knowledge.Embedder.create") as mock_make:
            mock_embedder = MagicMock()
            mock_embedder.embed = fake_embed
            mock_make.return_value = mock_embedder

            kb = KnowledgeBase(
                index_dir=tmp_path / "kb",
            )

            kb.add_entries(
                texts=[
                    "fastp -q 20 -l 50 --in1 sample1.fq.gz",
                    "megahit -1 sample1_R1.fq -2 sample1_R2.fq -o assembly1",
                    "fastp -q 30 -l 75 --in1 sample2.fq.gz",
                    "megahit -1 sample2_R1.fq -2 sample2_R2.fq -o assembly2",
                ],
                collection="executions",
                metadatas=[
                    {"tool_name": "fastp", "session_id": "s1", "return_code": 0},
                    {"tool_name": "megahit", "session_id": "s1", "return_code": 0},
                    {"tool_name": "fastp", "session_id": "s2", "return_code": 0},
                    {"tool_name": "megahit", "session_id": "s2", "return_code": 1},
                ],
            )

            # Filter by tool_name
            fastp_results = kb.search(
                "quality filtering",
                collection="executions",
                top_k=4,
                rerank=False,
                where={"tool_name": "fastp"},
            )
            for r in fastp_results:
                assert r["chunk"]["metadata"]["tool_name"] == "fastp"

            # Filter by session
            s1_results = kb.search(
                "assembly",
                collection="executions",
                top_k=4,
                rerank=False,
                where={"session_id": "s1"},
            )
            for r in s1_results:
                assert r["chunk"]["metadata"]["session_id"] == "s1"

            # Compound filter
            failed = kb.search(
                "megahit",
                collection="executions",
                top_k=4,
                rerank=False,
                where={"$and": [{"tool_name": "megahit"}, {"return_code": 1}]},
            )
            for r in failed:
                assert r["chunk"]["metadata"]["tool_name"] == "megahit"
                assert r["chunk"]["metadata"]["return_code"] == 1

            kb.close()
