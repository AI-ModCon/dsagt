"""
Tests for outlier detection, suggestion queue, and proxy injection.

Tests CategoryCentroids (incremental update, cosine distance, persistence),
SuggestionQueue (add, dismiss, persistence), check_and_queue_outliers
(threshold behavior, first-in-category skip), and the proxy injection
mechanism (pending_injection, dedup).
"""

import json

import numpy as np
import pytest

from dsagt.memory_extraction import CategoryCentroids, SuggestionQueue, check_and_queue_outliers
from dsagt.proxy_callback import ToolRecordStore


# ---------------------------------------------------------------------------
# CategoryCentroids
# ---------------------------------------------------------------------------

class TestCategoryCentroids:

    def test_first_entry_returns_zero_distance(self, tmp_path):
        """First entry in a category has no centroid to compare against."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        vec = np.array([1.0, 0.0, 0.0])
        distance = centroids.update("quality_control", vec)
        assert distance == 0.0
        assert centroids.count("quality_control") == 1

    def test_identical_vectors_zero_distance(self, tmp_path):
        """Identical vectors have zero distance."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        vec = np.array([1.0, 0.0, 0.0])
        centroids.update("qc", vec)
        distance = centroids.update("qc", vec)
        assert distance < 0.01

    def test_orthogonal_vectors_high_distance(self, tmp_path):
        """Orthogonal vectors have distance ~1.0."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        centroids.update("qc", np.array([1.0, 0.0, 0.0]))
        distance = centroids.update("qc", np.array([0.0, 1.0, 0.0]))
        assert distance > 0.9

    def test_incremental_update(self, tmp_path):
        """Centroid shifts toward new vectors incrementally."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        centroids.update("qc", np.array([1.0, 0.0, 0.0]))
        centroids.update("qc", np.array([1.0, 0.0, 0.0]))
        centroids.update("qc", np.array([1.0, 0.0, 0.0]))

        # Outlier vector — mostly in the y direction
        distance = centroids.update("qc", np.array([0.0, 1.0, 0.0]))
        assert distance > 0.8
        assert centroids.count("qc") == 4

    def test_save_and_reload(self, tmp_path):
        """Centroids persist across reloads."""
        path = tmp_path / "centroids.json"
        c1 = CategoryCentroids(path)
        c1.update("qc", np.array([1.0, 0.0]))
        c1.update("qc", np.array([0.9, 0.1]))
        c1.save()

        c2 = CategoryCentroids(path)
        assert c2.count("qc") == 2
        assert "qc" in c2.categories

    def test_separate_categories(self, tmp_path):
        """Different categories maintain independent centroids."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        centroids.update("qc", np.array([1.0, 0.0]))
        centroids.update("config", np.array([0.0, 1.0]))

        assert centroids.count("qc") == 1
        assert centroids.count("config") == 1


# ---------------------------------------------------------------------------
# SuggestionQueue
# ---------------------------------------------------------------------------

class TestSuggestionQueue:

    def test_add_returns_id(self, tmp_path):
        queue = SuggestionQueue(tmp_path / "suggestions.json")
        sid = queue.add("test fact", "qc", 0.45, "s1")
        assert sid.startswith("sug_")
        assert queue.count == 1

    def test_get_all(self, tmp_path):
        queue = SuggestionQueue(tmp_path / "suggestions.json")
        queue.add("fact 1", "qc", 0.4)
        queue.add("fact 2", "config", 0.5)
        suggestions = queue.get_all()
        assert len(suggestions) == 2
        assert suggestions[0]["text"] == "fact 1"
        assert suggestions[1]["category"] == "config"

    def test_dismiss_removes(self, tmp_path):
        queue = SuggestionQueue(tmp_path / "suggestions.json")
        sid = queue.add("test fact", "qc", 0.45)
        assert queue.dismiss(sid) is True
        assert queue.count == 0

    def test_dismiss_nonexistent(self, tmp_path):
        queue = SuggestionQueue(tmp_path / "suggestions.json")
        assert queue.dismiss("sug_nonexistent") is False

    def test_clear(self, tmp_path):
        queue = SuggestionQueue(tmp_path / "suggestions.json")
        queue.add("a", "qc", 0.3)
        queue.add("b", "qc", 0.4)
        removed = queue.clear()
        assert removed == 2
        assert queue.count == 0

    def test_persistence(self, tmp_path):
        """Suggestions survive reload."""
        path = tmp_path / "suggestions.json"
        q1 = SuggestionQueue(path)
        q1.add("persisted fact", "qc", 0.5)

        q2 = SuggestionQueue(path)
        assert q2.count == 1
        assert q2.get_all()[0]["text"] == "persisted fact"


# ---------------------------------------------------------------------------
# check_and_queue_outliers
# ---------------------------------------------------------------------------

class TestCheckAndQueueOutliers:

    def _make_embeddings(self, vectors):
        return np.array(vectors, dtype=np.float32)

    def test_flags_outlier(self, tmp_path):
        """A vector far from its category centroid gets flagged."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        queue = SuggestionQueue(tmp_path / "suggestions.json")

        # Seed the centroid with a few similar vectors
        for _ in range(5):
            centroids.update("qc", np.array([1.0, 0.0, 0.0]))
        centroids.save()

        texts = ["outlier observation"]
        categories = ["qc"]
        embeddings = self._make_embeddings([[0.0, 1.0, 0.0]])  # orthogonal

        ids = check_and_queue_outliers(
            texts, categories, embeddings, centroids, queue,
            threshold=0.3, session_id="s1",
        )

        assert len(ids) == 1
        assert queue.count == 1

    def test_normal_fact_not_flagged(self, tmp_path):
        """A vector close to its category centroid is not flagged."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        queue = SuggestionQueue(tmp_path / "suggestions.json")

        for _ in range(5):
            centroids.update("qc", np.array([1.0, 0.0, 0.0]))
        centroids.save()

        texts = ["normal observation"]
        categories = ["qc"]
        embeddings = self._make_embeddings([[0.98, 0.1, 0.0]])  # close

        ids = check_and_queue_outliers(
            texts, categories, embeddings, centroids, queue,
            threshold=0.3, session_id="s1",
        )

        assert len(ids) == 0
        assert queue.count == 0

    def test_first_in_category_not_flagged(self, tmp_path):
        """First entry in a category cannot be an outlier (no centroid yet)."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        queue = SuggestionQueue(tmp_path / "suggestions.json")

        texts = ["first observation"]
        categories = ["new_category"]
        embeddings = self._make_embeddings([[0.5, 0.5, 0.5]])

        ids = check_and_queue_outliers(
            texts, categories, embeddings, centroids, queue,
            threshold=0.1, session_id="s1",
        )

        assert len(ids) == 0

    def test_empty_category_skipped(self, tmp_path):
        """Facts with empty category are skipped."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        queue = SuggestionQueue(tmp_path / "suggestions.json")

        texts = ["uncategorized"]
        categories = [""]
        embeddings = self._make_embeddings([[1.0, 0.0, 0.0]])

        ids = check_and_queue_outliers(
            texts, categories, embeddings, centroids, queue,
            threshold=0.1,
        )

        assert len(ids) == 0

    def test_centroids_saved(self, tmp_path):
        """Centroids are saved to disk after checking."""
        centroids = CategoryCentroids(tmp_path / "centroids.json")
        queue = SuggestionQueue(tmp_path / "suggestions.json")

        texts = ["fact"]
        categories = ["qc"]
        embeddings = self._make_embeddings([[1.0, 0.0]])

        check_and_queue_outliers(
            texts, categories, embeddings, centroids, queue,
            threshold=0.3,
        )

        assert (tmp_path / "centroids.json").exists()


# ---------------------------------------------------------------------------
# Proxy injection
# ---------------------------------------------------------------------------

class TestProxyInjection:

    def test_no_suggestions_returns_none(self, tmp_path):
        store = ToolRecordStore(records_dir=tmp_path / "trace_archive", session_id="test")
        assert store.pending_injection() is None

    def test_with_suggestions_returns_message(self, tmp_path):
        project_dir = tmp_path
        trace_dir = project_dir / "trace_archive"
        trace_dir.mkdir(parents=True)

        # Write suggestions
        suggestions = [{"id": "sug_1", "text": "unusual QC result", "category": "qc", "distance": 0.45}]
        (project_dir / "suggestions.json").write_text(json.dumps(suggestions))

        store = ToolRecordStore(records_dir=trace_dir, session_id="test")
        msg = store.pending_injection()

        assert msg is not None
        assert "DSAGT Memory System" in msg
        assert "unusual QC result" in msg
        assert "kb_get_suggestions" in msg

    def test_injection_fires_once(self, tmp_path):
        """Injection only fires once per suggestion batch."""
        project_dir = tmp_path
        trace_dir = project_dir / "trace_archive"
        trace_dir.mkdir(parents=True)

        suggestions = [{"id": "sug_1", "text": "fact", "category": "qc", "distance": 0.4}]
        (project_dir / "suggestions.json").write_text(json.dumps(suggestions))

        store = ToolRecordStore(records_dir=trace_dir, session_id="test")

        first = store.pending_injection()
        assert first is not None

        second = store.pending_injection()
        assert second is None  # same count, don't inject again

    def test_injection_fires_again_on_new_suggestions(self, tmp_path):
        """New suggestions trigger a fresh injection."""
        project_dir = tmp_path
        trace_dir = project_dir / "trace_archive"
        trace_dir.mkdir(parents=True)

        # First batch
        suggestions = [{"id": "sug_1", "text": "fact", "category": "qc", "distance": 0.4}]
        (project_dir / "suggestions.json").write_text(json.dumps(suggestions))

        store = ToolRecordStore(records_dir=trace_dir, session_id="test")
        store.pending_injection()  # consume

        # Second batch (more suggestions added)
        suggestions.append({"id": "sug_2", "text": "another", "category": "config", "distance": 0.5})
        (project_dir / "suggestions.json").write_text(json.dumps(suggestions))

        msg = store.pending_injection()
        assert msg is not None
        assert "2 new observation" in msg
