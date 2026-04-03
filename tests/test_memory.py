"""
Tests for the explicit memory store.

Drop this file into tests/test_memory.py
"""

import yaml
import pytest

from dsagt.memory import ExplicitMemory


@pytest.fixture
def mem(tmp_path):
    return ExplicitMemory(runtime_dir=tmp_path)


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


class TestRemember:

    def test_stores_a_fact(self, mem):
        result = mem.remember("fastp uses Q20 by default")

        assert result["stored"] is True
        assert result["entry_id"]
        assert result["superseded_id"] is None

    def test_fact_persists_to_file(self, mem):
        mem.remember("fastp uses Q20")

        entries = mem.get_all()
        assert len(entries) == 1
        assert entries[0]["text"] == "fastp uses Q20"

    def test_stores_metadata(self, mem):
        mem.remember(
            "threshold is Q20",
            category="quality_control",
            session_id="sess_01",
        )

        entry = mem.get_all()[0]
        assert entry["category"] == "quality_control"
        assert entry["session_id"] == "sess_01"
        assert entry["timestamp"]
        assert entry["id"]

    def test_multiple_facts(self, mem):
        mem.remember("fact one")
        mem.remember("fact two")
        mem.remember("fact three")

        assert mem.count() == 3

    def test_unique_ids(self, mem):
        r1 = mem.remember("fact one")
        r2 = mem.remember("fact two")

        assert r1["entry_id"] != r2["entry_id"]

    def test_default_metadata(self, mem):
        mem.remember("simple fact")

        entry = mem.get_all()[0]
        assert entry["category"] == ""
        assert entry["session_id"] == ""


# ---------------------------------------------------------------------------
# supersede
# ---------------------------------------------------------------------------


class TestSupersede:

    def test_supersede_replaces_entry(self, mem):
        r1 = mem.remember("threshold is Q20")
        r2 = mem.remember("threshold is Q30", supersedes=r1["entry_id"])

        assert r2["stored"] is True
        assert r2["superseded_id"] == r1["entry_id"]
        assert mem.count() == 1
        assert mem.get_all()[0]["text"] == "threshold is Q30"

    def test_supersede_nonexistent_returns_error(self, mem):
        mem.remember("some fact")
        result = mem.remember("new fact", supersedes="nonexistent_id")

        assert result["stored"] is False
        assert "not found" in result["error"]
        # Original fact should be untouched
        assert mem.count() == 1

    def test_superseded_entry_in_history(self, mem, tmp_path):
        r1 = mem.remember("old fact")
        mem.remember("new fact", supersedes=r1["entry_id"])

        history_path = tmp_path / ExplicitMemory.HISTORY_FILENAME
        assert history_path.exists()
        history = yaml.safe_load(history_path.read_text())
        assert len(history) == 1
        assert history[0]["id"] == r1["entry_id"]
        assert history[0]["superseded_by"]

    def test_supersede_preserves_other_entries(self, mem):
        r1 = mem.remember("fact one")
        r2 = mem.remember("fact two")
        mem.remember("fact one updated", supersedes=r1["entry_id"])

        entries = mem.get_all()
        assert len(entries) == 2
        texts = {e["text"] for e in entries}
        assert "fact two" in texts
        assert "fact one updated" in texts
        assert "fact one" not in texts


# ---------------------------------------------------------------------------
# get_all / get_by_id
# ---------------------------------------------------------------------------


class TestRetrieval:

    def test_get_all_empty(self, mem):
        assert mem.get_all() == []

    def test_get_by_id(self, mem):
        r = mem.remember("a fact")
        entry = mem.get_by_id(r["entry_id"])

        assert entry is not None
        assert entry["text"] == "a fact"

    def test_get_by_id_nonexistent(self, mem):
        assert mem.get_by_id("nope") is None

    def test_count_empty(self, mem):
        assert mem.count() == 0


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:

    def test_remove_entry(self, mem):
        r = mem.remember("temporary fact")
        result = mem.remove(r["entry_id"])

        assert result["removed"] is True
        assert mem.count() == 0

    def test_remove_nonexistent(self, mem):
        result = mem.remove("nope")

        assert result["removed"] is False
        assert "not found" in result["error"]

    def test_removed_entry_in_history(self, mem, tmp_path):
        r = mem.remember("to be removed")
        mem.remove(r["entry_id"])

        history_path = tmp_path / ExplicitMemory.HISTORY_FILENAME
        history = yaml.safe_load(history_path.read_text())
        assert len(history) == 1
        assert history[0]["removed_at"]

    def test_remove_preserves_other_entries(self, mem):
        r1 = mem.remember("keep this")
        r2 = mem.remember("remove this")
        mem.remove(r2["entry_id"])

        assert mem.count() == 1
        assert mem.get_all()[0]["id"] == r1["entry_id"]


# ---------------------------------------------------------------------------
# render_context
# ---------------------------------------------------------------------------


class TestRenderContext:

    def test_empty_returns_empty_string(self, mem):
        assert mem.render_context() == ""

    def test_renders_facts(self, mem):
        mem.remember("fastp uses Q20")
        mem.remember("MEGAHIT needs 8GB", category="resources")

        text = mem.render_context()
        assert "# Explicit Memories" in text
        assert "- fastp uses Q20" in text
        assert "- MEGAHIT needs 8GB [resources]" in text

    def test_render_excludes_superseded(self, mem):
        r1 = mem.remember("old fact")
        mem.remember("new fact", supersedes=r1["entry_id"])

        text = mem.render_context()
        assert "old fact" not in text
        assert "new fact" in text


# ---------------------------------------------------------------------------
# file handling edge cases
# ---------------------------------------------------------------------------


class TestFileEdgeCases:

    def test_creates_directory_on_first_write(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        mem = ExplicitMemory(runtime_dir=nested)
        mem.remember("a fact")

        assert nested.exists()
        assert mem.count() == 1

    def test_handles_empty_file(self, tmp_path):
        mem = ExplicitMemory(runtime_dir=tmp_path)
        (tmp_path / ExplicitMemory.FILENAME).write_text("")

        assert mem.get_all() == []
        assert mem.count() == 0

    def test_handles_corrupt_file(self, tmp_path):
        mem = ExplicitMemory(runtime_dir=tmp_path)
        (tmp_path / ExplicitMemory.FILENAME).write_text("not: valid: yaml: [")

        # Should not crash — yaml.safe_load may return a string or raise
        # Either way, get_all should handle it gracefully
        try:
            result = mem.get_all()
            assert isinstance(result, list)
        except yaml.YAMLError:
            pass  # also acceptable — fail fast on corruption

    def test_file_is_human_readable(self, mem, tmp_path):
        mem.remember("readable fact", category="test")

        content = (tmp_path / ExplicitMemory.FILENAME).read_text()
        assert "readable fact" in content
        assert "category" in content
