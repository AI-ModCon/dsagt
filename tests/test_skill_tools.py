"""
Tests for the skill MCP tools (save_skill, search_skills, install_skill,
add_skill_source, list_skill_sources).

The skill surface lives in :mod:`dsagt.mcp.skill_tools`; ``create_skill_server``
exposes just that concern for driving via the MCP helpers.  Handlers return a
mix of ``str`` (save/search/install) and ``dict`` (add/list sources), so the two
``call_tool`` helpers are both used.
"""

import json
from unittest.mock import MagicMock

import pytest

from dsagt.mcp.skill_tools import create_skill_server
from dsagt.registry import SkillRegistry
from mcp_helpers import call_tool_json, call_tool_sync


def _make_skill_server(tmp_path):
    """Create (server, skill_registry, kb) with a real local-embedding KB.

    The skill registry is rooted at ``<tmp>/runtime`` so save_skill writes to
    ``<tmp>/runtime/skills/<name>/`` — the project layer the agent natively
    discovers.
    """
    from dsagt.knowledge import KnowledgeBase

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(
        index_dir=tmp_path / "kb_index",
        default_embedder="local",
    )
    skill_reg = SkillRegistry(
        source_skills_dir=None,  # package default (empty bundled is fine)
        runtime_dir=str(runtime_dir),
        kb=kb,
    )
    server = create_skill_server(skill_reg, kb, runtime_dir=str(runtime_dir))
    return server, skill_reg, kb


# ---------------------------------------------------------------------------
# save_skill
# ---------------------------------------------------------------------------


class TestSaveSkill:

    def test_add_new_skill_creates_files_and_indexes(self, tmp_path):
        """save_skill writes SKILL.md and the skill count goes up by one.

        The count includes any bundled skills that ship in the package (see
        SkillRegistry.list_skills which merges bundled + project layers), so we
        assert the file was created and the count incremented rather than
        equality on a specific number.
        """
        server, skill_reg, kb = _make_skill_server(tmp_path)
        before = len(skill_reg.list_skills())

        spec = {
            "name": "csv_inspector",
            "description": "Workflow for inspecting CSV columns and quality",
            "tags": ["data_management", "quality_control"],
        }
        body = "# csv_inspector\n\nFirst, run head on the file.  Then check nulls.\n"
        text = call_tool_sync(server, "save_skill", {"spec": spec, "body": body})

        assert "added" in text
        skill_md = tmp_path / "runtime" / "skills" / "csv_inspector" / "SKILL.md"
        assert skill_md.exists()
        content = skill_md.read_text()
        assert "csv_inspector" in content
        assert "First, run head" in content
        after = len(skill_reg.list_skills())
        assert after == before + 1

    def test_update_existing_skill_preserves_body_when_omitted(self, tmp_path):
        """Saving a spec for an existing skill without body keeps the body."""
        server, skill_reg, kb = _make_skill_server(tmp_path)
        first_body = "# orig\n\nOriginal workflow body.\n"
        call_tool_sync(
            server,
            "save_skill",
            {
                "spec": {"name": "wf", "description": "v1"},
                "body": first_body,
            },
        )
        # Update the description only — body should be preserved.
        text = call_tool_sync(
            server,
            "save_skill",
            {
                "spec": {"name": "wf", "description": "v2 description"},
            },
        )
        assert "updated" in text
        skill_md = tmp_path / "runtime" / "skills" / "wf" / "SKILL.md"
        content = skill_md.read_text()
        assert "v2 description" in content
        assert "Original workflow body" in content

    def test_save_skill_writes_reference_files(self, tmp_path):
        """reference_files dict lands as additional files in the skill dir."""
        server, skill_reg, kb = _make_skill_server(tmp_path)
        text = call_tool_sync(
            server,
            "save_skill",
            {
                "spec": {"name": "with_template", "description": "Has a template"},
                "body": "# with_template\n\nUses template.json.\n",
                "reference_files": {"template.json": '{"foo": "bar"}\n'},
            },
        )
        assert "added" in text
        skill_dir = tmp_path / "runtime" / "skills" / "with_template"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "template.json").read_text() == '{"foo": "bar"}\n'

    def test_save_skill_string_encoded_spec(self, tmp_path):
        """MCP clients that JSON-encode nested object args still work."""
        server, skill_reg, kb = _make_skill_server(tmp_path)
        spec_json = json.dumps({"name": "s1", "description": "d"})
        text = call_tool_sync(server, "save_skill", {"spec": spec_json, "body": "x"})
        assert "added" in text


# ---------------------------------------------------------------------------
# search_skills
# ---------------------------------------------------------------------------


class TestSearchSkills:

    def test_search_skills_empty_catalog_hints_to_sync(self, tmp_path):
        """With no catalog synced, search_skills explains how to enable one
        instead of returning a bare 'no match' the agent reads as exhausted."""
        server, skill_reg, kb = _make_skill_server(tmp_path)

        text = call_tool_sync(server, "search_skills", {"query": "vasp pymatgen dft"})
        assert "No catalog skills found" in text
        assert "no external skill catalog is synced" in text.lower()
        assert "add_skill_source" in text


# ---------------------------------------------------------------------------
# install_skill
# ---------------------------------------------------------------------------


class TestInstallSkill:

    def test_install_skill_routes_and_reports_missing(self, tmp_path):
        """install_skill is registered and reports a clean error when the
        named skill isn't in any synced catalog."""
        server, skill_reg, kb = _make_skill_server(tmp_path)
        text = call_tool_sync(
            server,
            "install_skill",
            {"skill_name": "zzz-definitely-not-a-real-skill-xyz"},
        )
        assert "No catalog skill" in text


# ---------------------------------------------------------------------------
# skill sources (add_skill_source / list_skill_sources)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kb(tmp_path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.collections = []
    return kb


class TestSkillSources:

    def test_list_skill_sources_returns_known(self, mock_kb):
        server = create_skill_server(kb=mock_kb)
        result = call_tool_json(server, "list_skill_sources", {})
        assert "k-dense-ai" in result["sources"]
        # Nothing synced → every known source flagged available, not synced.
        assert result["sources"]["k-dense-ai"]["synced"] is False
        assert result["sources"]["k-dense-ai"]["indexed"] == 0
        assert result["other_synced_collections"] == []
        assert "k-dense-ai" in result["note"]

    def test_add_skill_source_bad_source_errors(self, mock_kb):
        server = create_skill_server(kb=mock_kb)
        result = call_tool_json(
            server, "add_skill_source", {"source": "not-a-real-known-name"}
        )
        assert "error" in result
