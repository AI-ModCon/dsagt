"""
Tests for the registry MCP server.

Tests tool handlers: save_code_spec, get_registry, search_registry,
readiness_reports, reconstruct_pipeline.
"""

from pathlib import Path

import pytest
import yaml

from dsagt.registry import CodeRegistry
from dsagt.mcp.registry_tools import create_registry_server
from mcp_helpers import call_tool_sync as call_tool


def make_spec(
    name="test-tool",
    description="A test tool",
    executable="echo hello",
    dependencies=None,
):
    """Create a minimal valid tool spec."""
    spec = {
        "name": name,
        "description": description,
        "executable": executable,
        "parameters": {
            "input": {
                "type": "string",
                "required": True,
                "description": "Input path",
            },
        },
    }
    if dependencies is not None:
        spec["dependencies"] = dependencies
    return spec


def _write_tool(codes_dir: Path, spec: dict) -> None:
    code_dir = codes_dir / spec["name"]
    code_dir.mkdir(parents=True, exist_ok=True)
    fm = yaml.dump(spec, default_flow_style=False, sort_keys=False)
    (code_dir / "SKILL.md").write_text(f"---\n{fm}---\n\n# {spec['name']}\n")


def _make_server(tmp_path, tools=None):
    """Create (server, registry) with optional pre-populated tools.

    Pre-populated codes are written into ``<runtime>/skills/``, the
    single project layer every lookup reads.
    """
    runtime_dir = tmp_path / "runtime"
    project_tools_dir = runtime_dir / "skills"
    project_tools_dir.mkdir(parents=True, exist_ok=True)
    for spec in tools or []:
        _write_tool(project_tools_dir, spec)
    reg = CodeRegistry(runtime_dir=str(runtime_dir))
    return create_registry_server(reg), reg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server_and_registry(tmp_path):
    return _make_server(tmp_path)


@pytest.fixture
def server(server_and_registry):
    return server_and_registry[0]


@pytest.fixture
def registry(server_and_registry):
    return server_and_registry[1]


@pytest.fixture
def populated(tmp_path):
    server, reg = _make_server(
        tmp_path,
        tools=[
            make_spec("tool-alpha", "Alpha tool", "python alpha.py"),
            make_spec("tool-beta", "Beta data processor", "python beta.py"),
        ],
    )
    return server, reg


@pytest.fixture
def populated_server(populated):
    return populated[0]


# ---------------------------------------------------------------------------
# save_code_spec
# ---------------------------------------------------------------------------


class TestSaveToolSpec:

    def test_add_new_tool(self, server, registry):
        """Saving a new spec creates a skill file and reports added."""
        spec = make_spec("my-tool")
        text = call_tool(server, "save_code_spec", {"spec": spec})

        assert "added" in text
        assert "1 tools" in text
        assert registry.get_code("my-tool") is not None

    def test_reply_states_the_stored_command(self, server, registry):
        """The reply carries the wrapped executable, since the agent runs
        what it reads and the command it supplied has no dsagt-run prefix."""
        text = call_tool(server, "save_code_spec", {"spec": make_spec("my-tool")})
        stored = registry.get_code("my-tool")["executable"]
        assert stored.startswith("dsagt-run --code my-tool -- ")
        assert f"Run it as: {stored}" in text

    def test_update_existing_tool(self, server, registry):
        """Saving a spec with the same name updates rather than duplicates."""
        call_tool(
            server,
            "save_code_spec",
            {"spec": make_spec("my-tool", description="Version 1")},
        )
        text = call_tool(
            server,
            "save_code_spec",
            {"spec": make_spec("my-tool", description="Version 2")},
        )

        assert "updated" in text
        assert "1 tools" in text
        assert registry.get_code("my-tool")["description"] == "Version 2"

    def test_add_multiple_tools(self, server, registry):
        """Multiple distinct tools accumulate as separate skill files."""
        call_tool(server, "save_code_spec", {"spec": make_spec("tool-a")})
        text = call_tool(server, "save_code_spec", {"spec": make_spec("tool-b")})

        assert "2 tools" in text
        assert registry.get_code("tool-a") is not None
        assert registry.get_code("tool-b") is not None

    def test_accepts_stringified_spec(self, server, registry):
        """Some MCP clients (Claude Sonnet/Haiku 4.x) send nested-object args as
        JSON strings.  The handler must accept both shapes."""
        import json

        spec = make_spec("stringy-tool")
        text = call_tool(server, "save_code_spec", {"spec": json.dumps(spec)})

        assert "added" in text
        assert registry.get_code("stringy-tool") is not None

    def test_rejects_invalid_stringified_spec(self, server, registry):
        """Non-JSON strings produce a clear error."""
        text = call_tool(server, "save_code_spec", {"spec": "not valid json {"})

        assert "Error" in text
        assert "JSON object" in text


# ---------------------------------------------------------------------------
# install_skill
# ---------------------------------------------------------------------------


class TestGetRegistry:

    def test_empty_registry(self, server):
        """Getting an empty registry reports empty."""
        text = call_tool(server, "get_registry", {})
        assert "empty" in text.lower()

    def test_populated_registry(self, populated_server, populated):
        _, reg = populated
        text = call_tool(populated_server, "get_registry", {})

        data = yaml.safe_load(text)
        assert len(data["codes"]) == 2
        names = [t["name"] for t in data["codes"]]
        assert "tool-alpha" in names
        assert "tool-beta" in names


# ---------------------------------------------------------------------------
# search_registry
# ---------------------------------------------------------------------------


class TestSearchRegistryNoKB:
    """search_registry with no KB configured.

    Exact-name lookup needs no KB and works without one; query-based
    semantic search returns an error asking the user to configure embedding
    credentials, so a missing KB is never hidden behind substring matching,
    which gives worse results than semantic search.
    """

    def test_exact_name_lookup_works_without_kb(self, populated_server):
        """code_name lookup needs no KB."""
        text = call_tool(
            populated_server, "search_registry", {"code_name": "tool-alpha"}
        )
        assert "tool-alpha" in text

    def test_exact_name_miss_without_kb(self, populated_server):
        """code_name with a non-existent name returns a clean 'no tool' message."""
        text = call_tool(
            populated_server, "search_registry", {"code_name": "nonexistent"}
        )
        assert "No tool named 'nonexistent'" in text

    def test_query_search_without_kb_returns_helpful_error(self, populated_server):
        """A semantic search request when no KB is configured reports the
        missing KB.

        The query "alpha" is a substring of the registered ``tool_alpha``; a
        substring fallback would return it, so the ``not in`` assertion pins
        that no such fallback runs.
        """
        text = call_tool(populated_server, "search_registry", {"query": "alpha"})
        assert "tool-alpha" not in text  # no silent substring fallback
        assert "knowledge base" in text.lower()
        assert "embedding" in text.lower()

    def test_empty_query_without_kb_returns_helpful_error(self, populated_server):
        """Empty query with no KB also surfaces the same error."""
        text = call_tool(populated_server, "search_registry", {})
        assert "knowledge base" in text.lower()


# ---------------------------------------------------------------------------
# save_code_spec: declared dependencies
# ---------------------------------------------------------------------------


class TestSaveToolSpecDependencies:

    def test_deps_ride_in_the_stored_executable(self, server, registry):
        """A spec's dependencies are kept and become the uv run prefix."""
        spec = make_spec("dep-tool", dependencies=["requests>=2.28"])
        text = call_tool(server, "save_code_spec", {"spec": spec})

        assert "added" in text
        tool = registry.get_code("dep-tool")
        assert tool["dependencies"] == ["requests>=2.28"]
        assert "uv run --with requests>=2.28 --" in tool["executable"]


# ---------------------------------------------------------------------------
# KB-backed tool indexing and search
# ---------------------------------------------------------------------------


def _make_server_with_kb(tmp_path, tools=None):
    """Create (server, registry, kb) with a real local-embedding KnowledgeBase.

    Pre-populated codes are written to ``<runtime>/skills/`` so they
    exercise the agent-saved code path.
    """
    from dsagt.knowledge import KnowledgeBase

    runtime_dir = tmp_path / "runtime"
    project_tools_dir = runtime_dir / "skills"
    project_tools_dir.mkdir(parents=True, exist_ok=True)
    for spec in tools or []:
        _write_tool(project_tools_dir, spec)

    kb = KnowledgeBase(
        index_dir=tmp_path / "kb_index",
        default_embedder="local",
    )
    reg = CodeRegistry(
        runtime_dir=str(runtime_dir),
        kb=kb,
    )
    server = create_registry_server(reg, kb)
    return server, reg, kb


class TestToolIndexing:
    """Tests for KB-backed tool registration and search."""

    def test_save_tool_indexes_into_kb(self, tmp_path):
        """Saving a code indexes it into the codes collection."""
        from dsagt.registry import CODES_COLLECTION

        server, reg, kb = _make_server_with_kb(tmp_path)

        call_tool(
            server,
            "save_code_spec",
            {
                "spec": make_spec(
                    name="csv-filter",
                    description="Filter CSV rows by column value",
                )
            },
        )

        results = kb.search("filter", collection=CODES_COLLECTION)
        assert len(results) > 0
        assert any("csv-filter" in r["chunk"].get("text", "") for r in results)

    def test_search_registry_semantic(self, tmp_path):
        """Semantic search finds tools by description similarity."""
        server, reg, kb = _make_server_with_kb(tmp_path)
        call_tool(
            server,
            "save_code_spec",
            {
                "spec": make_spec(
                    name="csv-filter",
                    description="Filter and remove rows from a CSV spreadsheet based on column values",
                )
            },
        )

        text = call_tool(
            server, "search_registry", {"query": "delete rows from tabular data"}
        )
        assert "csv-filter" in text

    def test_search_registry_by_tag(self, tmp_path):
        """Tag-based filtering returns only matching tools."""
        server, reg, kb = _make_server_with_kb(tmp_path)

        spec_genomics = make_spec(name="fastp", description="FASTQ preprocessor")
        spec_genomics["tags"] = ["genomics", "data_processing"]
        call_tool(server, "save_code_spec", {"spec": spec_genomics})

        spec_other = make_spec(name="csvtool", description="CSV processor")
        spec_other["tags"] = ["data_processing"]
        call_tool(server, "save_code_spec", {"spec": spec_other})

        text = call_tool(
            server, "search_registry", {"query": "tool", "tag": "genomics"}
        )
        assert "fastp" in text


def test_save_code_spec_under_an_installed_skills_name_is_refused(tmp_path):
    """Codes and skills share a directory; a code named after a skill would
    replace the skill's frontmatter."""
    from dsagt.mcp.registry_tools import create_registry_server
    from dsagt.skills import register_skill_scripts

    runtime = tmp_path / "rt"
    skill = runtime / "skills" / "vasp-to-isaac"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: vasp-to-isaac\ndescription: d\n---\nbody\n"
    )
    (skill / "scripts" / "convert.py").write_text("import argparse\n")
    register_skill_scripts(runtime, "vasp-to-isaac")
    server = create_registry_server(CodeRegistry(runtime_dir=str(runtime)))
    reply = call_tool(
        server,
        "save_code_spec",
        {
            "spec": {
                "name": "vasp-to-isaac",
                "description": "d",
                "executable": "python other.py",
                "parameters": {},
            }
        },
    )
    assert "is an installed skill" in reply and "vasp-to-isaac-convert" in reply
    assert "body" in (skill / "SKILL.md").read_text()
    assert "executable" not in (skill / "SKILL.md").read_text().split("---")[1]


def test_readiness_reports_gives_the_current_report_or_says_how_to_make_one(tmp_path):
    import asyncio
    import hashlib
    import json

    from dsagt.mcp.registry_tools import _handle_readiness_reports

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "t.csv").write_text("a\n1\n")
    (tmp_path / "trace_archive").mkdir()

    def ask():
        return asyncio.run(
            _handle_readiness_reports({"path": "data/t.csv"}, runtime_dir=tmp_path)
        )

    reply = ask()
    assert reply["current"] is None
    assert "Check it with the aidrin skill" in reply["next"]

    record = {
        "record_id": "r1",
        "code_name": "aidrin",
        "execution": {
            "exact_command": ["aidrin", "data-quality", "data/t.csv"],
            "return_code": 0,
            "stdout": '{"outliers": 0.03}',
            "timestamp_start": "2026-01-01T00:00:00Z",
            "input_files": ["data/t.csv"],
            "output_files": [],
            "file_hashes": {"data/t.csv": hashlib.sha256(b"a\n1\n").hexdigest()},
        },
    }
    (tmp_path / "trace_archive" / "aidrin_r1.json").write_text(json.dumps(record))
    reply = ask()
    assert reply["current"]["report"] == '{"outliers": 0.03}'
    assert "next" not in reply

    (tmp_path / "data" / "t.csv").write_text("a\n2\n")
    reply = ask()
    assert reply["current"] is None
    assert [r["record_id"] for r in reply["earlier"]] == ["r1"]
