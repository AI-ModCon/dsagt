"""
Tests for the registry MCP server.

Tests tool handlers: save_tool_spec, get_registry, search_registry,
read_file, run_command, install_dependencies.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from dsagt.registry import SkillRegistry, ToolRegistry
from dsagt.commands.registry_server import create_registry_server
from mcp_helpers import call_tool_sync as call_tool


def make_spec(
    name="test_tool",
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


def _write_tool(tools_dir: Path, spec: dict) -> None:
    path = tools_dir / f"{spec['name']}.md"
    fm = yaml.dump(spec, default_flow_style=False, sort_keys=False)
    path.write_text(f"---\n{fm}---\n\n# {spec['name']}\n")


def _make_server(tmp_path, tools=None):
    """Create (server, registry) with optional pre-populated tools.

    Pre-populated tools are written into ``<runtime>/tools/`` (the
    project layer) so they exercise the agent-saved code path —
    ``reindex_all`` and most lookups happen here.  The ``source_dir``
    is still passed as the bundled layer override but left empty;
    tests that need bundled-layer behavior populate it explicitly.
    """
    source_dir = tmp_path / "source_skills"
    source_dir.mkdir()
    runtime_dir = tmp_path / "runtime"
    project_tools_dir = runtime_dir / "tools"
    project_tools_dir.mkdir(parents=True, exist_ok=True)
    for spec in tools or []:
        _write_tool(project_tools_dir, spec)
    reg = ToolRegistry(
        source_tools_dir=str(source_dir),
        runtime_dir=str(runtime_dir),
    )
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
            make_spec("tool_alpha", "Alpha tool", "python alpha.py"),
            make_spec("tool_beta", "Beta data processor", "python beta.py"),
        ],
    )
    return server, reg


@pytest.fixture
def populated_server(populated):
    return populated[0]


# ---------------------------------------------------------------------------
# save_tool_spec
# ---------------------------------------------------------------------------


class TestSaveToolSpec:

    def test_add_new_tool(self, server, registry):
        """Saving a new spec creates a skill file and reports added."""
        spec = make_spec("my_tool")
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "1 tools" in text
        assert registry.get_tool("my_tool") is not None

    def test_update_existing_tool(self, server, registry):
        """Saving a spec with the same name updates rather than duplicates."""
        call_tool(
            server,
            "save_tool_spec",
            {"spec": make_spec("my_tool", description="Version 1")},
        )
        text = call_tool(
            server,
            "save_tool_spec",
            {"spec": make_spec("my_tool", description="Version 2")},
        )

        assert "updated" in text
        assert "1 tools" in text
        assert registry.get_tool("my_tool")["description"] == "Version 2"

    def test_add_multiple_tools(self, server, registry):
        """Multiple distinct tools accumulate as separate skill files."""
        call_tool(server, "save_tool_spec", {"spec": make_spec("tool_a")})
        text = call_tool(server, "save_tool_spec", {"spec": make_spec("tool_b")})

        assert "2 tools" in text
        assert registry.get_tool("tool_a") is not None
        assert registry.get_tool("tool_b") is not None

    def test_accepts_stringified_spec(self, server, registry):
        """Some MCP clients (Claude Sonnet/Haiku 4.x) send nested-object args as
        JSON strings.  The handler must accept both shapes."""
        import json

        spec = make_spec("stringy_tool")
        text = call_tool(server, "save_tool_spec", {"spec": json.dumps(spec)})

        assert "added" in text
        assert registry.get_tool("stringy_tool") is not None

    def test_rejects_invalid_stringified_spec(self, server, registry):
        """Non-JSON strings produce a clear error rather than crashing."""
        text = call_tool(server, "save_tool_spec", {"spec": "not valid json {"})

        assert "Error" in text
        assert "JSON object" in text


# ---------------------------------------------------------------------------
# install_skill
# ---------------------------------------------------------------------------


class TestInstallSkill:

    def test_install_skill_routes_and_reports_missing(self, server):
        """install_skill is registered and reports a clean error when the
        named skill isn't in any synced catalog."""
        text = call_tool(
            server,
            "install_skill",
            {"skill_name": "zzz-definitely-not-a-real-skill-xyz"},
        )
        assert "No catalog skill" in text


# ---------------------------------------------------------------------------
# save_skill
# ---------------------------------------------------------------------------


class TestSaveSkill:

    def test_add_new_skill_creates_files_and_indexes(self, tmp_path):
        """save_skill writes SKILL.md and indexes when KB is configured.

        The skill count includes any bundled skills that ship in the
        package (see SkillRegistry.list_skills which merges bundled +
        project layers), so we assert the file was created and the
        count went up by one rather than equality on a specific number.
        """
        server, reg, kb = _make_server_with_kb(tmp_path)
        from dsagt.registry import SkillRegistry as _SR

        skill_reg = _SR(runtime_dir=str(tmp_path / "runtime"), kb=kb)
        before = len(skill_reg.list_skills())

        spec = {
            "name": "csv_inspector",
            "description": "Workflow for inspecting CSV columns and quality",
            "tags": ["data_management", "quality_control"],
        }
        body = "# csv_inspector\n\nFirst, run head on the file.  Then check nulls.\n"
        text = call_tool(server, "save_skill", {"spec": spec, "body": body})

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
        server, reg, kb = _make_server_with_kb(tmp_path)
        first_body = "# orig\n\nOriginal workflow body.\n"
        call_tool(
            server,
            "save_skill",
            {
                "spec": {"name": "wf", "description": "v1"},
                "body": first_body,
            },
        )
        # Update the description only — body should be preserved.
        text = call_tool(
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
        server, reg, kb = _make_server_with_kb(tmp_path)
        text = call_tool(
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
        server, reg, kb = _make_server_with_kb(tmp_path)
        spec_json = json.dumps({"name": "s1", "description": "d"})
        text = call_tool(server, "save_skill", {"spec": spec_json, "body": "x"})
        assert "added" in text


# ---------------------------------------------------------------------------
# get_registry
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
        assert len(data["tools"]) == 2
        names = [t["name"] for t in data["tools"]]
        assert "tool_alpha" in names
        assert "tool_beta" in names


# ---------------------------------------------------------------------------
# search_registry
# ---------------------------------------------------------------------------


class TestSearchRegistryNoKB:
    """search_registry with no KB configured.

    The previous behavior was to silently fall back to substring matching,
    which produced dramatically worse results than semantic search and hid
    real KB failures.  The new contract: exact-name lookup still works
    without a KB (it doesn't need one), but query-based semantic search
    returns a helpful error message asking the user to configure embedding
    credentials.
    """

    def test_exact_name_lookup_works_without_kb(self, populated_server):
        """tool_name lookup is KB-free and must keep working."""
        text = call_tool(
            populated_server, "search_registry", {"tool_name": "tool_alpha"}
        )
        assert "tool_alpha" in text

    def test_exact_name_miss_without_kb(self, populated_server):
        """tool_name with a non-existent name returns a clean 'no tool' message."""
        text = call_tool(
            populated_server, "search_registry", {"tool_name": "nonexistent"}
        )
        assert "No tool named 'nonexistent'" in text

    def test_query_search_without_kb_returns_helpful_error(self, populated_server):
        """A semantic search request when no KB is configured must surface
        the missing-KB condition clearly, not silently degrade."""
        text = call_tool(populated_server, "search_registry", {"query": "alpha"})
        assert "knowledge base" in text.lower()
        assert "embedding" in text.lower()

    def test_empty_query_without_kb_returns_helpful_error(self, populated_server):
        """Empty query with no KB also surfaces the same error."""
        text = call_tool(populated_server, "search_registry", {})
        assert "knowledge base" in text.lower()


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:

    def test_read_success(self, server, tmp_path):
        """Reading an existing file returns its contents."""
        test_file = tmp_path / "hello.txt"
        test_file.write_text("hello world")

        text = call_tool(server, "read_file", {"path": str(test_file)})
        assert text == "hello world"

    def test_read_missing_file(self, server):
        """Reading a nonexistent file returns an error message."""
        text = call_tool(server, "read_file", {"path": "/nonexistent/file.txt"})
        assert "Error reading file" in text


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


class TestRunCommand:

    def test_success(self, server):
        """Running a valid command returns its output."""
        text = call_tool(
            server,
            "run_command",
            {
                "command": "echo",
                "args": ["hello"],
            },
        )
        assert "hello" in text
        assert "Return code: 0" in text

    def test_command_not_found(self, server):
        """Running a nonexistent command returns not found error."""
        text = call_tool(
            server,
            "run_command",
            {
                "command": "nonexistent_command_xyz",
            },
        )
        assert "not found" in text

    def test_timeout(self, server):
        """A command that exceeds the timeout reports timeout."""
        text = call_tool(
            server,
            "run_command",
            {
                "command": "sleep",
                "args": ["30"],
                "timeout": 0.1,
            },
        )
        assert "timed out" in text


# ---------------------------------------------------------------------------
# save_tool_spec — dependency installation
# ---------------------------------------------------------------------------


class TestSaveToolSpecDependencies:

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_deps_installed_on_save(self, mock_run, server, registry):
        """When dependencies are provided, uv pip install is called."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Successfully installed pandas-2.1.0", stderr=""
        )
        spec = make_spec("tool_with_deps", dependencies=["pandas>=2.0", "numpy"])
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "Successfully installed" in text
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "pandas>=2.0",
            "numpy",
        ]

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_deps_failure_still_saves_spec(self, mock_run, server, registry):
        """Even if uv pip install fails, the spec is saved as a skill file."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No matching distribution for bogus-pkg"
        )
        spec = make_spec("tool_bad_deps", dependencies=["bogus-pkg"])
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "Installation failed" in text
        tool = registry.get_tool("tool_bad_deps")
        assert tool is not None
        assert tool["dependencies"] == ["bogus-pkg"]

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_deps_timeout(self, mock_run, server):
        """Timeout during install is reported, spec is still saved."""
        mock_run.side_effect = subprocess.TimeoutExpired("uv", 120)
        spec = make_spec("tool_slow_deps", dependencies=["heavy-pkg"])
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "timed out" in text

    def test_no_deps_no_install_message(self, server, registry):
        """When no dependencies are provided, no install message appears."""
        spec = make_spec("tool_no_deps")
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "Dependency" not in text

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_deps_persisted_in_skill_file(self, mock_run, server, registry):
        """Dependencies are stored in the skill file frontmatter."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        spec = make_spec("dep_tool", dependencies=["requests>=2.28"])
        call_tool(server, "save_tool_spec", {"spec": spec})

        tool = registry.get_tool("dep_tool")
        assert tool["dependencies"] == ["requests>=2.28"]

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_uv_not_found(self, mock_run, server):
        """FileNotFoundError from missing uv is reported gracefully."""
        mock_run.side_effect = FileNotFoundError("uv")
        spec = make_spec("tool_no_uv", dependencies=["pandas"])
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "'uv' command not found" in text


# ---------------------------------------------------------------------------
# install_dependencies
# ---------------------------------------------------------------------------


class TestInstallDependencies:

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_install_all(self, mock_run, tmp_path):
        """install_dependencies with no tool_name installs all unique deps."""
        server, reg = _make_server(
            tmp_path,
            tools=[
                make_spec("tool_a", dependencies=["pandas", "numpy"]),
                make_spec("tool_b", dependencies=["numpy", "scipy"]),
            ],
        )

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        text = call_tool(server, "install_dependencies", {})

        assert "tool_a" in text
        assert "tool_b" in text
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "pandas",
            "numpy",
            "scipy",
        ]

    @patch("dsagt.commands.registry_server.subprocess.run")
    def test_install_single_tool(self, mock_run, tmp_path):
        """install_dependencies with tool_name targets only that tool."""
        server, reg = _make_server(
            tmp_path,
            tools=[
                make_spec("tool_a", dependencies=["pandas"]),
                make_spec("tool_b", dependencies=["scipy"]),
            ],
        )

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        text = call_tool(server, "install_dependencies", {"tool_name": "tool_b"})

        cmd = mock_run.call_args[0][0]
        assert cmd == ["uv", "pip", "install", "--python", sys.executable, "scipy"]
        assert "tool_b" in text
        assert "tool_a" not in text

    def test_no_deps_in_registry(self, server):
        """install_dependencies on empty registry reports no tools."""
        text = call_tool(server, "install_dependencies", {})
        assert "empty" in text.lower() or "No tools" in text

    def test_tools_without_deps(self, tmp_path):
        """Tools without dependencies field are skipped gracefully."""
        server, reg = _make_server(tmp_path, tools=[make_spec("nodep_tool")])

        text = call_tool(server, "install_dependencies", {})
        assert "No dependencies" in text


# ---------------------------------------------------------------------------
# KB-backed tool indexing and search
# ---------------------------------------------------------------------------


def _make_server_with_kb(tmp_path, tools=None):
    """Create (server, registry, kb) with a real local-embedding KnowledgeBase.

    Pre-populated tools are written to ``<runtime>/tools/`` so they
    exercise the agent-saved code path that ``reindex_all`` operates on.
    """
    from dsagt.knowledge import KnowledgeBase

    source_dir = tmp_path / "source_skills"
    source_dir.mkdir()
    runtime_dir = tmp_path / "runtime"
    project_tools_dir = runtime_dir / "tools"
    project_tools_dir.mkdir(parents=True, exist_ok=True)
    for spec in tools or []:
        _write_tool(project_tools_dir, spec)

    kb = KnowledgeBase(
        index_dir=tmp_path / "kb_index",
        default_embedder="local",
        default_index="chroma",
    )
    reg = ToolRegistry(
        source_tools_dir=str(source_dir),
        runtime_dir=str(runtime_dir),
        kb=kb,
    )
    skill_reg = SkillRegistry(
        source_skills_dir=None,  # use package default (empty bundled is fine)
        runtime_dir=str(runtime_dir),
        kb=kb,
    )
    server = create_registry_server(reg, kb, skill_reg)
    return server, reg, kb


class TestToolIndexing:
    """Tests for KB-backed tool registration and search."""

    def test_save_tool_indexes_into_kb(self, tmp_path):
        """Saving a tool indexes it into the registered_tools collection."""
        from dsagt.registry import TOOL_REGISTRY_COLLECTION

        server, reg, kb = _make_server_with_kb(tmp_path)

        call_tool(
            server,
            "save_tool_spec",
            {
                "spec": make_spec(
                    name="csv_filter",
                    description="Filter CSV rows by column value",
                )
            },
        )

        results = kb.search("filter", collection=TOOL_REGISTRY_COLLECTION)
        assert len(results) > 0
        assert any("csv_filter" in r["chunk"].get("text", "") for r in results)

    def test_search_skills_empty_catalog_hints_to_sync(self, tmp_path):
        """With no catalog synced, search_skills explains how to enable one
        instead of returning a bare 'no match' the agent reads as exhausted."""
        server, reg, kb = _make_server_with_kb(tmp_path)

        text = call_tool(server, "search_skills", {"query": "vasp pymatgen dft"})
        assert "No catalog skills found" in text
        assert "no external skill catalog is synced" in text.lower()
        assert "add_skill_source" in text

    def test_search_registry_by_name(self, tmp_path):
        """Exact tool_name lookup returns the tool."""
        server, reg, kb = _make_server_with_kb(tmp_path)
        call_tool(server, "save_tool_spec", {"spec": make_spec(name="fastp")})

        text = call_tool(server, "search_registry", {"tool_name": "fastp"})
        assert "fastp" in text

    def test_search_registry_by_name_not_found(self, tmp_path):
        """Exact lookup for nonexistent tool returns not found."""
        server, reg, kb = _make_server_with_kb(tmp_path)

        text = call_tool(server, "search_registry", {"tool_name": "nonexistent"})
        assert "No tool" in text

    def test_search_registry_semantic(self, tmp_path):
        """Semantic search finds tools by description similarity."""
        server, reg, kb = _make_server_with_kb(tmp_path)
        call_tool(
            server,
            "save_tool_spec",
            {
                "spec": make_spec(
                    name="csv_filter",
                    description="Filter and remove rows from a CSV spreadsheet based on column values",
                )
            },
        )

        text = call_tool(
            server, "search_registry", {"query": "delete rows from tabular data"}
        )
        assert "csv_filter" in text

    def test_search_registry_by_tag(self, tmp_path):
        """Tag-based filtering returns only matching tools."""
        server, reg, kb = _make_server_with_kb(tmp_path)

        spec_genomics = make_spec(name="fastp", description="FASTQ preprocessor")
        spec_genomics["tags"] = ["genomics", "data_processing"]
        call_tool(server, "save_tool_spec", {"spec": spec_genomics})

        spec_other = make_spec(name="csvtool", description="CSV processor")
        spec_other["tags"] = ["data_processing"]
        call_tool(server, "save_tool_spec", {"spec": spec_other})

        text = call_tool(
            server, "search_registry", {"query": "tool", "tag": "genomics"}
        )
        assert "fastp" in text

    def test_reindex_all(self, tmp_path):
        """reindex_all populates KB from existing skill files."""
        from dsagt.registry import TOOL_REGISTRY_COLLECTION

        server, reg, kb = _make_server_with_kb(
            tmp_path,
            tools=[
                make_spec(name="preexisting", description="Already registered tool")
            ],
        )

        # Skills were copied to runtime on init but not indexed (KB was empty)
        # reindex_all should pick them up
        count = reg.reindex_all()
        assert count >= 1

        results = kb.search("registered", collection=TOOL_REGISTRY_COLLECTION)
        assert len(results) > 0

    def test_no_kb_query_search_returns_explicit_error(self, tmp_path):
        """Without a configured KB, query-based search MUST NOT silently
        fall back to substring matching.  It must return an explicit
        error so the user knows the KB is missing.

        Regression test for the deletion of the string-matching fallback
        in search_registry.  The old fallback hid embedding/KB failures
        and produced dramatically worse search results without telling
        anyone.
        """
        server, reg = _make_server(
            tmp_path,
            tools=[
                make_spec(name="csv_filter", description="Filter CSV rows"),
            ],
        )

        text = call_tool(server, "search_registry", {"query": "csv"})
        assert "csv_filter" not in text  # the substring match must NOT happen
        assert "knowledge base" in text.lower()
