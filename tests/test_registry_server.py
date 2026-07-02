"""
Tests for the registry MCP server.

Tests tool handlers: save_code_spec, get_registry, search_registry,
read_file, run_command, http_request, install_dependencies.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
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

    Pre-populated tools are written into ``<runtime>/codes/`` — the
    single project layer every lookup reads.
    """
    runtime_dir = tmp_path / "runtime"
    project_tools_dir = runtime_dir / "codes"
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
        """Non-JSON strings produce a clear error rather than crashing."""
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

    The previous behavior was to silently fall back to substring matching,
    which produced dramatically worse results than semantic search and hid
    real KB failures.  The new contract: exact-name lookup still works
    without a KB (it doesn't need one), but query-based semantic search
    returns a helpful error message asking the user to configure embedding
    credentials.
    """

    def test_exact_name_lookup_works_without_kb(self, populated_server):
        """code_name lookup is KB-free and must keep working."""
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
        """A semantic search request when no KB is configured must surface
        the missing-KB condition clearly, not silently degrade.

        The query "alpha" is a substring of the registered ``tool_alpha``;
        the deleted string-matching fallback would have returned it, so the
        ``not in`` assertion pins that the fallback stays gone.
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
# http_request
# ---------------------------------------------------------------------------


class TestHttpRequest:

    @patch("dsagt.mcp.registry_tools.httpx.AsyncClient")
    def test_success_with_defaults(self, mock_client_cls, server):
        """A plain URL issues a GET and returns status + body."""
        client = mock_client_cls.return_value.__aenter__.return_value
        client.request.return_value = MagicMock(status_code=200, text="pong")

        text = call_tool(
            server,
            "http_request",
            {"url": "https://example.test/ping"},
        )

        assert text == "Status: 200\n\npong"
        client.request.assert_awaited_once_with(
            method="GET",
            url="https://example.test/ping",
            headers={},
            timeout=30.0,
        )

    @patch("dsagt.mcp.registry_tools.httpx.AsyncClient")
    def test_method_and_headers_forwarded(self, mock_client_cls, server):
        """Explicit method and headers reach the client unchanged."""
        client = mock_client_cls.return_value.__aenter__.return_value
        client.request.return_value = MagicMock(status_code=201, text="created")

        text = call_tool(
            server,
            "http_request",
            {
                "url": "https://example.test/items",
                "method": "POST",
                "headers": {"Authorization": "Bearer tok"},
            },
        )

        assert text.startswith("Status: 201")
        client.request.assert_awaited_once_with(
            method="POST",
            url="https://example.test/items",
            headers={"Authorization": "Bearer tok"},
            timeout=30.0,
        )

    @patch("dsagt.mcp.registry_tools.httpx.AsyncClient")
    def test_transport_error_reported(self, mock_client_cls, server):
        """httpx transport failures surface as a clean error string."""
        client = mock_client_cls.return_value.__aenter__.return_value
        client.request.side_effect = httpx.ConnectError("Connection refused")

        text = call_tool(
            server,
            "http_request",
            {"url": "https://example.test/down"},
        )

        assert text == "Error making request: Connection refused"


# ---------------------------------------------------------------------------
# save_code_spec — dependency installation
# ---------------------------------------------------------------------------


class TestSaveToolSpecDependencies:

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_deps_installed_on_save(self, mock_run, server, registry):
        """When dependencies are provided, uv pip install is called."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Successfully installed pandas-2.1.0", stderr=""
        )
        spec = make_spec("tool-with-deps", dependencies=["pandas>=2.0", "numpy"])
        text = call_tool(server, "save_code_spec", {"spec": spec})

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

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_deps_failure_still_saves_spec(self, mock_run, server, registry):
        """Even if uv pip install fails, the spec is saved as a skill file."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No matching distribution for bogus-pkg"
        )
        spec = make_spec("tool-bad-deps", dependencies=["bogus-pkg"])
        text = call_tool(server, "save_code_spec", {"spec": spec})

        assert "added" in text
        assert "Installation failed" in text
        tool = registry.get_code("tool-bad-deps")
        assert tool is not None
        assert tool["dependencies"] == ["bogus-pkg"]

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_deps_timeout(self, mock_run, server):
        """Timeout during install is reported, spec is still saved."""
        mock_run.side_effect = subprocess.TimeoutExpired("uv", 120)
        spec = make_spec("tool-slow-deps", dependencies=["heavy-pkg"])
        text = call_tool(server, "save_code_spec", {"spec": spec})

        assert "added" in text
        assert "timed out" in text

    def test_no_deps_no_install_message(self, server, registry):
        """When no dependencies are provided, no install message appears."""
        spec = make_spec("tool-no-deps")
        text = call_tool(server, "save_code_spec", {"spec": spec})

        assert "added" in text
        assert "Dependency" not in text

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_deps_persisted_in_skill_file(self, mock_run, server, registry):
        """Dependencies are stored in the skill file frontmatter."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        spec = make_spec("dep-tool", dependencies=["requests>=2.28"])
        call_tool(server, "save_code_spec", {"spec": spec})

        tool = registry.get_code("dep-tool")
        assert tool["dependencies"] == ["requests>=2.28"]

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_uv_not_found(self, mock_run, server):
        """FileNotFoundError from missing uv is reported gracefully."""
        mock_run.side_effect = FileNotFoundError("uv")
        spec = make_spec("tool-no-uv", dependencies=["pandas"])
        text = call_tool(server, "save_code_spec", {"spec": spec})

        assert "added" in text
        assert "'uv' command not found" in text


# ---------------------------------------------------------------------------
# install_dependencies
# ---------------------------------------------------------------------------


class TestInstallDependencies:

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_install_all(self, mock_run, tmp_path):
        """install_dependencies with no code_name installs all unique deps."""
        server, reg = _make_server(
            tmp_path,
            tools=[
                make_spec("tool-a", dependencies=["pandas", "numpy"]),
                make_spec("tool-b", dependencies=["numpy", "scipy"]),
            ],
        )

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        text = call_tool(server, "install_dependencies", {})

        assert "tool-a" in text
        assert "tool-b" in text
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

    @patch("dsagt.mcp.registry_tools.subprocess.run")
    def test_install_single_tool(self, mock_run, tmp_path):
        """install_dependencies with code_name targets only that tool."""
        server, reg = _make_server(
            tmp_path,
            tools=[
                make_spec("tool-a", dependencies=["pandas"]),
                make_spec("tool-b", dependencies=["scipy"]),
            ],
        )

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        text = call_tool(server, "install_dependencies", {"code_name": "tool-b"})

        cmd = mock_run.call_args[0][0]
        assert cmd == ["uv", "pip", "install", "--python", sys.executable, "scipy"]
        assert "tool-b" in text
        assert "tool-a" not in text

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
    exercise the agent-saved code path.
    """
    from dsagt.knowledge import KnowledgeBase

    runtime_dir = tmp_path / "runtime"
    project_tools_dir = runtime_dir / "codes"
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
        """Saving a tool indexes it into the registered_tools collection."""
        from dsagt.registry import TOOL_REGISTRY_COLLECTION

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

        results = kb.search("filter", collection=TOOL_REGISTRY_COLLECTION)
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
