"""
Tests for the registry MCP server.

Tests the tool handlers exposed by create_registry_server:
save_tool_spec, get_registry, search_registry, read_file, run_command.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
import mcp.types as types

from dsagt.registry_server import create_registry_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_tool(server, name: str, arguments: dict) -> str:
    """Invoke a tool handler on an MCP server and return the response text."""
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    handler = server.request_handlers[types.CallToolRequest]
    result = asyncio.run(handler(req))
    return result.root.content[0].text


def make_spec(name="test_tool", description="A test tool", executable="echo hello",
              dependencies=None):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_path(tmp_path):
    """Path for a fresh registry file (does not exist yet)."""
    return tmp_path / "registry.yaml"


@pytest.fixture
def server(registry_path):
    """Registry builder server with an empty (nonexistent) registry."""
    return create_registry_server(registry_path)


@pytest.fixture
def populated_server(registry_path):
    """Registry builder server with two tools already registered."""
    registry = {
        "tools": [
            make_spec("tool_alpha", "Alpha tool", "python alpha.py"),
            make_spec("tool_beta", "Beta data processor", "python beta.py"),
        ],
    }
    registry_path.write_text(yaml.dump(registry, sort_keys=False))
    return create_registry_server(registry_path)


# ---------------------------------------------------------------------------
# save_tool_spec
# ---------------------------------------------------------------------------

class TestSaveToolSpec:

    def test_add_new_tool(self, server, registry_path):
        """Saving a new spec creates the registry and adds the tool."""
        spec = make_spec("my_tool")
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "1 tools" in text

        registry = yaml.safe_load(registry_path.read_text())
        assert len(registry["tools"]) == 1
        assert registry["tools"][0]["name"] == "my_tool"

    def test_update_existing_tool(self, server, registry_path):
        """Saving a spec with the same name updates rather than duplicates."""
        spec_v1 = make_spec("my_tool", description="Version 1")
        call_tool(server, "save_tool_spec", {"spec": spec_v1})

        spec_v2 = make_spec("my_tool", description="Version 2")
        text = call_tool(server, "save_tool_spec", {"spec": spec_v2})

        assert "updated" in text
        assert "1 tools" in text

        registry = yaml.safe_load(registry_path.read_text())
        assert registry["tools"][0]["description"] == "Version 2"

    def test_add_multiple_tools(self, server, registry_path):
        """Multiple distinct tools accumulate in the registry."""
        call_tool(server, "save_tool_spec", {"spec": make_spec("tool_a")})
        text = call_tool(server, "save_tool_spec", {"spec": make_spec("tool_b")})

        assert "2 tools" in text

        registry = yaml.safe_load(registry_path.read_text())
        names = [t["name"] for t in registry["tools"]]
        assert names == ["tool_a", "tool_b"]


# ---------------------------------------------------------------------------
# get_registry
# ---------------------------------------------------------------------------

class TestGetRegistry:

    def test_empty_registry(self, server):
        """Getting a nonexistent registry reports empty."""
        text = call_tool(server, "get_registry", {})
        assert "empty" in text.lower()

    def test_populated_registry(self, populated_server):
        """Getting a populated registry returns YAML with all tools."""
        text = call_tool(populated_server, "get_registry", {})

        data = yaml.safe_load(text)
        assert len(data["tools"]) == 2
        names = [t["name"] for t in data["tools"]]
        assert "tool_alpha" in names
        assert "tool_beta" in names


# ---------------------------------------------------------------------------
# search_registry
# ---------------------------------------------------------------------------

class TestSearchRegistry:

    def test_empty_registry(self, server):
        """Searching an empty registry reports empty."""
        text = call_tool(server, "search_registry", {"query": "anything"})
        assert "empty" in text.lower()

    def test_match_by_name(self, populated_server):
        """Query matching a tool name returns that tool."""
        text = call_tool(populated_server, "search_registry", {"query": "alpha"})
        assert "tool_alpha" in text
        assert "1 tool(s)" in text

    def test_match_by_description(self, populated_server):
        """Query matching a description returns that tool."""
        text = call_tool(populated_server, "search_registry", {"query": "data processor"})
        assert "tool_beta" in text

    def test_no_match(self, populated_server):
        """Query with no matches reports none found."""
        text = call_tool(populated_server, "search_registry", {"query": "zzzzz"})
        assert "No tools found" in text

    def test_empty_query_returns_all(self, populated_server):
        """An empty query returns all tools."""
        text = call_tool(populated_server, "search_registry", {})
        assert "2 tool(s)" in text


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
        text = call_tool(server, "run_command", {
            "command": "echo",
            "args": ["hello"],
        })
        assert "hello" in text
        assert "Return code: 0" in text

    def test_command_not_found(self, server):
        """Running a nonexistent command returns not found error."""
        text = call_tool(server, "run_command", {
            "command": "nonexistent_command_xyz",
        })
        assert "not found" in text

    def test_timeout(self, server):
        """A command that exceeds the timeout reports timeout."""
        text = call_tool(server, "run_command", {
            "command": "sleep",
            "args": ["30"],
            "timeout": 0.1,
        })
        assert "timed out" in text


# ---------------------------------------------------------------------------
# save_tool_spec — dependency installation
# ---------------------------------------------------------------------------

class TestSaveToolSpecDependencies:

    @patch("dsagt.registry_server.subprocess.run")
    def test_deps_installed_on_save(self, mock_run, server, registry_path):
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
        assert cmd == ["uv", "pip", "install", "--python", sys.executable,
                        "pandas>=2.0", "numpy"]

    @patch("dsagt.registry_server.subprocess.run")
    def test_deps_failure_still_saves_spec(self, mock_run, server, registry_path):
        """Even if uv pip install fails, the spec is saved to the registry."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No matching distribution for bogus-pkg"
        )
        spec = make_spec("tool_bad_deps", dependencies=["bogus-pkg"])
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "Installation failed" in text

        registry = yaml.safe_load(registry_path.read_text())
        assert registry["tools"][0]["name"] == "tool_bad_deps"
        assert registry["tools"][0]["dependencies"] == ["bogus-pkg"]

    @patch("dsagt.registry_server.subprocess.run")
    def test_deps_timeout(self, mock_run, server):
        """Timeout during install is reported, spec is still saved."""
        mock_run.side_effect = subprocess.TimeoutExpired("uv", 120)
        spec = make_spec("tool_slow_deps", dependencies=["heavy-pkg"])
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "timed out" in text

    def test_no_deps_no_install_message(self, server, registry_path):
        """When no dependencies are provided, no install message appears."""
        spec = make_spec("tool_no_deps")
        text = call_tool(server, "save_tool_spec", {"spec": spec})

        assert "added" in text
        assert "Dependency" not in text

    @patch("dsagt.registry_server.subprocess.run")
    def test_deps_persisted_in_yaml(self, mock_run, server, registry_path):
        """Dependencies are stored in the registry YAML for reproducibility."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        spec = make_spec("dep_tool", dependencies=["requests>=2.28"])
        call_tool(server, "save_tool_spec", {"spec": spec})

        registry = yaml.safe_load(registry_path.read_text())
        assert registry["tools"][0]["dependencies"] == ["requests>=2.28"]

    @patch("dsagt.registry_server.subprocess.run")
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

    @patch("dsagt.registry_server.subprocess.run")
    def test_install_all(self, mock_run, registry_path):
        """install_dependencies with no tool_name installs all unique deps."""
        registry = {
            "tools": [
                make_spec("tool_a", dependencies=["pandas", "numpy"]),
                make_spec("tool_b", dependencies=["numpy", "scipy"]),
            ],
        }
        registry_path.write_text(yaml.dump(registry, sort_keys=False))
        server = create_registry_server(registry_path)

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        text = call_tool(server, "install_dependencies", {})

        assert "tool_a" in text
        assert "tool_b" in text
        cmd = mock_run.call_args[0][0]
        # numpy should appear only once (deduplication)
        assert cmd == ["uv", "pip", "install", "--python", sys.executable,
                        "pandas", "numpy", "scipy"]

    @patch("dsagt.registry_server.subprocess.run")
    def test_install_single_tool(self, mock_run, registry_path):
        """install_dependencies with tool_name targets only that tool."""
        registry = {
            "tools": [
                make_spec("tool_a", dependencies=["pandas"]),
                make_spec("tool_b", dependencies=["scipy"]),
            ],
        }
        registry_path.write_text(yaml.dump(registry, sort_keys=False))
        server = create_registry_server(registry_path)

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

    def test_tools_without_deps(self, registry_path):
        """Tools without dependencies field are skipped gracefully."""
        registry = {
            "tools": [make_spec("nodep_tool")],
        }
        registry_path.write_text(yaml.dump(registry, sort_keys=False))
        server = create_registry_server(registry_path)

        text = call_tool(server, "install_dependencies", {})
        assert "No dependencies" in text
