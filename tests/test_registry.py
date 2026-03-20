"""
Tests for ToolRegistry.

Covers tool listing, MCP schema conversion, tool lookup, command execution,
runtime isolation, and provenance logging.
"""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest
import yaml

from dsagt.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOOL_WITH_MIXED_PARAMS = {
    "name": "process",
    "description": "Process a file",
    "executable": "python process.py",
    "parameters": {
        "input_file": {
            "type": "string",
            "required": True,
            "positional": True,
            "description": "Path to input file",
        },
        "output_file": {
            "type": "string",
            "required": True,
            "positional": True,
            "description": "Path to output file",
        },
        "threshold": {
            "type": "number",
            "required": False,
            "default": 0.5,
            "description": "Threshold value",
        },
    },
}

TOOL_NO_PARAMS = {
    "name": "ping",
    "description": "Check availability",
    "executable": "echo pong",
    "parameters": {},
}


def make_registry(tmp_path, tools: list[dict]) -> ToolRegistry:
    """Create a ToolRegistry with the given tool definitions."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.dump({"tools": tools}, sort_keys=False))
    runtime_dir = tmp_path / "runtime"
    return ToolRegistry(
        source_registry=str(registry_path),
        runtime_dir=str(runtime_dir),
    )


@pytest.fixture
def registry(tmp_path):
    """Registry with one tool that has required and optional params."""
    return make_registry(tmp_path, [TOOL_WITH_MIXED_PARAMS])


@pytest.fixture
def empty_registry(tmp_path):
    """Registry with no tools."""
    return make_registry(tmp_path, [])


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:

    def test_schema_structure(self, registry):
        """MCP schema has name, description, and well-formed inputSchema."""
        tools = registry.list_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "process"
        assert tool["description"] == "Process a file"

        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_required_vs_optional(self, registry):
        """Required params appear in 'required', optional ones don't."""
        tool = registry.list_tools()[0]
        required = tool["inputSchema"]["required"]

        assert "input_file" in required
        assert "output_file" in required
        assert "threshold" not in required

    def test_default_values_propagate(self, registry):
        """Default values from the registry appear in the MCP schema."""
        tool = registry.list_tools()[0]
        props = tool["inputSchema"]["properties"]

        assert props["threshold"]["default"] == 0.5
        assert "default" not in props["input_file"]

    def test_empty_registry(self, empty_registry):
        """An empty registry gives an empty tool list."""
        assert empty_registry.list_tools() == []

    def test_multiple_tools(self, tmp_path):
        """Multiple tools are listed in order."""
        reg = make_registry(tmp_path, [TOOL_WITH_MIXED_PARAMS, TOOL_NO_PARAMS])
        tools = reg.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "process"
        assert tools[1]["name"] == "ping"

    def test_no_params_tool(self, tmp_path):
        """Tool with empty parameters gives empty properties and required."""
        reg = make_registry(tmp_path, [TOOL_NO_PARAMS])
        tool = reg.list_tools()[0]

        assert tool["inputSchema"]["properties"] == {}
        assert tool["inputSchema"]["required"] == []


# ---------------------------------------------------------------------------
# get_tool
# ---------------------------------------------------------------------------

class TestGetTool:

    def test_found(self, registry):
        """Returns the raw tool definition for an existing tool."""
        tool = registry.get_tool("process")
        assert tool is not None
        assert tool["name"] == "process"
        assert tool["executable"] == "python process.py"

    def test_not_found(self, registry):
        """Returns None for a nonexistent tool."""
        assert registry.get_tool("nonexistent") is None


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------

class TestCallTool:

    def test_unknown_tool(self, registry):
        """Calling an unknown tool returns an error dict."""
        result = registry.call_tool("nonexistent", {})

        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @patch("dsagt.registry.subprocess.run")
    def test_success(self, mock_run, registry):
        """Successful execution returns stdout and no error."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"status": "ok"}',
            stderr="",
        )

        result = registry.call_tool("process", {
            "input_file": "data.csv",
            "output_file": "out.csv",
        })

        assert result["success"] is True
        assert result["output"] == '{"status": "ok"}'
        assert result["error"] is None

    @patch("dsagt.registry.subprocess.run")
    def test_failure(self, mock_run, registry):
        """Failed execution returns stderr as error."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="File not found",
        )

        result = registry.call_tool("process", {
            "input_file": "missing.csv",
            "output_file": "out.csv",
        })

        assert result["success"] is False
        assert result["error"] == "File not found"

    @patch("dsagt.registry.subprocess.run")
    def test_command_construction_required_params(self, mock_run, registry):
        """Positional params are bare, optional use --flag syntax."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        registry.call_tool("process", {
            "input_file": "in.csv",
            "output_file": "out.csv",
            "threshold": 0.8,
        })

        cmd = mock_run.call_args[0][0]
        cmd_str = cmd[2] if isinstance(cmd, list) else cmd
        assert "in.csv" in cmd_str
        assert "out.csv" in cmd_str
        assert "--threshold" in cmd_str

    @patch("dsagt.registry.subprocess.run")
    def test_omitted_optional_params_excluded(self, mock_run, registry):
        """Optional params omitted by caller are excluded from command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        registry.call_tool("process", {
            "input_file": "in.csv",
            "output_file": "out.csv",
            # threshold omitted — should NOT appear in command
        })

        cmd = mock_run.call_args[0][0]
        cmd_str = cmd[2] if isinstance(cmd, list) else cmd
        assert "--threshold" not in cmd_str

    @patch("dsagt.registry.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300))
    def test_timeout(self, mock_run, registry):
        """TimeoutExpired propagates (not caught by call_tool)."""
        with pytest.raises(subprocess.TimeoutExpired):
            registry.call_tool("process", {
                "input_file": "in.csv",
                "output_file": "out.csv",
            })


# ---------------------------------------------------------------------------
# Runtime isolation
# ---------------------------------------------------------------------------

class TestRuntimeIsolation:

    def test_source_unchanged_after_init(self, tmp_path):
        """Source registry is not modified; runtime copy is separate."""
        source_path = tmp_path / "registry.yaml"
        source_content = yaml.dump({"tools": [TOOL_NO_PARAMS]}, sort_keys=False)
        source_path.write_text(source_content)

        runtime_dir = tmp_path / "runtime"
        reg = ToolRegistry(
            source_registry=str(source_path),
            runtime_dir=str(runtime_dir),
        )

        # Modify runtime copy
        runtime_data = reg._load_registry()
        runtime_data["tools"].append({
            "name": "new_tool",
            "description": "added",
            "executable": "echo",
            "parameters": {},
        })
        reg._save_registry(runtime_data)

        # Source should be unchanged
        assert source_path.read_text() == source_content

        # Runtime should have the new tool
        assert len(reg._load_registry()["tools"]) == 2


# ---------------------------------------------------------------------------
# Provenance logging
# ---------------------------------------------------------------------------

class TestProvenance:

    @patch("dsagt.registry.subprocess.run")
    def test_call_tool_logs_entry(self, mock_run, registry):
        """Calling a tool writes a line to the provenance log."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        registry.call_tool("process", {"input_file": "in.csv", "output_file": "out.csv"})

        log_content = registry.provenance_log.read_text()
        assert "process" in log_content
        assert "in.csv" in log_content

    def test_session_header(self, registry):
        """Provenance log starts with a session header."""
        log_content = registry.provenance_log.read_text()
        assert "# Session started:" in log_content


# ---------------------------------------------------------------------------
# Default Registry
# ---------------------------------------------------------------------------

class TestDefaultRegistry:
    """Validate the bundled registry.yaml that ships with the package."""

    def test_default_registry_is_valid_yaml(self):
        """The shipped registry.yaml must parse without errors."""
        registry_path = ToolRegistry._DEFAULT_REGISTRY
        assert registry_path.exists(), f"Default registry not found at {registry_path}"

        with open(registry_path) as f:
            data = yaml.safe_load(f)

        assert "tools" in data
        assert len(data["tools"]) > 0

    def test_default_registry_round_trips(self):
        """The shipped registry.yaml must survive a dump/reload cycle.

        This is critical because save_tool_spec reads and re-dumps the
        registry. If the YAML contains syntax that doesn't round-trip
        cleanly (e.g., unquoted colons in values), writes will corrupt it.
        """
        with open(ToolRegistry._DEFAULT_REGISTRY) as f:
            data = yaml.safe_load(f)

        dumped = yaml.dump(data, default_flow_style=False, sort_keys=False)
        reloaded = yaml.safe_load(dumped)

        assert len(reloaded["tools"]) == len(data["tools"])
        original_names = [t["name"] for t in data["tools"]]
        reloaded_names = [t["name"] for t in reloaded["tools"]]
        assert original_names == reloaded_names

    def test_default_registry_tools_have_required_fields(self):
        """Every tool in the default registry must have name, description,
        executable, and parameters."""
        with open(ToolRegistry._DEFAULT_REGISTRY) as f:
            data = yaml.safe_load(f)

        for tool in data["tools"]:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert "executable" in tool, f"Tool {tool['name']} missing 'executable'"
            assert "parameters" in tool, f"Tool {tool['name']} missing 'parameters'"

    def test_default_registry_init_fallback(self, tmp_path):
        """ToolRegistry with no source_registry falls back to default."""
        reg = ToolRegistry(source_registry=None, runtime_dir=str(tmp_path / "rt"))
        tools = reg.list_tools()
        assert len(tools) > 0
        names = [t["name"] for t in tools]
        assert "scan_directory" in names
