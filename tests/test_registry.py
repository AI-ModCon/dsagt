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


def _write_skill(skills_dir, spec: dict) -> None:
    """Write a minimal skill file for the given spec dict."""
    path = skills_dir / f"{spec['name']}.md"
    frontmatter = yaml.dump(spec, default_flow_style=False, sort_keys=False)
    path.write_text(f"---\n{frontmatter}---\n\n# {spec['name']}\n")


def make_registry(tmp_path, tools: list[dict]) -> ToolRegistry:
    """Create a ToolRegistry with the given tool definitions."""
    skills_dir = tmp_path / "source_skills"
    skills_dir.mkdir()
    for tool in tools:
        _write_skill(skills_dir, tool)
    return ToolRegistry(
        source_skills_dir=str(skills_dir),
        runtime_dir=str(tmp_path / "runtime"),
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
        """Default values from the skill file appear in the MCP schema."""
        tool = registry.list_tools()[0]
        props = tool["inputSchema"]["properties"]

        assert props["threshold"]["default"] == 0.5
        assert "default" not in props["input_file"]

    def test_empty_registry(self, empty_registry):
        """An empty skills directory gives an empty tool list."""
        assert empty_registry.list_tools() == []

    def test_multiple_tools(self, tmp_path):
        """Multiple tools are listed in alphabetical filename order."""
        reg = make_registry(tmp_path, [TOOL_WITH_MIXED_PARAMS, TOOL_NO_PARAMS])
        tools = reg.list_tools()

        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"process", "ping"}

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
# save_tool
# ---------------------------------------------------------------------------

class TestSaveTool:

    def test_add_new_tool(self, empty_registry):
        """Saving a new tool creates a skill file."""
        empty_registry.save_tool(TOOL_NO_PARAMS)

        tool = empty_registry.get_tool("ping")
        assert tool is not None
        assert tool["name"] == "ping"
        assert tool["executable"] == "echo pong"

    def test_add_returns_added(self, empty_registry):
        """save_tool returns 'added' for new tools."""
        assert empty_registry.save_tool(TOOL_NO_PARAMS) == "added"

    def test_update_returns_updated(self, empty_registry):
        """save_tool returns 'updated' when overwriting an existing tool."""
        empty_registry.save_tool(TOOL_NO_PARAMS)
        assert empty_registry.save_tool(TOOL_NO_PARAMS) == "updated"

    def test_update_preserves_body(self, empty_registry):
        """Updating a tool preserves any hand-edited markdown body."""
        skill_path = empty_registry.skills_dir / "ping.md"
        spec = TOOL_NO_PARAMS
        fm = __import__("yaml").dump(spec, default_flow_style=False, sort_keys=False)
        skill_path.write_text(f"---\n{fm}---\n\n# Custom docs written by hand.\n")

        updated = {**spec, "description": "Updated description"}
        empty_registry.save_tool(updated)

        content = skill_path.read_text()
        assert "Custom docs written by hand." in content

    def test_update_overwrites_frontmatter(self, empty_registry):
        """Updating a tool writes the new spec into the frontmatter."""
        empty_registry.save_tool(TOOL_NO_PARAMS)
        updated = {**TOOL_NO_PARAMS, "description": "New description"}
        empty_registry.save_tool(updated)

        tool = empty_registry.get_tool("ping")
        assert tool["description"] == "New description"


# ---------------------------------------------------------------------------
# Runtime isolation
# ---------------------------------------------------------------------------

class TestRuntimeIsolation:

    def test_source_unchanged_after_init(self, tmp_path):
        """Source skills directory is not modified; runtime copy is separate."""
        source_dir = tmp_path / "source_skills"
        source_dir.mkdir()
        _write_skill(source_dir, TOOL_NO_PARAMS)

        runtime_dir = tmp_path / "runtime"
        reg = ToolRegistry(
            source_skills_dir=str(source_dir),
            runtime_dir=str(runtime_dir),
        )

        # Add a tool to runtime
        reg.save_tool(TOOL_WITH_MIXED_PARAMS)

        # Source should be unchanged
        source_files = list(source_dir.glob("*.md"))
        assert len(source_files) == 1
        assert source_files[0].stem == "ping"

        # Runtime should have both tools
        assert len(reg.list_tools()) == 2


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
# Default skills
# ---------------------------------------------------------------------------

class TestDefaultSkills:
    """Validate the skill files that ship with the package."""

    def test_skills_directory_exists(self):
        """The package ships a skills directory."""
        assert ToolRegistry._PACKAGE_SKILLS_DIR.exists()
        assert ToolRegistry._PACKAGE_SKILLS_DIR.is_dir()

    def test_skills_are_valid(self):
        """Every skill file must parse cleanly and have required fields."""
        skill_files = list(ToolRegistry._PACKAGE_SKILLS_DIR.glob("*.md"))
        assert len(skill_files) > 0, "No skill files found in package"

        for path in skill_files:
            tool = ToolRegistry._parse_skill_file(path)
            assert tool.get("name"), f"{path.name}: missing 'name'"
            assert tool.get("description"), f"{path.name}: missing 'description'"
            assert tool.get("executable"), f"{path.name}: missing 'executable'"
            assert "parameters" in tool, f"{path.name}: missing 'parameters'"

    def test_default_init_fallback(self, tmp_path):
        """ToolRegistry with no source_skills_dir falls back to package skills."""
        reg = ToolRegistry(source_skills_dir=None, runtime_dir=str(tmp_path / "rt"))
        tools = reg.list_tools()
        assert len(tools) > 0
        names = [t["name"] for t in tools]
        assert "scan_directory" in names
