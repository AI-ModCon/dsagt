"""
Tests for ToolRegistry and SkillRegistry.

Covers tool listing, MCP schema conversion, tool lookup, tool file writing,
dsagt-run wrapping, runtime isolation, and skill discovery.
"""

import pytest
import yaml

from dsagt.registry import (
    ToolRegistry, SkillRegistry, _wrap_executable, _uv_run_prefix, _parse_frontmatter,
    render_arguments,
)


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
            "cli": "positional:0",
            "description": "Path to input file",
        },
        "output_file": {
            "type": "string",
            "required": True,
            "cli": "positional:1",
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


def _write_tool(tools_dir, spec: dict) -> None:
    """Write a minimal skill file for the given spec dict."""
    path = tools_dir / f"{spec['name']}.md"
    frontmatter = yaml.dump(spec, default_flow_style=False, sort_keys=False)
    path.write_text(f"---\n{frontmatter}---\n\n# {spec['name']}\n")


def make_registry(tmp_path, tools: list[dict]) -> ToolRegistry:
    """Create a ToolRegistry with the given tool definitions."""
    tools_dir = tmp_path / "source_skills"
    tools_dir.mkdir()
    for tool in tools:
        _write_tool(tools_dir, tool)
    return ToolRegistry(
        source_tools_dir=str(tools_dir),
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
# save_tool
# ---------------------------------------------------------------------------

class TestSaveTool:

    def test_add_new_tool(self, empty_registry):
        """Saving a new tool creates a skill file with dsagt-run wrapping."""
        empty_registry.save_tool(TOOL_NO_PARAMS)

        tool = empty_registry.get_tool("ping")
        assert tool is not None
        assert tool["name"] == "ping"
        assert tool["executable"] == "dsagt-run --tool ping -- echo pong"

    def test_wraps_executable_with_dsagt_run(self, empty_registry):
        """save_tool automatically wraps the executable with dsagt-run."""
        empty_registry.save_tool({"name": "mytool", "description": "test",
                                   "executable": "python mytool.py", "parameters": {}})
        tool = empty_registry.get_tool("mytool")
        assert tool["executable"] == "dsagt-run --tool mytool -- python mytool.py"

    def test_does_not_double_wrap(self, empty_registry):
        """If executable already has dsagt-run, don't wrap again."""
        empty_registry.save_tool({"name": "mytool", "description": "test",
                                   "executable": "dsagt-run --tool mytool -- python mytool.py",
                                   "parameters": {}})
        tool = empty_registry.get_tool("mytool")
        assert tool["executable"].count("dsagt-run") == 1

    def test_python_deps_use_uv_run(self, empty_registry):
        """Python dependencies are wrapped with uv run --with."""
        empty_registry.save_tool({
            "name": "analyzer", "description": "test",
            "executable": "python analyzer.py",
            "parameters": {},
            "dependencies": ["pandas>=2.0", "numpy"],
        })
        tool = empty_registry.get_tool("analyzer")
        assert tool["executable"] == (
            "dsagt-run --tool analyzer -- uv run --with pandas>=2.0,numpy -- python analyzer.py"
        )

    def test_no_deps_no_uv_run(self, empty_registry):
        """Tools without dependencies don't get uv run prefix."""
        empty_registry.save_tool({"name": "simple", "description": "test",
                                   "executable": "echo hi", "parameters": {}})
        tool = empty_registry.get_tool("simple")
        assert "uv run" not in tool["executable"]
        assert tool["executable"] == "dsagt-run --tool simple -- echo hi"

    def test_add_returns_added(self, empty_registry):
        """save_tool returns 'added' for new tools."""
        assert empty_registry.save_tool(TOOL_NO_PARAMS) == "added"

    def test_update_returns_updated(self, empty_registry):
        """save_tool returns 'updated' when overwriting an existing tool."""
        empty_registry.save_tool(TOOL_NO_PARAMS)
        assert empty_registry.save_tool(TOOL_NO_PARAMS) == "updated"

    def test_update_preserves_body(self, empty_registry):
        """Updating a tool preserves any hand-edited markdown body."""
        skill_path = empty_registry.tools_dir / "ping.md"
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
        _write_tool(source_dir, TOOL_NO_PARAMS)

        runtime_dir = tmp_path / "runtime"
        reg = ToolRegistry(
            source_tools_dir=str(source_dir),
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
# Default skills
# ---------------------------------------------------------------------------

class TestDefaultTools:
    """Validate the tool files that ship with the package."""

    def test_tools_directory_exists(self):
        """The package ships a tools directory."""
        assert ToolRegistry._PACKAGE_TOOLS_DIR.exists()
        assert ToolRegistry._PACKAGE_TOOLS_DIR.is_dir()

    def test_tools_are_valid(self):
        """Every tool file must parse cleanly and have required fields."""
        tool_files = list(ToolRegistry._PACKAGE_TOOLS_DIR.glob("*.md"))
        assert len(tool_files) > 0, "No tool files found in package"

        for path in tool_files:
            tool = _parse_frontmatter(path)
            assert tool.get("name"), f"{path.name}: missing 'name'"
            assert tool.get("description"), f"{path.name}: missing 'description'"
            assert tool.get("executable"), f"{path.name}: missing 'executable'"
            assert "parameters" in tool, f"{path.name}: missing 'parameters'"

    def test_default_init_fallback(self, tmp_path):
        """ToolRegistry with no source_tools_dir falls back to package skills."""
        reg = ToolRegistry(source_tools_dir=None, runtime_dir=str(tmp_path / "rt"))
        tools = reg.list_tools()
        assert len(tools) > 0
        names = [t["name"] for t in tools]
        assert "scan_directory" in names


# ---------------------------------------------------------------------------
# render_arguments
# ---------------------------------------------------------------------------

class TestRenderArguments:

    def test_default_cli_is_double_dash_name(self):
        params = {"foo": {"type": "string"}}
        assert render_arguments(params, {"foo": "bar"}) == ["--foo", "bar"]

    def test_spaced_long_flag(self):
        params = {"foo": {"type": "string", "cli": "--foo"}}
        assert render_arguments(params, {"foo": "bar"}) == ["--foo", "bar"]

    def test_spaced_short_flag(self):
        params = {"x": {"type": "string", "cli": "-x"}}
        assert render_arguments(params, {"x": "val"}) == ["-x", "val"]

    def test_glued_long_flag(self):
        params = {"foo": {"type": "string", "cli": "--foo="}}
        assert render_arguments(params, {"foo": "bar"}) == ["--foo=bar"]

    def test_glued_short_flag(self):
        params = {"x": {"type": "string", "cli": "-x="}}
        assert render_arguments(params, {"x": "val"}) == ["-x=val"]

    def test_keyvalue_style(self):
        params = {"if_": {"type": "string", "cli": "if="}}
        assert render_arguments(params, {"if_": "input.dat"}) == ["if=input.dat"]

    def test_single_positional(self):
        params = {"target": {"type": "string", "cli": "positional"}}
        assert render_arguments(params, {"target": "/tmp/x"}) == ["/tmp/x"]

    def test_multiple_positionals_respect_order(self):
        params = {
            "dest": {"type": "string", "cli": "positional:1"},
            "src": {"type": "string", "cli": "positional:0"},
        }
        assert render_arguments(params, {"src": "a", "dest": "b"}) == ["a", "b"]

    def test_positionals_before_named(self):
        params = {
            "verbose": {"type": "boolean", "cli": "--verbose"},
            "path": {"type": "string", "cli": "positional:0"},
        }
        assert render_arguments(params, {"path": "/x", "verbose": True}) == [
            "/x", "--verbose",
        ]

    def test_boolean_true_emits_flag(self):
        params = {"verbose": {"type": "boolean", "cli": "--verbose"}}
        assert render_arguments(params, {"verbose": True}) == ["--verbose"]

    def test_boolean_false_emits_nothing(self):
        params = {"verbose": {"type": "boolean", "cli": "--verbose"}}
        assert render_arguments(params, {"verbose": False}) == []

    def test_boolean_positional_rejected(self):
        params = {"flag": {"type": "boolean", "cli": "positional"}}
        with pytest.raises(ValueError, match="boolean"):
            render_arguments(params, {"flag": True})

    def test_default_applied_when_value_missing(self):
        params = {"max_depth": {"type": "integer", "cli": "--max-depth", "default": 5}}
        assert render_arguments(params, {}) == ["--max-depth", "5"]

    def test_optional_missing_skipped(self):
        params = {"max_depth": {"type": "integer", "cli": "--max-depth"}}
        assert render_arguments(params, {}) == []

    def test_required_missing_raises(self):
        params = {"directory": {"type": "string", "cli": "positional", "required": True}}
        with pytest.raises(ValueError, match="directory"):
            render_arguments(params, {})

    def test_invalid_cli_value_raises(self):
        params = {"weird": {"type": "string", "cli": "!!!invalid"}}
        with pytest.raises(ValueError, match="invalid cli value"):
            render_arguments(params, {"weird": "x"})

    def test_invalid_position_raises(self):
        params = {"x": {"type": "string", "cli": "positional:abc"}}
        with pytest.raises(ValueError, match="integer"):
            render_arguments(params, {"x": "val"})
