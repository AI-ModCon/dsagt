"""
Tests for registry builder server schema compatibility.

Verifies that the registry builder server writes YAML in the correct format
that can be consumed by the pipeline server's ToolRegistry.
"""

import pytest
import yaml
from pathlib import Path

from dsagt import ToolRegistry


@pytest.fixture
def example_tool_spec():
    """Example tool specification in registry builder format."""
    return {
        "name": "example_tool",
        "description": "An example tool for testing",
        "executable": "python example.py",
        "parameters": {
            "input_file": {
                "type": "string",
                "required": True,
                "description": "Path to input file"
            },
            "output_file": {
                "type": "string",
                "required": True,
                "description": "Path to output file"
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "Threshold value"
            }
        }
    }


@pytest.fixture
def test_registry_file(tmp_path, example_tool_spec):
    """Create a temporary registry file with example tool."""
    registry_path = tmp_path / "test_registry.yaml"
    registry = {"tools": [example_tool_spec]}

    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    return registry_path


@pytest.fixture
def tool_registry(test_registry_file, tmp_path):
    """Create a ToolRegistry instance with test data."""
    runtime_dir = tmp_path / "runtime"
    return ToolRegistry(
        source_registry=str(test_registry_file),
        runtime_dir=str(runtime_dir)
    )


def test_registry_yaml_format(test_registry_file):
    """Test that registry is written in valid YAML format."""
    with open(test_registry_file) as f:
        registry = yaml.safe_load(f)

    assert registry is not None
    assert "tools" in registry
    assert isinstance(registry["tools"], list)


def test_registry_schema_structure(test_registry_file):
    """Test that registry has correct schema structure."""
    with open(test_registry_file) as f:
        registry = yaml.safe_load(f)

    tools = registry["tools"]
    assert len(tools) > 0

    tool = tools[0]
    # Required fields
    assert "name" in tool
    assert "description" in tool
    assert "executable" in tool
    assert "parameters" in tool

    # Parameters should be a dict, not a list
    assert isinstance(tool["parameters"], dict)


def test_parameter_schema(example_tool_spec):
    """Test that parameter definitions have correct structure."""
    params = example_tool_spec["parameters"]

    for param_name, param_def in params.items():
        assert "type" in param_def
        assert "description" in param_def
        assert param_def["type"] in ["string", "integer", "number", "boolean", "array", "object"]


def test_tool_registry_can_load(tool_registry):
    """Test that ToolRegistry can load the registry file."""
    tools = tool_registry.list_tools()

    assert len(tools) == 1
    assert tools[0]["name"] == "example_tool"


def test_tool_registry_schema_conversion(tool_registry):
    """Test that ToolRegistry correctly converts to MCP schema."""
    tools = tool_registry.list_tools()
    tool = tools[0]

    # Check MCP schema structure
    assert "name" in tool
    assert "description" in tool
    assert "inputSchema" in tool

    input_schema = tool["inputSchema"]
    assert input_schema["type"] == "object"
    assert "properties" in input_schema
    assert "required" in input_schema


def test_required_parameters_conversion(tool_registry):
    """Test that required parameters are correctly identified."""
    tools = tool_registry.list_tools()
    tool = tools[0]

    required = tool["inputSchema"]["required"]
    assert "input_file" in required
    assert "output_file" in required
    assert "threshold" not in required  # Optional parameter


def test_parameter_properties(tool_registry):
    """Test that parameter properties are correctly converted."""
    tools = tool_registry.list_tools()
    tool = tools[0]

    properties = tool["inputSchema"]["properties"]

    # Check required parameter
    assert "input_file" in properties
    assert properties["input_file"]["type"] == "string"
    assert properties["input_file"]["description"] == "Path to input file"

    # Check optional parameter with default
    assert "threshold" in properties
    assert properties["threshold"]["type"] == "number"
    assert properties["threshold"]["default"] == 0.5


def test_multiple_tools(tmp_path):
    """Test registry with multiple tools."""
    registry_path = tmp_path / "multi_registry.yaml"
    registry = {
        "tools": [
            {
                "name": "tool1",
                "description": "First tool",
                "executable": "python tool1.py",
                "parameters": {
                    "arg1": {"type": "string", "required": True, "description": "Argument 1"}
                }
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "executable": "python tool2.py",
                "parameters": {
                    "arg2": {"type": "integer", "required": False, "description": "Argument 2", "default": 42}
                }
            }
        ]
    }

    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    runtime_dir = tmp_path / "runtime"
    tool_registry = ToolRegistry(
        source_registry=str(registry_path),
        runtime_dir=str(runtime_dir)
    )

    tools = tool_registry.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "tool1"
    assert tools[1]["name"] == "tool2"


def test_empty_parameters(tmp_path):
    """Test tool with no parameters."""
    registry_path = tmp_path / "no_params_registry.yaml"
    registry = {
        "tools": [
            {
                "name": "simple_tool",
                "description": "Tool with no parameters",
                "executable": "echo hello",
                "parameters": {}
            }
        ]
    }

    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    runtime_dir = tmp_path / "runtime"
    tool_registry = ToolRegistry(
        source_registry=str(registry_path),
        runtime_dir=str(runtime_dir)
    )

    tools = tool_registry.list_tools()
    assert len(tools) == 1
    assert len(tools[0]["inputSchema"]["properties"]) == 0
    assert len(tools[0]["inputSchema"]["required"]) == 0


def test_yaml_preserves_order(test_registry_file):
    """Test that YAML maintains insertion order of parameters."""
    with open(test_registry_file) as f:
        content = f.read()

    # Check that input_file appears before output_file in the file
    input_file_pos = content.find("input_file")
    output_file_pos = content.find("output_file")
    threshold_pos = content.find("threshold")

    assert input_file_pos < output_file_pos < threshold_pos
