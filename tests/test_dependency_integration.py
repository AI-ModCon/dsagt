"""
Integration test for dependency installation during tool registration.

Registers a tool with a real dependency (cowsay), installs it via the
registry server, then executes the tool through the pipeline's ToolRegistry
to verify the package is usable.

This test actually modifies the venv (installs and uninstalls cowsay).

Skip conditions:
  - uv not available on PATH
  - cowsay already installed (test would be meaningless)

Usage:
    pytest test_dependency_integration.py -v
"""

import asyncio
import importlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
import mcp.types as types

from dsagt.registry_server import create_registry_server
from dsagt.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _uv_available() -> bool:
    return shutil.which("uv") is not None


def _cowsay_installed() -> bool:
    try:
        importlib.import_module("cowsay")
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(not _uv_available(), reason="uv not available on PATH"),
    pytest.mark.skipif(_cowsay_installed(), reason="cowsay already installed"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def uninstall_cowsay_after():
    """Ensure cowsay is uninstalled after all tests in this module."""
    yield
    subprocess.run(
        ["uv", "pip", "uninstall", "cowsay", "--python", sys.executable],
        capture_output=True,
        timeout=30,
    )


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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_register_and_run_tool_with_dependency(tmp_path):
    """End-to-end: register a tool with a dependency, install it, run the tool."""
    # 1. Write a test script that imports cowsay
    script = tmp_path / "cowsay_tool.py"
    script.write_text(textwrap.dedent("""\
        import argparse
        import json
        import cowsay

        parser = argparse.ArgumentParser()
        parser.add_argument("message")
        args = parser.parse_args()

        output = cowsay.get_output_string("cow", args.message)
        print(json.dumps({"cow_says": output, "status": "ok"}))
    """))

    # 2. Create a registry server with a fresh registry
    registry_path = tmp_path / "registry.yaml"
    server = create_registry_server(registry_path)

    # 3. Register the tool with dependencies
    spec = {
        "name": "cowsay_tool",
        "description": "Print a cow saying a message",
        "executable": f"python {script}",
        "dependencies": ["cowsay"],
        "parameters": {
            "message": {
                "type": "string",
                "required": True,
                "description": "Message for the cow to say",
            },
        },
    }
    text = call_tool(server, "save_tool_spec", {"spec": spec})

    # Verify the tool was saved and deps were installed
    assert "added" in text
    assert "Successfully installed" in text

    # 4. Verify the spec is in the YAML with dependencies
    registry = yaml.safe_load(registry_path.read_text())
    assert registry["tools"][0]["dependencies"] == ["cowsay"]

    # 5. Execute the tool through ToolRegistry
    tool_registry = ToolRegistry(
        source_registry=str(registry_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    result = tool_registry.call_tool("cowsay_tool", {"message": "hello"})

    assert result["success"] is True, f"Tool failed: {result['error']}"
    output = json.loads(result["output"])
    assert output["status"] == "ok"
    assert "hello" in output["cow_says"]
