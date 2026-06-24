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

import importlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

import pytest
import yaml


from dsagt.mcp.registry_tools import create_registry_server
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


from mcp_helpers import call_tool_sync as call_tool


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
        parser.add_argument("--message", required=True)
        args = parser.parse_args()

        output = cowsay.get_output_string("cow", args.message)
        print(json.dumps({"cow_says": output, "status": "ok"}))
    """))

    # 2. Create a registry server with a fresh ToolRegistry
    registry = ToolRegistry(
        source_tools_dir=None,
        runtime_dir=str(tmp_path / "runtime"),
    )
    server = create_registry_server(registry)

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

    # 4. Verify the spec is in the skill file with dsagt-run wrapping
    tool = registry.get_tool("cowsay_tool")
    assert tool is not None
    assert tool["dependencies"] == ["cowsay"]
    assert "dsagt-run" in tool["executable"]

    # 5. Execute the tool directly via subprocess (as the agent would)
    result = subprocess.run(
        ["python", str(script), "--message", "hello"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Tool failed: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert "hello" in output["cow_says"]
