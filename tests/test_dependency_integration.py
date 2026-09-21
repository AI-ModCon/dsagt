"""
Integration test for a registered code's declared dependencies.

Registers a code that needs a package the venv does not have (cowsay), then
runs the code by its stored line.  The ``uv run --with`` prefix in that line
is what supplies the dependency, so the run succeeds while the venv running
the server stays as it was.

Skip conditions:
  - uv not available on PATH
  - cowsay already installed (the test would prove nothing)

Usage:
    pytest test_dependency_integration.py -v
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest


from dsagt.mcp.registry_tools import create_registry_server
from dsagt.registry import CodeRegistry

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
    pytest.mark.integration,
    pytest.mark.skipif(not _uv_available(), reason="uv not available on PATH"),
    pytest.mark.skipif(_cowsay_installed(), reason="cowsay already installed"),
]


from mcp_helpers import call_tool_sync as call_tool

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_a_declared_dependency_reaches_the_run(tmp_path):
    """End to end: register a code with a dependency, run its stored line."""
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

    project = tmp_path / "runtime"
    registry = CodeRegistry(runtime_dir=str(project))
    server = create_registry_server(registry)

    spec = {
        "name": "cowsay-tool",
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
    text = call_tool(server, "save_code_spec", {"spec": spec})
    assert "added" in text

    # The dependency is in the stored line, which is what the agent runs.
    stored = registry.get_code("cowsay-tool")
    assert stored["dependencies"] == ["cowsay"]
    assert "dsagt-run --code cowsay-tool --" in stored["executable"]
    assert "uv run --with cowsay --" in stored["executable"]

    (project / ".dsagt").mkdir(parents=True, exist_ok=True)
    (project / ".dsagt" / "config.yaml").write_text("project: dep-test\n")
    env = {**os.environ, "DSAGT_PROJECT_DIR": str(project)}
    result = subprocess.run(
        f"{stored['executable']} --message hello",
        shell=True,
        capture_output=True,
        text=True,
        cwd=project,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"run failed: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert "hello" in output["cow_says"]

    # The run supplied cowsay to itself; the interpreter the server runs on
    # is the one it was.
    probe = subprocess.run(
        [sys.executable, "-c", "import cowsay"], capture_output=True, text=True
    )
    assert probe.returncode != 0, "the venv gained cowsay; a run must not install"
