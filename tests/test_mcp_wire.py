"""Wire-level tests for the dispatch shell over a real stdio transport.

Every other server test drives handlers in-process (``mcp_helpers.call_tool_sync``),
which never touches :class:`mcp.server.lowlevel.Server` — so the attachment
between :func:`~dsagt.mcp.server.build_dispatch_server` and the SDK's transport
had no coverage at all.  That is precisely the seam the MCP SDK breaks across
major versions, and the seam a broken ``dsagt-server`` shows up on as "the agent
sees zero dsagt tools".

These spawn ``wire_server.py`` (no project config, no KB) and speak JSON-RPC to
it, so they stay fast enough for the default suite.
"""

import json
import sys
from pathlib import Path

import pytest

from mcp_helpers import (
    mcp_call_tool,
    mcp_initialize,
    mcp_list_tools,
    start_server,
)

_WIRE_SERVER = str(Path(__file__).parent / "wire_server.py")


@pytest.fixture
def wire_proc():
    """A live ``wire_server.py`` subprocess, handshake already completed."""
    proc = start_server([sys.executable, _WIRE_SERVER])
    try:
        response = mcp_initialize(proc)
        assert "result" in response, response
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _tool_text(response: dict) -> str:
    """Pull the single TextContent payload out of a tools/call response."""
    assert "error" not in response, response
    return response["result"]["content"][0]["text"]


class TestWireProtocol:

    def test_handshake_reports_tools_capability(self):
        """A server with tools must advertise them, or the agent shows none."""
        proc = start_server([sys.executable, _WIRE_SERVER])
        try:
            result = mcp_initialize(proc)["result"]
            assert "tools" in result["capabilities"]
            assert result["serverInfo"]["name"] == "wire-test"
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_list_tools_returns_the_registered_tools(self, wire_proc):
        tools = mcp_list_tools(wire_proc)["result"]["tools"]
        assert {t["name"] for t in tools} == {"echo", "spawn", "boom"}
        echo = next(t for t in tools if t["name"] == "echo")
        assert echo["inputSchema"]["properties"]["text"]["type"] == "string"

    def test_call_tool_round_trips_a_json_payload(self, wire_proc):
        response = mcp_call_tool(wire_proc, "echo", {"text": "hi"})
        assert json.loads(_tool_text(response)) == {"echoed": {"text": "hi"}}

    def test_handler_error_comes_back_as_an_error_payload(self, wire_proc):
        """The dispatch shell converts handler failures into a result payload,
        not a transport error — the agent must stay connected."""
        response = mcp_call_tool(wire_proc, "boom", {})
        assert json.loads(_tool_text(response)) == {
            "status": "error",
            "error": "handler exploded",
        }

    def test_bad_arguments_are_rejected_against_the_tool_schema(self, wire_proc):
        """Argument validation is dsagt's own (the SDK's does not survive its
        major versions), so it must hold on the wire and report the house error
        shape."""
        response = mcp_call_tool(wire_proc, "echo", {})
        payload = json.loads(_tool_text(response))
        assert payload["status"] == "error"
        assert payload["error"].startswith("Input validation error:")

    def test_spawning_a_child_process_leaves_the_wire_clean(self, wire_proc):
        """``install_dependencies`` and the KB ingest spawn child processes
        from inside the serving process, whose fd 0/1 the transport owns.  The child must run, and the JSON-RPC
        stream must survive it, verified by a further call on the same
        connection."""
        response = mcp_call_tool(wire_proc, "spawn", {})
        assert json.loads(_tool_text(response)) == {
            "returncode": 0,
            "stdout": "child stdout",
        }

        followup = mcp_call_tool(wire_proc, "echo", {"text": "still here"}, msg_id=4)
        assert json.loads(_tool_text(followup)) == {"echoed": {"text": "still here"}}
