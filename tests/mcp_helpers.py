"""
Shared MCP test helpers.

Provides:
- In-process tool invocation: call_tool_sync(), call_tool_async()
- Subprocess helpers: start_server(), mcp_initialize(), mcp_call_tool(), etc.
"""

import asyncio
import json
import os
import subprocess
import time

import mcp.types as types

# ---------------------------------------------------------------------------
# In-process MCP tool invocation (for unit tests)
# ---------------------------------------------------------------------------


def call_tool_sync(server, name: str, arguments: dict) -> str:
    """Invoke a tool handler on an MCP server and return the response text."""
    params = types.CallToolRequestParams(name=name, arguments=arguments)
    handler = server.get_request_handler("tools/call").handler
    result = asyncio.run(handler(None, params))
    return result.content[0].text


def call_tool_json(server, name: str, arguments: dict) -> dict:
    """Invoke a tool handler and return the parsed JSON response."""
    return json.loads(call_tool_sync(server, name, arguments))


async def call_tool_async(server, name: str, arguments: dict) -> str:
    """Invoke a tool handler inside a running event loop."""
    params = types.CallToolRequestParams(name=name, arguments=arguments)
    handler = server.get_request_handler("tools/call").handler
    result = await handler(None, params)
    return result.content[0].text


# ---------------------------------------------------------------------------
# Subprocess helpers (for integration tests)
# ---------------------------------------------------------------------------


def send_mcp_message(proc, message: dict):
    """Send a JSON-RPC message as newline-delimited JSON (MCP stdio framing)."""
    line = json.dumps(message) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def read_line_timeout(proc, timeout: float) -> str:
    """Read one line from proc.stdout with a timeout using a daemon thread."""
    import threading

    result = [None]

    def reader():
        result[0] = proc.stdout.readline()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError("Timed out reading line from server stdout")

    line = result[0]
    if line == "":
        raise ConnectionError(
            f"Server process exited (rc={proc.poll()}). "
            f"stderr: {proc.stderr.read()}"
        )
    return line


def read_mcp_message(proc, timeout: float = 10.0, expect_id=None) -> dict:
    """Read one JSON-RPC message from stdout (newline-delimited JSON)."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out reading MCP message (expect_id={expect_id})."
            )

        line = read_line_timeout(proc, remaining).strip()
        if not line:
            continue

        msg = json.loads(line)
        if expect_id is not None and msg.get("id") != expect_id:
            continue
        return msg


def mcp_initialize(proc) -> dict:
    """Send MCP initialize handshake and return the server's response."""
    send_mcp_message(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
    )
    response = read_mcp_message(proc, expect_id=1)

    send_mcp_message(
        proc,
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
    )

    return response


def mcp_list_tools(proc) -> dict:
    """Request tools/list and return the response."""
    send_mcp_message(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        },
    )
    return read_mcp_message(proc, expect_id=2)


def mcp_call_tool(
    proc, tool_name: str, arguments: dict, msg_id: int = 3, timeout: float = 30.0
) -> dict:
    """Call an MCP tool and return the JSON-RPC response."""
    send_mcp_message(
        proc,
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        },
    )
    return read_mcp_message(proc, timeout=timeout, expect_id=msg_id)


def start_server(cmd: list[str], env: dict = None, cwd: str = None) -> subprocess.Popen:
    """Start a server subprocess with stdio pipes.

    ``cwd`` sets the working directory — ``dsagt-server`` discovers its project
    config from cwd (see ``observability.find_project_config``), so startup tests
    must run it from the project dir.
    """
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
        cwd=cwd,
    )
