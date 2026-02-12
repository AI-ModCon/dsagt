"""Shared MCP server utilities for DSAGT servers."""

import json
import os
import sys

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions


def create_server(name: str) -> Server:
    """Create an MCP server instance."""
    return Server(name)


async def run_stdio(server: Server, name: str, version: str = "0.1.0"):
    """Run an MCP server over stdio transport."""
    # Force unbuffered I/O — required when stdout is a pipe (e.g., Goose).
    # Without this, MCP responses can get stuck in the buffer and cause
    # the transport to time out.
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=name,
                server_version=version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def text_result(data) -> list[types.TextContent]:
    """Wrap a value as a JSON TextContent response."""
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, indent=2)
    return [types.TextContent(type="text", text=text)]
