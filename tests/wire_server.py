"""Minimal ``dsagt`` MCP server for wire-level tests — not collected by pytest.

Spawned as a subprocess by ``test_mcp_wire.py`` to exercise the one layer the
in-process tests skip: :func:`dsagt.mcp.server.build_dispatch_server` attached to
a real stdio transport via :func:`dsagt.mcp.server._run_stdio`.

It deliberately does *not* go through ``main()`` — no project config, no
KnowledgeBase, no tracing — so the transport is tested on its own, in under a
second, without a model download.
"""

import asyncio
import subprocess
import sys

import mcp.types as types

from dsagt.mcp.server import _run_stdio, build_dispatch_server

TOOLS = [
    types.Tool(
        name="echo",
        description="Echo the arguments back.",
        inputSchema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ),
    types.Tool(
        name="spawn",
        description="Run a chatty child process, as the KB ingest does.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="boom",
        description="Always raises.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


async def _echo(arguments: dict) -> dict:
    return {"echoed": arguments}


async def _boom(arguments: dict) -> dict:
    raise ValueError("handler exploded")


async def _spawn(arguments: dict) -> dict:
    """Mirror ``provenance.run_and_record``: a captured child that writes to both
    streams.  The transport owns fd 0/1 of this process, so spawning at all —
    and keeping the JSON-RPC stream clean afterward — is the thing under test."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; print('child stdout'); print('child stderr', file=sys.stderr)",
        ],
        capture_output=True,
        text=True,
    )
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip()}


def main() -> None:
    server = build_dispatch_server(
        "wire-test", TOOLS, {"echo": _echo, "spawn": _spawn, "boom": _boom}
    )
    asyncio.run(_run_stdio(server, "wire-test"))


if __name__ == "__main__":
    main()
