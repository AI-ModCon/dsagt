"""
DSAGT Pipeline MCP Server

Exposes data processing tools to MCP-compatible agents (Goose, Claude, etc.).

Usage:
    dsagt-pipeline-server
    dsagt-pipeline-server --registry ./my_registry.yaml
    dsagt-pipeline-server --runtime-dir ./my_session
"""

import argparse
import asyncio

from dsagt.mcp_utils import create_server, run_stdio, text_result, types
from dsagt.registry import ToolRegistry


def create_pipeline_server(registry: ToolRegistry):
    """Create and configure the MCP server."""
    server = create_server("pipeline")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in registry.list_tools()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        result = registry.call_tool(name, arguments)
        return text_result(result)

    return server


def main():
    parser = argparse.ArgumentParser(description="DSAGT MCP Server")
    parser.add_argument(
        "--registry",
        default=None,
        help="Source registry YAML (default: bundled registry with standard tools)",
    )
    parser.add_argument("--runtime-dir", default="./runtime", help="Runtime directory")
    args = parser.parse_args()

    registry = ToolRegistry(source_registry=args.registry, runtime_dir=args.runtime_dir)
    server = create_pipeline_server(registry)
    asyncio.run(run_stdio(server, "pipeline"))


if __name__ == "__main__":
    main()
