#!/usr/bin/env python
"""
DSAGT MCP Server

Exposes data processing tools to MCP-compatible agents (Goose, Claude, etc.).

Usage:
    python mcp_server.py
    python mcp_server.py --registry ./my_registry.yaml
    python mcp_server.py --runtime-dir ./my_session
"""

import argparse
import asyncio
import json

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from registry import ToolRegistry, DEFAULT_REGISTRY


def create_server(registry: ToolRegistry) -> Server:
    """Create and configure the MCP server."""
    server = Server("dsagt")
    
    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"]
            )
            for t in registry.list_tools()
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "register_tool":
            result = registry.register_tool(
                name=arguments["name"],
                description=arguments["description"],
                executable=arguments["executable"],
                parameters=arguments.get("parameters", {}),
            )
        else:
            result = registry.call_tool(name, arguments)
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    return server


async def run_server(registry_path: str, runtime_dir: str):
    """Run the MCP server."""
    registry = ToolRegistry(source_registry=registry_path, runtime_dir=runtime_dir)
    server = create_server(registry)
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="dsagt",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="DSAGT MCP Server")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="Source registry YAML")
    parser.add_argument("--runtime-dir", default="./runtime", help="Runtime directory")
    args = parser.parse_args()
    
    asyncio.run(run_server(args.registry, args.runtime_dir))


if __name__ == "__main__":
    main()
