"""
DSAGT Registry Builder MCP Server

Provides tools for building a tool registry by reading documentation,
fetching web resources, and running commands to extract tool specifications.

The registry is saved in YAML format compatible with the DSAGT pipeline server.

Usage:
    dsagt-registry-builder
    dsagt-registry-builder --registry ./my_registry.yaml
"""

import argparse
import asyncio
import subprocess
import httpx
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import yaml

app = Server("tool-registry-server")

# Global registry path (set in main())
REGISTRY_PATH = Path("tool_registry.yaml")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read contents of a text file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="http_request",
            description="Make an HTTP request to fetch documentation or API specs",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    "headers": {"type": "object", "description": "Optional headers"},
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="run_command",
            description="Execute a command to get help/usage information",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments (e.g., ['--help'])",
                        "default": []
                    },
                    "timeout": {"type": "number", "default": 10}
                },
                "required": ["command"]
            }
        ),
        Tool(
            name="save_tool_spec",
            description="Save a tool specification to the registry",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": "Tool specification matching DSAGT registry schema",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Unique tool identifier"
                            },
                            "description": {
                                "type": "string",
                                "description": "What the tool does"
                            },
                            "executable": {
                                "type": "string",
                                "description": "Command to execute (e.g., 'python script.py')"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameter definitions keyed by parameter name",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Parameter type (string, integer, number, boolean, array, object)"
                                        },
                                        "required": {
                                            "type": "boolean",
                                            "description": "Whether this parameter is required"
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Parameter description"
                                        },
                                        "default": {
                                            "description": "Default value if not provided"
                                        }
                                    },
                                    "required": ["type", "description"]
                                }
                            }
                        },
                        "required": ["name", "description", "executable", "parameters"]
                    }
                },
                "required": ["spec"]
            }
        ),
        Tool(
            name="get_registry",
            description="Get all tools from the registry",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="search_registry",
            description="Search for tools in the registry by name or category",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Filter by category"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "read_file":
        path = Path(arguments["path"])
        try:
            content = path.read_text()
            return [TextContent(type="text", text=content)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading file: {str(e)}")]

    elif name == "http_request":
        url = arguments["url"]
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {})

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=30.0
                )
                return [TextContent(
                    type="text",
                    text=f"Status: {response.status_code}\n\n{response.text}"
                )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error making request: {str(e)}")]

    elif name == "run_command":
        command = arguments["command"]
        args = arguments.get("args", [])
        timeout = arguments.get("timeout", 10)

        try:
            result = subprocess.run(
                [command] + args,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            output += f"\nReturn code: {result.returncode}"

            return [TextContent(type="text", text=output)]

        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text=f"Command timed out after {timeout} seconds")]
        except FileNotFoundError:
            return [TextContent(type="text", text=f"Command '{command}' not found")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error executing command: {str(e)}")]

    elif name == "save_tool_spec":
        spec = arguments["spec"]

        try:
            # Load existing registry or create new one
            if REGISTRY_PATH.exists():
                with open(REGISTRY_PATH) as f:
                    registry = yaml.safe_load(f) or {}
            else:
                registry = {}

            # Ensure tools list exists
            if "tools" not in registry:
                registry["tools"] = []

            # Check if tool already exists and update, or add new
            existing_idx = None
            for idx, tool in enumerate(registry["tools"]):
                if tool["name"] == spec["name"]:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                registry["tools"][existing_idx] = spec
                action = "updated"
            else:
                registry["tools"].append(spec)
                action = "added"

            # Save registry in YAML format
            with open(REGISTRY_PATH, "w") as f:
                yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

            return [TextContent(
                type="text",
                text=f"Tool '{spec['name']}' {action} successfully. Registry now contains {len(registry['tools'])} tools."
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error saving tool spec: {str(e)}")]

    elif name == "get_registry":
        try:
            if not REGISTRY_PATH.exists():
                return [TextContent(type="text", text="Registry is empty. No tools registered yet.")]

            with open(REGISTRY_PATH) as f:
                registry = yaml.safe_load(f)

            return [TextContent(
                type="text",
                text=yaml.dump(registry, default_flow_style=False, sort_keys=False)
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading registry: {str(e)}")]

    elif name == "search_registry":
        query = arguments.get("query", "").lower()
        category = arguments.get("category", "").lower()

        try:
            if not REGISTRY_PATH.exists():
                return [TextContent(type="text", text="Registry is empty.")]

            with open(REGISTRY_PATH) as f:
                registry = yaml.safe_load(f)

            results = []

            for tool in registry.get("tools", []):
                matches = True
                if query and query not in tool["name"].lower() and query not in tool["description"].lower():
                    matches = False
                if category and tool.get("category", "").lower() != category:
                    matches = False

                if matches:
                    results.append(tool)

            if results:
                return [TextContent(
                    type="text",
                    text=f"Found {len(results)} tool(s):\n\n" + yaml.dump(results, default_flow_style=False, sort_keys=False)
                )]
            else:
                return [TextContent(type="text", text="No tools found matching the criteria.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching registry: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")

def main():
    """Entry point for the registry builder server."""
    global REGISTRY_PATH

    parser = argparse.ArgumentParser(description="DSAGT Registry Builder MCP Server")
    parser.add_argument(
        "--registry",
        default="tool_registry.yaml",
        help="Path to the registry YAML file to create/update (default: tool_registry.yaml)"
    )
    args = parser.parse_args()

    # Set the global registry path
    REGISTRY_PATH = Path(args.registry)

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run())

if __name__ == "__main__":
    main()
