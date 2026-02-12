"""
DSAGT Registry MCP Server

Provides tools for building a tool registry by reading documentation,
fetching web resources, and running commands to extract tool specifications.

The registry is saved in YAML format compatible with the DSAGT pipeline server.

Usage:
    dsagt-registry-server
    dsagt-registry-server --registry ./my_registry.yaml
"""

import argparse
import asyncio
import subprocess
from pathlib import Path

import httpx
import yaml

from dsagt.mcp_utils import create_server, run_stdio, text_result, types


def create_registry_server(registry_path: Path):
    """Create and configure the MCP server."""
    server = create_server("registry")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="read_file",
                description="Read contents of a text file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="http_request",
                description="Make an HTTP request to fetch documentation or API specs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to request"},
                        "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                        "headers": {"type": "object", "description": "Optional headers"},
                    },
                    "required": ["url"],
                },
            ),
            types.Tool(
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
                            "default": [],
                        },
                        "timeout": {"type": "number", "default": 10},
                    },
                    "required": ["command"],
                },
            ),
            types.Tool(
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
                                    "description": "Unique tool identifier",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What the tool does",
                                },
                                "executable": {
                                    "type": "string",
                                    "description": "Command to execute (e.g., 'python script.py')",
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Parameter definitions keyed by parameter name",
                                    "additionalProperties": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "description": "Parameter type (string, integer, number, boolean, array, object)",
                                            },
                                            "required": {
                                                "type": "boolean",
                                                "description": "Whether this parameter is required",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Parameter description",
                                            },
                                            "default": {
                                                "description": "Default value if not provided",
                                            },
                                        },
                                        "required": ["type", "description"],
                                    },
                                },
                            },
                            "required": ["name", "description", "executable", "parameters"],
                        },
                    },
                    "required": ["spec"],
                },
            ),
            types.Tool(
                name="get_registry",
                description="Get all tools from the registry",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="search_registry",
                description="Search for tools in the registry by name or category",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {"type": "string", "description": "Filter by category"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "read_file":
            path = Path(arguments["path"])
            try:
                content = path.read_text()
                return text_result(content)
            except Exception as e:
                return text_result(f"Error reading file: {e}")

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
                        timeout=30.0,
                    )
                    return text_result(f"Status: {response.status_code}\n\n{response.text}")
            except Exception as e:
                return text_result(f"Error making request: {e}")

        elif name == "run_command":
            command = arguments["command"]
            args = arguments.get("args", [])
            timeout = arguments.get("timeout", 10)

            try:
                result = subprocess.run(
                    [command] + args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                output = ""
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n"
                output += f"\nReturn code: {result.returncode}"

                return text_result(output)

            except subprocess.TimeoutExpired:
                return text_result(f"Command timed out after {timeout} seconds")
            except FileNotFoundError:
                return text_result(f"Command '{command}' not found")
            except Exception as e:
                return text_result(f"Error executing command: {e}")

        elif name == "save_tool_spec":
            spec = arguments["spec"]

            try:
                if registry_path.exists():
                    with open(registry_path) as f:
                        registry = yaml.safe_load(f) or {}
                else:
                    registry = {}

                if "tools" not in registry:
                    registry["tools"] = []

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

                with open(registry_path, "w") as f:
                    yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

                return text_result(
                    f"Tool '{spec['name']}' {action} successfully. "
                    f"Registry now contains {len(registry['tools'])} tools."
                )
            except Exception as e:
                return text_result(f"Error saving tool spec: {e}")

        elif name == "get_registry":
            try:
                if not registry_path.exists():
                    return text_result("Registry is empty. No tools registered yet.")

                with open(registry_path) as f:
                    registry = yaml.safe_load(f)

                return text_result(
                    yaml.dump(registry, default_flow_style=False, sort_keys=False)
                )
            except Exception as e:
                return text_result(f"Error reading registry: {e}")

        elif name == "search_registry":
            query = arguments.get("query", "").lower()
            category = arguments.get("category", "").lower()

            try:
                if not registry_path.exists():
                    return text_result("Registry is empty.")

                with open(registry_path) as f:
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
                    return text_result(
                        f"Found {len(results)} tool(s):\n\n"
                        + yaml.dump(results, default_flow_style=False, sort_keys=False)
                    )
                else:
                    return text_result("No tools found matching the criteria.")
            except Exception as e:
                return text_result(f"Error searching registry: {e}")

        raise ValueError(f"Unknown tool: {name}")

    return server


def main():
    """Entry point for the registry builder server."""
    parser = argparse.ArgumentParser(description="DSAGT Registry Builder MCP Server")
    parser.add_argument(
        "--registry",
        default="./runtime/registry.yaml",
        help="Path to the registry YAML file to create/update (default: ./runtime/registry.yaml)",
    )
    parser.add_argument(
        "--runtime-dir",
        default="./runtime",
        help="Runtime directory (default: ./runtime)",
    )
    args = parser.parse_args()

    # Ensure registry parent directory exists
    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    server = create_registry_server(registry_path)
    asyncio.run(run_stdio(server, "registry"))


if __name__ == "__main__":
    main()
