"""
DSAGT Registry MCP Server

Provides tools for building a tool registry by reading documentation,
fetching web resources, and running commands to extract tool specifications.

Tool specs are saved as skill markdown files in the runtime skills directory.

Usage:
    dsagt-registry-server
    dsagt-registry-server --runtime-dir ./my_session
"""

import argparse
import asyncio
import subprocess
import sys

import httpx
import yaml

from dsagt.mcp_utils import create_server, run_stdio, text_result, types
from dsagt.registry import ToolRegistry


def _install_dependencies(packages: list[str], timeout: int = 120) -> str:
    """Install packages using uv pip install. Returns a status string."""
    cmd = ["uv", "pip", "install", "--python", sys.executable] + packages
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            return f"Successfully installed: {', '.join(packages)}\n{output}"
        else:
            return (
                f"Installation failed (exit code {result.returncode}):\n"
                f"{result.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        return f"Installation timed out after {timeout}s for: {', '.join(packages)}"
    except FileNotFoundError:
        return "Error: 'uv' command not found. Install uv: https://github.com/astral-sh/uv"


def create_registry_server(registry: ToolRegistry):
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
                description="Save a tool specification to the registry as a skill file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "object",
                            "description": "Tool specification",
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
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Python packages to install (e.g., ['pandas>=2.0', 'numpy'])",
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
                description="Search for tools in the registry by name or description",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {"type": "string", "description": "Filter by category"},
                    },
                },
            ),
            types.Tool(
                name="install_dependencies",
                description="Install Python dependencies for one or all tools in the registry using uv pip install",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Install deps for a specific tool (omit for all tools)",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "read_file":
            from pathlib import Path
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
                action = registry.save_tool(spec)
                tool_count = len(registry.list_tools_raw())
                message = (
                    f"Tool '{spec['name']}' {action} successfully. "
                    f"Registry now contains {tool_count} tools."
                )
                deps = spec.get("dependencies", [])
                if deps:
                    dep_result = _install_dependencies(deps)
                    message += f"\n\nDependency installation:\n{dep_result}"
                return text_result(message)
            except Exception as e:
                return text_result(f"Error saving tool spec: {e}")

        elif name == "get_registry":
            tools = registry.list_tools_raw()
            if not tools:
                return text_result("Registry is empty. No tools registered yet.")
            return text_result(
                yaml.dump({"tools": tools}, default_flow_style=False, sort_keys=False)
            )

        elif name == "search_registry":
            query = arguments.get("query", "").lower()
            category = arguments.get("category", "").lower()

            tools = registry.list_tools_raw()
            if not tools:
                return text_result("Registry is empty.")

            results = []
            for tool in tools:
                if query and query not in tool.get("name", "").lower() and query not in tool.get("description", "").lower():
                    continue
                if category and tool.get("category", "").lower() != category:
                    continue
                results.append(tool)

            if results:
                return text_result(
                    f"Found {len(results)} tool(s):\n\n"
                    + yaml.dump(results, default_flow_style=False, sort_keys=False)
                )
            return text_result("No tools found matching the criteria.")

        elif name == "install_dependencies":
            tool_name = arguments.get("tool_name")
            tools = registry.list_tools_raw()
            if not tools:
                return text_result("Registry is empty. No tools registered yet.")

            all_deps = []
            tools_with_deps = []
            for tool in tools:
                if tool_name and tool.get("name") != tool_name:
                    continue
                tool_deps = tool.get("dependencies", [])
                if tool_deps:
                    all_deps.extend(tool_deps)
                    tools_with_deps.append(tool["name"])

            if not all_deps:
                scope = f"tool '{tool_name}'" if tool_name else "registry"
                return text_result(f"No dependencies declared in {scope}.")

            seen = set()
            unique_deps = [d for d in all_deps if not (d in seen or seen.add(d))]

            result = _install_dependencies(unique_deps)
            return text_result(
                f"Installing dependencies for: {', '.join(tools_with_deps)}\n\n{result}"
            )

        raise ValueError(f"Unknown tool: {name}")

    return server


def main():
    """Entry point for the registry builder server."""
    parser = argparse.ArgumentParser(description="DSAGT Registry Builder MCP Server")
    parser.add_argument(
        "--runtime-dir",
        default="./runtime",
        help="Runtime directory (default: ./runtime)",
    )
    parser.add_argument(
        "--source-skills-dir",
        default=None,
        help="Source skills directory to seed from (default: bundled skills)",
    )
    args = parser.parse_args()

    registry = ToolRegistry(
        source_skills_dir=args.source_skills_dir,
        runtime_dir=args.runtime_dir,
    )
    server = create_registry_server(registry)
    asyncio.run(run_stdio(server, "registry"))


if __name__ == "__main__":
    main()
