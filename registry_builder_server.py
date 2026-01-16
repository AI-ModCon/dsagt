import asyncio
import subprocess
import httpx
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("tool-registry-server")

# Path to store the registry
REGISTRY_PATH = Path("tool_registry.json")

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
                        "description": "Tool specification in JSON format",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "command": {"type": "string"},
                            "parameters": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                        "description": {"type": "string"},
                                        "required": {"type": "boolean"},
                                        "default": {"type": "string"},
                                        "enum": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["name", "type", "description"]
                                }
                            },
                            "examples": {"type": "array", "items": {"type": "string"}},
                            "category": {"type": "string"}
                        },
                        "required": ["name", "description", "command", "parameters"]
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
                registry = json.loads(REGISTRY_PATH.read_text())
            else:
                registry = {"version": "1.0", "tools": []}

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

            # Save registry
            REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

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

            registry = json.loads(REGISTRY_PATH.read_text())
            return [TextContent(
                type="text",
                text=json.dumps(registry, indent=2)
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading registry: {str(e)}")]

    elif name == "search_registry":
        query = arguments.get("query", "").lower()
        category = arguments.get("category", "").lower()

        try:
            if not REGISTRY_PATH.exists():
                return [TextContent(type="text", text="Registry is empty.")]

            registry = json.loads(REGISTRY_PATH.read_text())
            results = []

            for tool in registry["tools"]:
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
                    text=f"Found {len(results)} tool(s):\n\n" + json.dumps(results, indent=2)
                )]
            else:
                return [TextContent(type="text", text="No tools found matching the criteria.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching registry: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
