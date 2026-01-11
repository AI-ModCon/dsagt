#!/usr/bin/env python
"""
BASEDATA MCP Server

Exposes data processing tools to MCP-compatible agents (Goose, Claude, etc.).

Usage:
    python mcp_server.py
    python mcp_server.py --registry ./my_registry.yaml
    python mcp_server.py --runtime-dir ./my_session
"""

import argparse
import asyncio
import json
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


DEFAULT_REGISTRY = str(Path(__file__).parent / "registry.yaml")


class ToolRegistry:
    """
    Manages tool definitions and execution.
    
    Copies source registry to runtime directory on init.
    All modifications happen to the runtime copy.
    """
    
    def __init__(
        self, 
        source_registry: str = DEFAULT_REGISTRY,
        runtime_dir: str = "./runtime",
    ):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_registry = self.runtime_dir / "registry.yaml"
        self.provenance_log = self.runtime_dir / "provenance.log"
        
        # Create runtime directory and copy registry
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_registry, self.runtime_registry)
        
        # Initialize provenance log
        with open(self.provenance_log, "a") as f:
            f.write(f"# Session started: {datetime.now().isoformat()}\n")
            f.write(f"# Source registry: {source_registry}\n\n")
    
    def _load_registry(self) -> dict:
        with open(self.runtime_registry) as f:
            return yaml.safe_load(f)
    
    def _save_registry(self, registry: dict) -> None:
        with open(self.runtime_registry, "w") as f:
            yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    
    def list_tools(self) -> list[dict]:
        """List all tools with MCP-compatible schemas."""
        registry = self._load_registry()
        tools = []
        
        for tool in registry.get("tools", []):
            properties = {}
            required = []
            
            for param_name, param_def in tool.get("parameters", {}).items():
                properties[param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", ""),
                }
                if "default" in param_def:
                    properties[param_name]["default"] = param_def["default"]
                if param_def.get("required", False):
                    required.append(param_name)
            
            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        
        return tools
    
    def get_tool(self, name: str) -> dict | None:
        registry = self._load_registry()
        for tool in registry.get("tools", []):
            if tool["name"] == name:
                return tool
        return None
    
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return result."""
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "output": "", "error": f"Unknown tool: {name}"}
        
        # Build command
        cmd = tool["executable"]
        params = tool.get("parameters", {})
        
        for param_name, param_def in params.items():
            value = arguments.get(param_name, param_def.get("default"))
            if value is not None:
                if param_def.get("required", False):
                    cmd += f" {shlex.quote(str(value))}"
                else:
                    cmd += f" --{param_name} {shlex.quote(str(value))}"
        
        self._log_provenance(name, arguments, cmd)
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
        }
    
    def register_tool(
        self,
        name: str,
        description: str,
        executable: str,
        parameters: dict[str, dict],
    ) -> dict[str, Any]:
        """Register a new tool."""
        registry = self._load_registry()
        
        for tool in registry.get("tools", []):
            if tool["name"] == name:
                return {"success": False, "error": f"Tool '{name}' already exists"}
        
        registry.setdefault("tools", []).append({
            "name": name,
            "description": description,
            "executable": executable,
            "parameters": parameters,
        })
        
        self._save_registry(registry)
        self._log_provenance("_register_tool", {"name": name, "executable": executable}, f"Registered: {name}")
        
        return {"success": True}
    
    def _log_provenance(self, tool: str, args: dict, cmd: str) -> None:
        with open(self.provenance_log, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {tool} | {json.dumps(args)} | {cmd}\n")


def main():
    parser = argparse.ArgumentParser(description="BASEDATA MCP Server")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="Source registry YAML")
    parser.add_argument("--runtime-dir", default="./runtime", help="Runtime directory")
    args = parser.parse_args()
    
    registry = ToolRegistry(source_registry=args.registry, runtime_dir=args.runtime_dir)
    server = Server("basedata")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in registry.list_tools()
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "register_tool":
            result = registry.register_tool(
                name=arguments["name"],
                description=arguments["description"],
                executable=arguments["executable"],
                parameters=arguments.get("parameters", {}),
            )
        else:
            result = registry.call_tool(name, arguments)
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
