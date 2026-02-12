"""
Tool Registry

Manages tool definitions, execution, and provenance logging.
"""

import json
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class ToolRegistry:
    """
    Manages tool definitions and execution.
    
    Copies source registry to runtime directory on init.
    All modifications happen to the runtime copy.
    """
    
    # Default registry shipped with the package
    _PACKAGE_DIR = Path(__file__).parent
    _DEFAULT_REGISTRY = _PACKAGE_DIR / "registry.yaml"

    def __init__(
        self,
        source_registry: str | None = None,
        runtime_dir: str = "./runtime",
    ):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_registry = self.runtime_dir / "registry.yaml"
        self.provenance_log = self.runtime_dir / "provenance.log"
        
        # Resolve source registry: explicit path > default bundled
        if source_registry and Path(source_registry).exists():
            source = Path(source_registry)
        else:
            source = self._DEFAULT_REGISTRY
        
        # Store base directory for resolving relative tool paths
        self.base_dir = source.parent
        
        # Create runtime directory and copy registry
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, self.runtime_registry)
        
        # Initialize provenance log
        with open(self.provenance_log, "a") as f:
            f.write(f"# Session started: {datetime.now().isoformat()}\n")
            f.write(f"# Source registry: {source}\n\n")
        
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
        
        # Run from base directory so relative tool paths resolve correctly
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(self.base_dir),
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
        }
    
    def _log_provenance(self, tool: str, args: dict, cmd: str) -> None:
        with open(self.provenance_log, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {tool} | {json.dumps(args)} | {cmd}\n")
