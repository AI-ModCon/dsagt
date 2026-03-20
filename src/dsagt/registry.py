"""
Tool Registry

Manages tool skill files, execution, and provenance logging.

Each tool is a markdown file with YAML frontmatter containing the machine-readable
spec (name, description, executable, parameters, dependencies) and a markdown body
with rich usage instructions for the agent.
"""

import json
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _generate_skill_body(spec: dict) -> str:
    """Generate a markdown body for a new skill file from its spec."""
    lines = [
        f"\n# {spec['name']}\n\n{spec['description']}\n\n",
        "## Shell Command\n\n```bash\n",
        f"{spec['executable']} [options]\n",
        "```\n\n## Parameters\n\n",
    ]
    params = spec.get("parameters", {})
    if params:
        lines.append("| Parameter | Required | Default | Description |\n")
        lines.append("|-----------|----------|---------|-------------|\n")
        for name, p in params.items():
            req = "yes" if p.get("required") else "no"
            default = p.get("default", "—")
            lines.append(f"| `{name}` | {req} | {default} | {p.get('description', '')} |\n")
    return "".join(lines)


class ToolRegistry:
    """
    Manages tool skill files and execution.

    Copies the source skills directory to the runtime directory on init.
    All modifications (new tools, updates) happen in the runtime copy.
    """

    _PACKAGE_SKILLS_DIR = Path(__file__).parent / "skills"

    def __init__(
        self,
        source_skills_dir: str | None = None,
        runtime_dir: str = "./runtime",
    ):
        self.runtime_dir = Path(runtime_dir)
        self.skills_dir = self.runtime_dir / "skills"
        self.provenance_log = self.runtime_dir / "provenance.log"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        if not self.skills_dir.exists():
            source = (
                Path(source_skills_dir)
                if source_skills_dir and Path(source_skills_dir).exists()
                else self._PACKAGE_SKILLS_DIR
            )
            shutil.copytree(source, self.skills_dir)

        self.base_dir = self.runtime_dir

        with open(self.provenance_log, "a") as f:
            f.write(f"# Session started: {datetime.now().isoformat()}\n")

    @staticmethod
    def _parse_skill_file(path: Path) -> dict:
        """Parse YAML frontmatter from a skill markdown file."""
        text = path.read_text()
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        return yaml.safe_load(parts[1]) or {}

    def list_tools_raw(self) -> list[dict]:
        """Return full frontmatter dicts for all skill files."""
        return [
            self._parse_skill_file(p)
            for p in sorted(self.skills_dir.glob("*.md"))
        ]

    def list_tools(self) -> list[dict]:
        """List all tools with MCP-compatible schemas."""
        tools = []
        for tool in self.list_tools_raw():
            if not tool.get("name"):
                continue
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
        path = self.skills_dir / f"{name}.md"
        if not path.exists():
            return None
        tool = self._parse_skill_file(path)
        return tool if tool.get("name") == name else None

    def save_tool(self, spec: dict) -> str:
        """Write or update a skill file. Returns 'added' or 'updated'."""
        path = self.skills_dir / f"{spec['name']}.md"
        action = "updated" if path.exists() else "added"

        # Preserve existing body when updating so hand-edited docs survive
        body = ""
        if path.exists():
            text = path.read_text()
            parts = text.split("---", 2)
            if len(parts) == 3:
                body = parts[2]

        if not body:
            body = _generate_skill_body(spec)

        frontmatter = yaml.dump(spec, default_flow_style=False, sort_keys=False)
        path.write_text(f"---\n{frontmatter}---\n{body}")
        return action

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return result."""
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "output": "", "error": f"Unknown tool: {name}"}

        cmd = tool["executable"]
        params = tool.get("parameters", {})

        for param_name, param_def in params.items():
            if param_name in arguments:
                value = arguments[param_name]
            elif param_def.get("required", False) and "default" in param_def:
                value = param_def["default"]
            else:
                continue

            is_boolean = param_def.get("type") == "boolean"
            is_positional = param_def.get("positional", False)

            if is_positional:
                cmd += f" {shlex.quote(str(value))}"
            elif is_boolean:
                if value:
                    cmd += f" --{param_name}"
            else:
                cmd += f" --{param_name} {shlex.quote(str(value))}"

        self._log_provenance(name, arguments, cmd)

        result = subprocess.run(
            ["bash", "-lc", cmd],
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
