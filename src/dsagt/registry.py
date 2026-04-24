"""
Tool and Skill Registries.

Two parallel registries for agent capabilities:

**Tools** (CLI executables) — markdown files with YAML frontmatter specifying
name, description, executable, parameters, dependencies, tags. Stored in
`<project>/tools/`. Agent-written scripts go in `<project>/tools/code/`.
When registered, executables are wrapped with dsagt-run + uv run --with.

**Skills** (agent instructions) — directories containing a SKILL.md with
YAML frontmatter (name, description, tags) and optional reference docs.
Stored in `<project>/skills/`. The agent reads SKILL.md and follows the
workflow instructions.

Both registries support optional KB indexing for semantic search via
`search_registry` (tools) and `search_skills` (skills) MCP tools.
"""

import logging
import shutil
from pathlib import Path

import yaml

from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

TOOL_REGISTRY_COLLECTION = "registered_tools"
SKILL_REGISTRY_COLLECTION = "registered_skills"


# ---------------------------------------------------------------------------
# Helpers (tools only)
# ---------------------------------------------------------------------------

def _uv_run_prefix(deps: list[str]) -> str:
    """Build a 'uv run --with dep1,dep2 --' prefix for Python dependencies."""
    if not deps:
        return ""
    return f"uv run --with {','.join(deps)} -- "


def _wrap_executable(name: str, executable: str, deps: list[str] | None = None) -> str:
    """Wrap an executable with uv run (for Python deps) and dsagt-run (for provenance).

    Result: dsagt-run --tool <name> -- [uv run --with deps --] <executable>
    """
    if "dsagt-run" in executable:
        return executable
    inner = f"{_uv_run_prefix(deps or [])}{executable}"
    return f"dsagt-run --tool {name} -- {inner}"


def _generate_tool_body(spec: dict) -> str:
    """Generate a markdown body for a new tool file from its spec."""
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


def _parse_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file."""
    text = path.read_text()
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter in {path}: {e}") from e


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------
#
# Each parameter in a tool spec may declare a `cli` field that pins how its
# value should be placed on the command line.  Supported forms:
#
#   positional         — first positional slot
#   positional:N       — Nth positional slot (0-based)
#   --name             — `--name <value>` (spaced long flag)
#   -n                 — `-n <value>`    (spaced short flag)
#   --name=            — `--name=<value>` (glued long flag)
#   -n=                — `-n=<value>`    (glued short flag)
#   key=               — `key=<value>`   (dd-style, no dashes)
#
# A missing `cli` field defaults to `--<param_name>` (the convention the
# agent was guessing before this field existed; avoids breaking old specs).
# Parameters with `type: boolean` render as a bare flag when truthy and emit
# nothing when falsy; positional booleans are not supported.

def _parse_cli(cli: str, param_name: str) -> dict:
    """Classify a cli string into a rendering descriptor. Fails fast on invalid input."""
    if cli == "positional":
        return {"kind": "positional", "position": 0}
    if cli.startswith("positional:"):
        try:
            return {"kind": "positional", "position": int(cli.split(":", 1)[1])}
        except ValueError:
            raise ValueError(
                f"Parameter {param_name!r}: cli position must be an integer, got {cli!r}"
            )
    glued = cli.endswith("=")
    body = cli[:-1] if glued else cli
    if body.startswith("-"):
        return {"kind": "flag", "flag": body, "glued": glued}
    if glued:
        # No dashes + trailing `=` → dd-style key=value
        return {"kind": "keyvalue", "prefix": cli}
    raise ValueError(
        f"Parameter {param_name!r}: invalid cli value {cli!r}. "
        f"Expected 'positional[:N]', '--name[=]', '-n[=]', or 'key='."
    )


def render_arguments(parameters: dict, values: dict) -> list[str]:
    """Render argv elements for *values* per each parameter's ``cli`` spec.

    Returns only the parameter portion — caller prepends the executable.
    Positional args are emitted in declared position order, followed by all
    named/keyvalue args in declaration order.
    """
    positionals: list[tuple[int, str]] = []
    named: list[str] = []

    for name, param in parameters.items():
        cli = param.get("cli", f"--{name}")
        descriptor = _parse_cli(cli, name)

        value = values.get(name, param.get("default"))
        if value is None:
            if param.get("required"):
                raise ValueError(f"Missing required parameter: {name!r}")
            continue

        is_bool = param.get("type") == "boolean"
        if is_bool:
            if descriptor["kind"] != "flag":
                raise ValueError(
                    f"Parameter {name!r}: boolean parameters must use a flag cli spec"
                )
            if value:
                named.append(descriptor["flag"])
            continue

        if descriptor["kind"] == "positional":
            positionals.append((descriptor["position"], str(value)))
        elif descriptor["kind"] == "flag":
            if descriptor["glued"]:
                named.append(f"{descriptor['flag']}={value}")
            else:
                named.extend([descriptor["flag"], str(value)])
        else:  # keyvalue
            named.append(f"{descriptor['prefix']}{value}")

    positionals.sort(key=lambda item: item[0])
    return [v for _, v in positionals] + named


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Manages CLI tool spec files and optional KB indexing.

    Copies source tools to the runtime directory on init.
    All modifications (new tools, updates) happen in the runtime copy.
    """

    _PACKAGE_TOOLS_DIR = Path(__file__).parent / "tools"

    def __init__(
        self,
        source_tools_dir: str | None = None,
        runtime_dir: str = "./runtime",
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.tools_dir = self.runtime_dir / "tools"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        if not self.tools_dir.exists():
            source = (
                Path(source_tools_dir)
                if source_tools_dir and Path(source_tools_dir).exists()
                else self._PACKAGE_TOOLS_DIR
            )
            shutil.copytree(source, self.tools_dir)

        # Ensure code/ subdirectory exists for agent-written scripts
        (self.tools_dir / "code").mkdir(exist_ok=True)

    def list_tools_raw(self) -> list[dict]:
        """Return full frontmatter dicts for all tool files."""
        return [
            _parse_frontmatter(p)
            for p in sorted(self.tools_dir.glob("*.md"))
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
        path = self.tools_dir / f"{name}.md"
        if not path.exists():
            return None
        tool = _parse_frontmatter(path)
        return tool if tool.get("name") == name else None

    def save_tool(self, spec: dict) -> str:
        """Write or update a tool file. Returns 'added' or 'updated'.

        Automatically wraps the executable:
        - With `uv run --with <deps>` if Python dependencies are specified
        - With `dsagt-run --tool <name>` for provenance capture

        If a KnowledgeBase is available, indexes the tool for semantic search.
        """
        path = self.tools_dir / f"{spec['name']}.md"
        action = "updated" if path.exists() else "added"

        spec = dict(spec)
        spec["executable"] = _wrap_executable(
            spec["name"], spec["executable"], spec.get("dependencies"),
        )

        # Preserve existing body when updating so hand-edited docs survive
        body = ""
        if path.exists():
            text = path.read_text()
            parts = text.split("---", 2)
            if len(parts) == 3:
                body = parts[2]

        if not body:
            body = _generate_tool_body(spec)

        frontmatter = yaml.dump(spec, default_flow_style=False, sort_keys=False)
        path.write_text(f"---\n{frontmatter}---\n{body}")

        if self._kb:
            self._index_tool(spec, path)

        return action

    def _index_tool(self, spec: dict, tool_path: Path) -> None:
        """Index a tool file into the registered_tools KB collection.

        Errors propagate to the caller — a tool that lives on disk but
        isn't searchable in the KB is a half-broken state that the agent
        cannot recover from (it would write a duplicate next time it
        searched).  Atomic registration: in the index or not registered.
        """
        text = tool_path.read_text()
        metadata = {
            "tool_name": spec["name"],
            "tags": ",".join(spec.get("tags", [])),
            "executable": spec["executable"],
            "has_dependencies": str(bool(spec.get("dependencies"))),
        }
        self._kb.add_entries(
            texts=[text],
            collection=TOOL_REGISTRY_COLLECTION,
            metadatas=[metadata],
        )

    def reindex_all(self) -> int:
        """Reindex all tool files into the KB. Returns count indexed."""
        if not self._kb:
            return 0
        count = 0
        for path in sorted(self.tools_dir.glob("*.md")):
            spec = _parse_frontmatter(path)
            if spec.get("name"):
                self._index_tool(spec, path)
                count += 1
        return count


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """
    Manages instruction-based agent skills and optional KB indexing.

    Each skill is a directory containing a SKILL.md with YAML frontmatter
    (name, description, tags) and optional reference/template files.

    Copies source skills to the runtime directory on init.
    """

    _PACKAGE_SKILLS_DIR = Path(__file__).parent / "skills"

    def __init__(
        self,
        source_skills_dir: str | None = None,
        runtime_dir: str = "./runtime",
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.skills_dir = self.runtime_dir / "skills"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        if not self.skills_dir.exists():
            self.skills_dir.mkdir()

        # Copy bundled skills that aren't already present
        self._seed_from(self._PACKAGE_SKILLS_DIR)

        # Copy user-specified skills
        if source_skills_dir and Path(source_skills_dir).exists():
            self._seed_from(Path(source_skills_dir))

    def _seed_from(self, source: Path) -> None:
        """Copy skill directories from source that don't already exist in runtime."""
        if not source.exists():
            return
        for skill_dir in source.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                dest = self.skills_dir / skill_dir.name
                if not dest.exists():
                    shutil.copytree(skill_dir, dest)

    def list_skills(self) -> list[dict]:
        """Return name + description for each skill."""
        skills = []
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and skill_md.exists():
                spec = _parse_frontmatter(skill_md)
                if spec.get("name"):
                    skills.append(spec)
        return skills

    def get_skill(self, name: str) -> dict | None:
        """Get a skill's frontmatter by name."""
        skill_dir = self.skills_dir / name
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None
        spec = _parse_frontmatter(skill_md)
        return spec if spec.get("name") else None

    def get_skill_content(self, name: str) -> str | None:
        """Get the full SKILL.md content for a skill."""
        skill_md = self.skills_dir / name / "SKILL.md"
        if skill_md.exists():
            return skill_md.read_text()
        return None

    def _index_skill(self, spec: dict, skill_md: Path) -> None:
        """Index a skill into the registered_skills KB collection.

        Errors propagate to the caller — see _index_tool for the rationale.
        """
        text = skill_md.read_text()
        metadata = {
            "skill_name": spec["name"],
            "tags": ",".join(spec.get("tags", [])),
        }
        self._kb.add_entries(
            texts=[text],
            collection=SKILL_REGISTRY_COLLECTION,
            metadatas=[metadata],
        )

    def reindex_all(self) -> int:
        """Reindex all skills into the KB. Returns count indexed."""
        if not self._kb:
            return 0
        count = 0
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if skill_dir.is_dir() and skill_md.exists():
                spec = _parse_frontmatter(skill_md)
                if spec.get("name"):
                    self._index_skill(spec, skill_md)
                    count += 1
        return count
