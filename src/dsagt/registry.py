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
from pathlib import Path

import yaml

from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

#: Single project-local collection holding both bundled (package-shipped)
#: and registered (agent-saved) tools.  Bundled entries carry
#: ``metadata.source = "bundled"`` and ``metadata.dsagt_version`` so
#: they can be evicted and refreshed on dsagt upgrade without touching
#: agent-registered entries.
TOOLS_COLLECTION = "tools"
SKILLS_COLLECTION = "skills"

#: Backwards-compat aliases — kept so external code that imported the
#: previous names still resolves.  New code should use the names above.
TOOL_REGISTRY_COLLECTION = TOOLS_COLLECTION
SKILL_REGISTRY_COLLECTION = SKILLS_COLLECTION


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

    Two layers:

    * **Bundled tools** ship with the dsagt package at
      ``_PACKAGE_TOOLS_DIR``.  They are read-only; their KB embeddings
      live in the shared ``bundled_tools`` collection (built once per
      machine per dsagt version by ``dsagt setup-kb``).  Never copied
      into projects, so package upgrades automatically reach all
      existing projects.
    * **Project tools** are agent-saved or user-edited specs in
      ``<project>/tools/``.  Embeddings go into the project-local
      ``registered_tools`` collection on save.

    Listing / lookup methods merge both layers (project wins on name
    collision so agents can override a bundled tool).  Search the
    KB-side via ``search_registry`` which queries both collections.
    """

    _PACKAGE_TOOLS_DIR = Path(__file__).parent / "tools"

    def __init__(
        self,
        runtime_dir: str | Path,
        source_tools_dir: str | None = None,
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.tools_dir = self.runtime_dir / "tools"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        # Optional override of the package-bundled directory (used by
        # tests; production callers leave source_tools_dir=None and let
        # the package dir stand).
        self._bundled_dir = (
            Path(source_tools_dir)
            if source_tools_dir and Path(source_tools_dir).exists()
            else self._PACKAGE_TOOLS_DIR
        )

        # Project tool dir is always agent-writable.  We no longer
        # pre-populate it with bundled tools — they're served directly
        # from the package via the merge in list_tools / get_tool.
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        # Ensure code/ subdirectory exists for agent-written scripts.
        (self.tools_dir / "code").mkdir(exist_ok=True)

    def _bundled_tool_paths(self) -> list[Path]:
        """Return .md tool spec paths shipped with the package."""
        if not self._bundled_dir.exists():
            return []
        return sorted(self._bundled_dir.glob("*.md"))

    def _project_tool_paths(self) -> list[Path]:
        """Return .md tool spec paths the agent has saved into this project."""
        return sorted(self.tools_dir.glob("*.md"))

    def list_tools_raw(self) -> list[dict]:
        """Return full frontmatter dicts for all tools.

        Merges bundled (package) + project (``<project>/tools/``).
        Project tools win on name collision so agents can override a
        bundled tool with their own implementation.
        """
        seen: dict[str, dict] = {}
        for p in self._bundled_tool_paths():
            spec = _parse_frontmatter(p)
            name = spec.get("name")
            if name:
                seen[name] = spec
        for p in self._project_tool_paths():
            spec = _parse_frontmatter(p)
            name = spec.get("name")
            if name:
                seen[name] = spec  # project layer overrides bundled
        return [seen[name] for name in sorted(seen)]

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
        """Look up a tool spec by name.  Project layer overrides bundled."""
        project_path = self.tools_dir / f"{name}.md"
        if project_path.exists():
            tool = _parse_frontmatter(project_path)
            if tool.get("name") == name:
                return tool
        bundled_path = self._bundled_dir / f"{name}.md"
        if bundled_path.exists():
            tool = _parse_frontmatter(bundled_path)
            if tool.get("name") == name:
                return tool
        return None

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
        """Index a tool file into the ``tools`` KB collection.

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
            "source": "registered",  # vs "bundled" — see ensure_bundled_*
        }
        self._kb.add_entries(
            texts=[text],
            collection=TOOLS_COLLECTION,
            metadatas=[metadata],
        )

    def reindex_all(self) -> int:
        """Reindex project-local tool files into the ``tools`` collection.

        Returns count indexed.  Bundled tools are NOT indexed here — they
        live in the shared ``tools`` collection built by ``dsagt setup-kb``
        and copied into the project at ``dsagt init`` time.  Search via
        ``search_registry`` queries the merged collection.
        """
        if not self._kb:
            return 0
        count = 0
        for path in self._project_tool_paths():
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

    Two layers (mirroring ToolRegistry):

    * **Bundled skills** ship with the dsagt package at
      ``_PACKAGE_SKILLS_DIR``.  Read-only; embeddings live in shared
      ``bundled_skills`` collection.
    * **Project skills** are user/agent-edited skills in
      ``<project>/skills/``.  Embeddings go into project-local
      ``registered_skills`` on save.

    List / lookup methods merge both layers; project wins on name
    collision so a project can override a bundled skill.
    """

    _PACKAGE_SKILLS_DIR = Path(__file__).parent / "skills"

    def __init__(
        self,
        runtime_dir: str | Path,
        source_skills_dir: str | None = None,
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.skills_dir = self.runtime_dir / "skills"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        # Optional override of the package-bundled directory (tests).
        self._bundled_dir = (
            Path(source_skills_dir)
            if source_skills_dir and Path(source_skills_dir).exists()
            else self._PACKAGE_SKILLS_DIR
        )

        # Project skill dir always exists; agents save into it.  We no
        # longer seed it with bundled skills — they're served directly
        # from the package via the merge in list_skills / get_skill.
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _bundled_skill_dirs(self) -> list[Path]:
        """Return skill directories shipped with the package."""
        if not self._bundled_dir.exists():
            return []
        return [
            d for d in sorted(self._bundled_dir.iterdir())
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

    def _project_skill_dirs(self) -> list[Path]:
        """Return skill directories the agent has saved into this project."""
        return [
            d for d in sorted(self.skills_dir.iterdir())
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

    def list_skills(self) -> list[dict]:
        """Return name + description for each skill (bundled + project, project wins)."""
        seen: dict[str, dict] = {}
        for d in self._bundled_skill_dirs():
            spec = _parse_frontmatter(d / "SKILL.md")
            if spec.get("name"):
                seen[spec["name"]] = spec
        for d in self._project_skill_dirs():
            spec = _parse_frontmatter(d / "SKILL.md")
            if spec.get("name"):
                seen[spec["name"]] = spec
        return [seen[name] for name in sorted(seen)]

    def save_skill(
        self,
        spec: dict,
        body: str | None = None,
        reference_files: dict[str, str] | None = None,
    ) -> str:
        """Write or update a skill in ``<project>/skills/<name>/``.

        ``spec`` carries the YAML frontmatter (``name``, ``description``,
        optional ``tags``).  ``body`` is the markdown after the
        frontmatter — typically the workflow / instructions the agent
        will follow.  ``reference_files`` is an optional mapping
        ``{relative_path: contents}`` for additional files the skill
        wants in its directory (templates, schemas, etc.).

        Returns "added" or "updated".  Indexes the resulting SKILL.md
        into ``registered_skills`` via ``_index_skill`` if a KB is
        configured — symmetric with ``ToolRegistry.save_tool``.
        """
        name = spec.get("name")
        if not name:
            raise ValueError("save_skill: spec must include 'name'")

        skill_dir = self.skills_dir / name
        action = "updated" if skill_dir.exists() else "added"
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md = skill_dir / "SKILL.md"
        # Preserve hand-edited body when updating, unless caller passed
        # an explicit replacement — same contract as ToolRegistry.save_tool.
        if body is None and skill_md.exists():
            existing = skill_md.read_text()
            parts = existing.split("---", 2)
            if len(parts) == 3:
                body = parts[2]
        if body is None:
            body = f"\n# {name}\n\n{spec.get('description', '')}\n"

        frontmatter = yaml.dump(spec, default_flow_style=False, sort_keys=False)
        skill_md.write_text(f"---\n{frontmatter}---\n{body}")

        for rel_path, contents in (reference_files or {}).items():
            target = skill_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)

        if self._kb:
            self._index_skill(spec, skill_md)

        return action

    def _skill_md_path(self, name: str) -> Path | None:
        """Resolve a skill name to its SKILL.md, project layer first."""
        project = self.skills_dir / name / "SKILL.md"
        if project.exists():
            return project
        bundled = self._bundled_dir / name / "SKILL.md"
        if bundled.exists():
            return bundled
        return None

    def get_skill(self, name: str) -> dict | None:
        """Get a skill's frontmatter by name.  Project overrides bundled."""
        path = self._skill_md_path(name)
        if path is None:
            return None
        spec = _parse_frontmatter(path)
        return spec if spec.get("name") else None

    def get_skill_content(self, name: str) -> str | None:
        """Get the full SKILL.md content for a skill."""
        path = self._skill_md_path(name)
        return path.read_text() if path is not None else None

    def _index_skill(self, spec: dict, skill_md: Path) -> None:
        """Index a skill into the ``skills`` KB collection.

        Errors propagate to the caller — see _index_tool for the rationale.
        """
        text = skill_md.read_text()
        metadata = {
            "skill_name": spec["name"],
            "tags": ",".join(spec.get("tags", [])),
            "source": "registered",  # vs "bundled"
        }
        self._kb.add_entries(
            texts=[text],
            collection=SKILLS_COLLECTION,
            metadatas=[metadata],
        )

    def reindex_all(self) -> int:
        """Reindex project-local skills into ``registered_skills``.

        Bundled skills are NOT indexed here — they live in the shared
        ``bundled_skills`` collection (see ToolRegistry.reindex_all
        docstring for the same architecture).
        """
        if not self._kb:
            return 0
        count = 0
        for skill_dir in self._project_skill_dirs():
            skill_md = skill_dir / "SKILL.md"
            spec = _parse_frontmatter(skill_md)
            if spec.get("name"):
                self._index_skill(spec, skill_md)
                count += 1
        return count
