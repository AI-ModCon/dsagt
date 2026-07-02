"""
Code and Skill Registries.

Two parallel registries for agent capabilities:

**Codes** (CLI executables) — skill-standard directories
(`<project>/codes/<name>/SKILL.md`) whose frontmatter carries the machine
fields (name, description, executable, parameters, dependencies, tags) on
top of the skill-required name/description.  Agent-written scripts live
beside their spec in `<project>/codes/<name>/scripts/`, making each
registered code a self-contained, portable directory.  The skill-standard
envelope means codes mirror into the agent's native skills dir unchanged
(see ``AgentSetup.setup_skills``) — native discovery puts the exact
runnable command in context at invocation time, alongside MCP discovery
via ``search_registry``.
When registered, executables are wrapped with dsagt-run + uv run --with.
The wrapper lives *inside* the stored shell command by design: execution
used to be dispatched by MCP-server tools, but agents routinely sidestepped
those with their own bash tools, losing provenance.  Baking dsagt-run into
the command the agent copies makes the bash path harmless — the residual
failure mode is an agent reconstructing the command from memory and
dropping the wrapper, which is why specs render the exact runnable command
and agent instructions say to copy it verbatim.

**Skills** (agent instructions) — directories containing a SKILL.md with
YAML frontmatter (name, description, tags) and optional reference docs.
Stored in `<project>/skills/`. The agent reads SKILL.md and follows the
workflow instructions.

Both registries support optional KB indexing for semantic search via
`search_registry` (codes) and `search_skills` (skills) MCP tools.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    # Annotation-only.  A runtime import would pull the whole retrieval module
    # into anything that touches the registry — including ``dsagt-run`` via the
    # package ``__init__`` — even though the registry only ever holds an
    # injected KB instance, never references the class.
    from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

#: Single project-local collection holding both bundled (package-shipped)
#: and registered (agent-saved) codes.  Bundled entries carry
#: ``metadata.source = "bundled"`` and ``metadata.dsagt_version`` so
#: they can be evicted and refreshed on dsagt upgrade without touching
#: agent-registered entries.
CODES_COLLECTION = "codes"
#: Legacy installed-skills collection.  No longer written or read: installed
#: skills are natively auto-discovered by every supported agent, so skill
#: search covers only the *catalog* tier below.  Kept as a name for back-compat
#: and ``dsagt info`` display of any pre-existing index.
SKILLS_COLLECTION = "skills"

#: External skill catalogs (fetched from GitHub repos) live in their own
#: per-source collections named ``skills_catalog__<slug>``.  Keeping each
#: source in its own collection lets a re-sync drop+rebuild one source's
#: directory without disturbing other catalogs — no delete-by-metadata
#: primitive needed.
CATALOG_COLLECTION_PREFIX = "skills_catalog__"


def catalog_collection(slug: str) -> str:
    """KB collection name holding the indexed catalog for source *slug*."""
    return f"{CATALOG_COLLECTION_PREFIX}{slug}"


#: Backwards-compat aliases — kept so external code that imported the
#: previous names still resolves.  New code should use the names above.
TOOL_REGISTRY_COLLECTION = CODES_COLLECTION
SKILL_REGISTRY_COLLECTION = SKILLS_COLLECTION


# ---------------------------------------------------------------------------
# Helpers (codes only)
# ---------------------------------------------------------------------------


def _uv_run_prefix(deps: list[str]) -> str:
    """Build a 'uv run --with dep1,dep2 --' prefix for Python dependencies."""
    if not deps:
        return ""
    return f"uv run --with {','.join(deps)} -- "


#: Skill-standard name charset — agent native skill loaders (claude et al.)
#: require lowercase-hyphen names, and codes mirror into those dirs.
_CODE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _wrap_executable(name: str, executable: str, deps: list[str] | None = None) -> str:
    """Wrap an executable with uv run (for Python deps) and dsagt-run (for provenance).

    Result: dsagt-run --code <name> -- [uv run --with deps --] <executable>
    """
    if "dsagt-run" in executable:
        return executable
    inner = f"{_uv_run_prefix(deps or [])}{executable}"
    return f"dsagt-run --code {name} -- {inner}"


def _generate_code_body(spec: dict) -> str:
    """Generate a markdown body for a new code file from its spec.

    The exact runnable command leads the body: native skill discovery
    injects SKILL.md at invocation time, and the thing the agent must copy
    verbatim (the dsagt-run-wrapped command) belongs at the top, not after
    prose it may stop reading.
    """
    lines = [
        f"\n# {spec['name']}\n\n",
        "Run this registered code with the exact shell command below — copy "
        "it byte-for-byte (the `dsagt-run` prefix writes the execution "
        "record to `trace_archive/`):\n\n```bash\n",
        f"{spec['executable']} [options]\n",
        "```\n\n",
        f"{spec['description']}\n\n## Parameters\n\n",
    ]
    params = spec.get("parameters", {})
    if params:
        lines.append("| Parameter | Required | Default | Description |\n")
        lines.append("|-----------|----------|---------|-------------|\n")
        for name, p in params.items():
            req = "yes" if p.get("required") else "no"
            default = p.get("default", "—")
            lines.append(
                f"| `{name}` | {req} | {default} | {p.get('description', '')} |\n"
            )
    return "".join(lines)


def _parse_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file.

    Third-party skill catalogs (e.g. Genesis) ship SKILL.md files whose
    frontmatter is *intended* as flat ``key: value`` but isn't strict YAML —
    most commonly an unquoted ``description`` value that contains a colon
    (``...readiness levels: Level 1...``), which PyYAML rejects as a nested
    mapping. Rather than silently dropping such skills from discovery, fall back
    to a best-effort flat parse (:func:`_lenient_frontmatter`) on YAML error so
    ``name`` / ``description`` / ``tags`` are still recovered. dsagt-authored
    code/skill specs are valid YAML, so the fallback never fires for them.
    """
    text = path.read_text()
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        # Benign: the frontmatter is flat ``key: value`` but not strict YAML;
        # we recover the fields below.  DEBUG, not WARNING — nothing is lost
        # and it's pure noise during ``dsagt init`` catalog indexing.
        logger.debug(
            "Frontmatter in %s isn't strict YAML (%s); recovering flat fields.",
            path,
            str(e).splitlines()[0],
        )
        return _lenient_frontmatter(parts[1])


def _lenient_frontmatter(block: str) -> dict:
    """Best-effort flat ``key: value`` parse for frontmatter that isn't strict YAML.

    Splits each top-level line on its **first** colon (so a value may itself
    contain colons); indented ``- item`` lines extend the previous key into a
    list, other indented lines continue the previous string value. Inline
    ``[...]`` / ``{...}`` values are parsed as YAML when they can be. Lines
    without a colon, and comments, are ignored. This recovers the discovery
    fields (name/description/tags) from technically-invalid-but-obvious
    frontmatter instead of dropping the skill.
    """
    out: dict = {}
    key: str | None = None
    for raw in block.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1].isspace() and key is not None:
            # Continuation of the previous key.
            if stripped.startswith("- "):
                if not isinstance(out.get(key), list):
                    out[key] = []
                out[key].append(stripped[2:].strip())
            elif isinstance(out.get(key), str):
                out[key] = (out[key] + " " + stripped).strip()
            continue
        if ":" not in stripped:
            continue
        k, _, v = stripped.partition(":")
        key = k.strip()
        v = v.strip()
        if v.startswith(("[", "{")):
            try:
                out[key] = yaml.safe_load(v)
            except yaml.YAMLError:
                out[key] = v
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------
#
# Each parameter in a code spec may declare a `cli` field that pins how its
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
# Code Registry
# ---------------------------------------------------------------------------


class CodeRegistry:
    """
    Manages CLI code spec files and optional KB indexing.

    Two layers:

    * **Bundled codes** ship with the dsagt package at
      ``_PACKAGE_CODES_DIR``.  They are read-only; their KB embeddings
      live in the shared ``bundled_tools`` collection (built once per
      machine per dsagt version by ``dsagt init``).  Never copied
      into projects, so package upgrades automatically reach all
      existing projects.
    * **Project codes** are agent-saved or user-edited specs in
      ``<project>/codes/``.  Embeddings go into the project-local
      ``registered_tools`` collection on save.

    Listing / lookup methods merge both layers (project wins on name
    collision so agents can override a bundled code).  Search the
    KB-side via ``search_registry`` which queries both collections.
    """

    _PACKAGE_CODES_DIR = Path(__file__).parent / "codes"

    def __init__(
        self,
        runtime_dir: str | Path,
        source_tools_dir: str | None = None,
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.codes_dir = self.runtime_dir / "codes"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        # Optional override of the package-bundled directory (used by
        # tests; production callers leave source_tools_dir=None and let
        # the package dir stand).
        self._bundled_dir = (
            Path(source_tools_dir)
            if source_tools_dir and Path(source_tools_dir).exists()
            else self._PACKAGE_CODES_DIR
        )

        # Project code dir is always agent-writable.  We no longer
        # pre-populate it with bundled codes — they're served directly
        # from the package via the merge in list_codes / get_code.
        # Each code is a self-contained skill-standard directory
        # (``codes/<name>/SKILL.md`` + optional ``scripts/``), so there is
        # no shared scripts/ dir to pre-create.
        self.codes_dir.mkdir(parents=True, exist_ok=True)

    def _bundled_code_paths(self) -> list[Path]:
        """Return SKILL.md spec paths shipped with the package."""
        if not self._bundled_dir.exists():
            return []
        return sorted(self._bundled_dir.glob("*/SKILL.md"))

    def _project_code_paths(self) -> list[Path]:
        """Return SKILL.md spec paths the agent has saved into this project."""
        return sorted(self.codes_dir.glob("*/SKILL.md"))

    def code_dirs(self) -> list[Path]:
        """All code directories, bundled first so project wins downstream
        name collisions (mirror order — see ``AgentSetup.setup_skills``)."""
        return [p.parent for p in self._bundled_code_paths()] + [
            p.parent for p in self._project_code_paths()
        ]

    def list_codes_raw(self) -> list[dict]:
        """Return full frontmatter dicts for all codes.

        Merges bundled (package) + project (``<project>/codes/``).
        Project codes win on name collision so agents can override a
        bundled code with their own implementation.
        """
        seen: dict[str, dict] = {}
        for p in self._bundled_code_paths():
            spec = _parse_frontmatter(p)
            name = spec.get("name")
            if name:
                seen[name] = spec
        for p in self._project_code_paths():
            spec = _parse_frontmatter(p)
            name = spec.get("name")
            if name:
                seen[name] = spec  # project layer overrides bundled
        return [seen[name] for name in sorted(seen)]

    def list_codes(self) -> list[dict]:
        """List all codes with MCP-compatible schemas."""
        codes = []
        for code in self.list_codes_raw():
            if not code.get("name"):
                continue
            properties = {}
            required = []
            for param_name, param_def in code.get("parameters", {}).items():
                properties[param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", ""),
                }
                if "default" in param_def:
                    properties[param_name]["default"] = param_def["default"]
                if param_def.get("required", False):
                    required.append(param_name)
            codes.append(
                {
                    "name": code["name"],
                    "description": code["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return codes

    def get_code(self, name: str) -> dict | None:
        """Look up a code spec by name.  Project layer overrides bundled."""
        project_path = self.codes_dir / name / "SKILL.md"
        if project_path.exists():
            code = _parse_frontmatter(project_path)
            if code.get("name") == name:
                return code
        bundled_path = self._bundled_dir / name / "SKILL.md"
        if bundled_path.exists():
            code = _parse_frontmatter(bundled_path)
            if code.get("name") == name:
                return code
        return None

    def save_tool(self, spec: dict) -> str:
        """Write or update a code's SKILL.md. Returns 'added' or 'updated'.

        Automatically wraps the executable:
        - With `uv run --with <deps>` if Python dependencies are specified
        - With `dsagt-run --code <name>` for provenance capture

        If a KnowledgeBase is available, indexes the code for semantic search.
        """
        # Codes share the skill-standard envelope so they mirror into agent
        # native skills dirs, whose loaders require lowercase-hyphen names.
        if not _CODE_NAME_RE.match(spec["name"]):
            raise ValueError(
                f"invalid code name {spec['name']!r}: use lowercase letters, "
                "digits, and hyphens (the skill-standard charset agent native "
                "skill loaders require), e.g. 'scan-directory'"
            )
        code_dir = self.codes_dir / spec["name"]
        path = code_dir / "SKILL.md"
        action = "updated" if path.exists() else "added"
        code_dir.mkdir(parents=True, exist_ok=True)

        spec = dict(spec)
        spec["executable"] = _wrap_executable(
            spec["name"],
            spec["executable"],
            spec.get("dependencies"),
        )

        # Preserve existing body when updating so hand-edited docs survive
        body = ""
        if path.exists():
            text = path.read_text()
            parts = text.split("---", 2)
            if len(parts) == 3:
                body = parts[2]

        if not body:
            body = _generate_code_body(spec)

        frontmatter = yaml.dump(spec, default_flow_style=False, sort_keys=False)
        path.write_text(f"---\n{frontmatter}---\n{body}")

        if self._kb:
            self._index_code(spec, path)

        return action

    def _index_code(self, spec: dict, tool_path: Path) -> None:
        """Index a code file into the ``codes`` KB collection.

        Errors propagate to the caller — a code that lives on disk but
        isn't searchable in the KB is a half-broken state that the agent
        cannot recover from (it would write a duplicate next time it
        searched).  Atomic registration: in the index or not registered.
        """
        text = tool_path.read_text()
        metadata = {
            "code_name": spec["name"],
            "tags": ",".join(spec.get("tags", [])),
            "executable": spec["executable"],
            "has_dependencies": str(bool(spec.get("dependencies"))),
            "source": "registered",  # vs "bundled" — see ensure_bundled_*
        }
        self._kb.add_entries(
            texts=[text],
            collection=CODES_COLLECTION,
            metadatas=[metadata],
        )

    def reindex_all(self) -> int:
        """Reindex project-local code files into the ``codes`` collection.

        Returns count indexed.  Bundled codes are NOT indexed here — they
        live in the shared ``codes`` collection built and copied into the
        project at ``dsagt init`` time.  Search via
        ``search_registry`` queries the merged collection.
        """
        if not self._kb:
            return 0
        count = 0
        for path in self._project_code_paths():
            spec = _parse_frontmatter(path)
            if spec.get("name"):
                self._index_code(spec, path)
                count += 1
        return count


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """
    Manages instruction-based agent skills and optional KB indexing.

    Two layers (mirroring CodeRegistry):

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
            d
            for d in sorted(self._bundled_dir.iterdir())
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

    def _project_skill_dirs(self) -> list[Path]:
        """Return skill directories the agent has saved into this project."""
        return [
            d
            for d in sorted(self.skills_dir.iterdir())
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

        Returns "added" or "updated".  Does **not** index into a KB:
        saved skills land in ``<project>/skills/`` where every supported
        agent natively auto-discovers them, so search only covers the
        not-yet-installed *catalog* tier (see ``SkillRouter``).  The old
        ``skills`` collection is no longer read by anything.
        """
        name = spec.get("name")
        if not name:
            raise ValueError("save_skill: spec must include 'name'")

        skill_dir = self.skills_dir / name
        action = "updated" if skill_dir.exists() else "added"
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md = skill_dir / "SKILL.md"
        # Preserve hand-edited body when updating, unless caller passed
        # an explicit replacement — same contract as CodeRegistry.save_tool.
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
