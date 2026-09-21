"""
Agent capabilities: the project's skills directory and its two registries.

A project stores and extends agent capabilities in ``<project>/skills/``.
Each entry's directory holds a ``SKILL.md`` whose YAML frontmatter gives its
name and description, plus any scripts and reference documents.  Some entries
are instructions the agent reads and follows.  Others also name an
``executable`` in that frontmatter, a command line (code) the agent can run.
``CodeRegistry`` reads/writes skills with associated codes whereas
``SkillRegistry`` reads/writes general purpose instructions/workflows.

Claude Code, Codex and the rest each discover skills from a directory of
their own, and ``AgentSetup.setup_skills`` links ``<project>/skills/`` there,
so registered codes/skills are in the agent's context at invocation.

The ``executable`` is stored as the whole command line rather than the
program alone::

    dsagt-run --code <name> -- [uv run --with <deps> --] <command>

Execution belongs to the agent's own shell, outside anything dsagt mediates,
so the dsagt-run wrapper that writes the execution record is part of the
stored string.

    CodeRegistry  ◇── knowledge.KnowledgeBase   (the `codes` collection)
    SkillRegistry ◇── knowledge.KnowledgeBase   (a catalog collection)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    # Annotation-only.  A runtime import would load the whole retrieval module
    # into anything that imports the registry, including ``dsagt-run`` through
    # the package ``__init__``; the registry holds an injected KB instance and
    # names the class only in annotations.
    from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

#: Single project-local collection holding the base-skill codes and the
#: registered (agent-saved) codes.  The base-skill codes are embedded once by
#: the shared knowledge-base build and copied into the project at init;
#: ``metadata.source`` says which kind an entry is.
CODES_COLLECTION = "codes"

#: External skill catalogs (fetched from GitHub repos) are stored one per
#: source in collections named ``skills_catalog__<slug>``, so a re-sync drops
#: and rebuilds one source's collection and leaves the other catalogs as they
#: are.
CATALOG_COLLECTION_PREFIX = "skills_catalog__"


def catalog_collection(slug: str) -> str:
    """KB collection name holding the indexed catalog for source *slug*."""
    return f"{CATALOG_COLLECTION_PREFIX}{slug}"


# ---------------------------------------------------------------------------
# Helpers (codes only)
# ---------------------------------------------------------------------------


def _uv_run_prefix(deps: list[str]) -> str:
    """Build a 'uv run --with dep1,dep2 --' prefix for Python dependencies."""
    if not deps:
        return ""
    return f"uv run --with {','.join(deps)} -- "


#: Skill-standard name charset: agent native skill loaders (claude et al.)
#: require lowercase-hyphen names, and codes mirror into those dirs.
_CODE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _wrap_executable(name: str, executable: str, deps: list[str] | None = None) -> str:
    """Wrap an executable with uv run (for Python deps) and dsagt-run (for provenance).

    Result: dsagt-run --code <name> -- [uv run --with deps --] <executable>
    """
    if "dsagt-run" in executable:
        return executable
    # An executable that already carries its own ``uv run`` gets no second one.
    prefix = "" if executable.startswith("uv run") else _uv_run_prefix(deps or [])
    return f"dsagt-run --code {name} -- {prefix}{executable}"


def code_metadata(spec: dict, source: str) -> dict:
    """The ``codes`` collection metadata for a spec whose executable is wrapped.

    *source* says who put the entry there: ``registered`` for an agent-saved
    code, ``base-skill`` for a code a base skill declares.  One builder so an entry embedded by the shared
    knowledge-base build and one indexed by :meth:`CodeRegistry.save_tool`
    have the same shape.
    """
    return {
        "code_name": spec["name"],
        "tags": ",".join(spec.get("tags", [])),
        "executable": spec["executable"],
        "has_dependencies": str(bool(spec.get("dependencies"))),
        "source": source,
    }


def render_code_spec(spec: dict) -> str:
    """Render a code's SKILL.md text from its spec.

    The executable is wrapped with ``dsagt-run`` and, when the spec declares
    dependencies, ``uv run --with``; the frontmatter is the wrapped spec and
    the body is generated from it.  :meth:`CodeRegistry.save_tool` writes
    this text for a new code, and the shared knowledge-base build embeds the
    same text for the base-skill codes, so the ``codes`` collection a project
    copies at init matches the files init writes.
    """
    wrapped = dict(spec)
    wrapped["executable"] = _wrap_executable(
        spec["name"], spec["executable"], spec.get("dependencies")
    )
    frontmatter = yaml.dump(wrapped, default_flow_style=False, sort_keys=False)
    return f"---\n{frontmatter}---\n{_generate_code_body(wrapped)}"


def _generate_code_body(spec: dict) -> str:
    """Generate a markdown body for a new code file from its spec.

    The exact runnable command leads the body: native skill discovery
    injects SKILL.md at invocation time, and the thing the agent must copy
    verbatim (the dsagt-run-wrapped command) belongs at the top, not after
    prose it may stop reading.
    """
    lines = [
        f"\n# {spec['name']}\n\n",
        "Run this registered code with the exact shell command below: copy "
        "it byte-for-byte (the `dsagt-run` prefix writes the execution "
        "record to `trace_archive/` when the process exits, so run it in the "
        "foreground and wait; never a background task or a background "
        "subagent, which end with the turn):"
        "\n\n```bash\n",
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

    Third-party skill catalogs (Genesis, for example) carry SKILL.md files
    whose frontmatter is intended as flat ``key: value`` but is not strict
    YAML, most commonly an unquoted ``description`` value that contains a
    colon (``...readiness levels: Level 1...``), which PyYAML rejects as a
    nested mapping. Rather than silently dropping such skills from discovery, fall back
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
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        # Benign: the frontmatter is flat ``key: value`` but not strict YAML;
        # the fields are recovered below.  DEBUG, not WARNING: nothing is lost
        # and a warning per file is noise during ``dsagt init`` catalog
        # indexing.
        logger.debug(
            "Frontmatter in %s is not strict YAML (%s); recovering flat fields.",
            path,
            str(e).splitlines()[0],
        )
        return _lenient_frontmatter(parts[1])
    # safe_load gives a scalar or list for a non-mapping body (a bare prose
    # frontmatter, for example); callers do ``spec.get(...)``, so a dict is
    # always returned.
    return data if isinstance(data, dict) else _lenient_frontmatter(parts[1])


def _lenient_frontmatter(block: str) -> dict:
    """Best-effort flat ``key: value`` parse for frontmatter that is not strict YAML.

    Splits each top-level line on its first colon (so a value may itself
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
#   positional           first positional slot
#   positional:N         Nth positional slot (0-based)
#   --name               `--name <value>` (spaced long flag)
#   -n                   `-n <value>`    (spaced short flag)
#   --name=              `--name=<value>` (glued long flag)
#   -n=                  `-n=<value>`    (glued short flag)
#   key=                 `key=<value>`   (dd-style, no dashes)
#
# A missing `cli` field defaults to `--<param_name>`.  Parameters with
# `type: boolean` render as a bare flag when truthy and emit nothing when
# falsy; a boolean parameter must use a flag form (``render_arguments``
# raises for any other).


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

    Returns only the parameter portion; the caller prepends the executable.
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
    """The entries whose frontmatter names an ``executable``.

    A code's directory holds its spec and its scripts, so it is portable on
    its own.  ``save_tool`` is the one writer, called for a base skill's
    script, a catalog skill's, an agent-authored one, and a single script
    saved through ``save_code_spec``, so all four arrive in the same shape.
    Indexes into the project's ``codes`` collection when a knowledge base is
    given, which is what ``search_registry`` reads.
    """

    def __init__(
        self,
        runtime_dir: str | Path,
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.codes_dir = self.runtime_dir / "skills"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.codes_dir.mkdir(parents=True, exist_ok=True)

    def _project_code_paths(self) -> list[Path]:
        """The SKILL.md paths under ``skills/`` whose frontmatter declares an
        executable."""
        return [
            p
            for p in sorted(self.codes_dir.glob("*/SKILL.md"))
            if _parse_frontmatter(p).get("executable")
        ]

    def code_dirs(self) -> list[Path]:
        """All code directories (for the native-skills mirror,
        ``AgentSetup.setup_skills``)."""
        return [p.parent for p in self._project_code_paths()]

    def list_codes_raw(self) -> list[dict]:
        """Return full frontmatter dicts for all codes in the project."""
        seen: dict[str, dict] = {}
        for p in self._project_code_paths():
            spec = _parse_frontmatter(p)
            name = spec.get("name")
            if name:
                seen[name] = spec
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
                    "description": code.get("description", ""),
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return codes

    def get_code(self, name: str) -> dict | None:
        """Look up a code spec by name."""
        path = self.codes_dir / name / "SKILL.md"
        if path.exists():
            code = _parse_frontmatter(path)
            if code.get("name") == name and code.get("executable"):
                return code
        return None

    def save_tool(self, spec: dict) -> str:
        """Write or update a code's SKILL.md; returns 'added' or 'updated'.

        The spec's ``executable`` is stored as the whole command line the
        agent runs: ``uv run --with <deps> --`` when the spec declares
        dependencies, under ``dsagt-run --code <name> --``.  An existing
        frontmatter is kept underneath the spec's keys and a hand-edited body
        survives, because a skill whose CLI is a code of its own name (aidrin)
        would otherwise lose its upstream text to the spec.  Indexes the code
        when a knowledge base is given.
        """
        # Codes share the skill-standard envelope so they mirror into agent
        # native skills dirs, whose loaders require lowercase-hyphen names.
        if not _CODE_NAME_RE.match(spec["name"]):
            raise ValueError(
                f"invalid code name {spec['name']!r}: use lowercase letters, "
                "digits, and hyphens (the skill-standard charset agent native "
                "skill loaders require), e.g. 'datacard-introspect'"
            )
        code_dir = self.codes_dir / spec["name"]
        path = code_dir / "SKILL.md"
        action = "updated" if path.exists() else "added"
        code_dir.mkdir(parents=True, exist_ok=True)

        wrapped = dict(spec)
        wrapped["executable"] = _wrap_executable(
            spec["name"],
            spec["executable"],
            spec.get("dependencies"),
        )

        # Preserve existing body when updating so hand-edited docs survive.
        # An existing frontmatter is kept underneath the spec's keys: a skill
        # whose CLI is a code of its own name (aidrin) keeps its upstream
        # fields, and its description, which is the skill's, on the first
        # registration; a code re-saved by the agent takes every key from
        # the new spec.
        body = ""
        existing: dict = {}
        if path.exists():
            parts = path.read_text().split("---", 2)
            if len(parts) == 3:
                body = parts[2]
            existing = _parse_frontmatter(path)

        if body:
            merged = {**existing, **wrapped}
            if existing.get("description") and not existing.get("executable"):
                merged["description"] = existing["description"]
            frontmatter = yaml.dump(merged, default_flow_style=False, sort_keys=False)
            path.write_text(f"---\n{frontmatter}---\n{body}")
            wrapped = merged
        else:
            path.write_text(render_code_spec(spec))

        if self._kb:
            self._index_code(wrapped, path)

        return action

    def _index_code(self, spec: dict, tool_path: Path) -> None:
        """Index a code file into the ``codes`` KB collection.

        Errors propagate to the caller: a code that is on disk and absent
        from the KB is a state the agent cannot recover from (it would write
        a duplicate the next time it searched).  Registration is atomic: in
        the index, or not registered.
        """
        self._kb.add_entries(
            texts=[tool_path.read_text()],
            collection=CODES_COLLECTION,
            metadatas=[code_metadata(spec, "registered")],
        )


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Every entry in the directory, whether or not it names an executable.

    A skill arrives three ways, and each ends here: the base skills ``dsagt
    init`` installs from their upstream repositories, a catalog skill through
    ``install_skill``, and the agent's own through ``save_skill``.  The
    package holds no skills of its own.  ``skill_dirs`` is what the native
    mirror (``AgentSetup.setup_skills``) copies, so it lists codes too.
    """

    def __init__(
        self,
        runtime_dir: str | Path,
        kb: KnowledgeBase | None = None,
    ):
        self.runtime_dir = Path(runtime_dir)
        self.skills_dir = self.runtime_dir / "skills"
        self._kb = kb
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def skill_dirs(self) -> list[Path]:
        """All skill directories in the project (for the native-skills
        mirror, ``AgentSetup.setup_skills``)."""
        return [
            d
            for d in sorted(self.skills_dir.iterdir())
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

    def list_skills(self) -> list[dict]:
        """Return the frontmatter of each skill whose SKILL.md has a name."""
        seen: dict[str, dict] = {}
        for d in self.skill_dirs():
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
        optional ``tags``).  ``body`` is the markdown after the frontmatter,
        typically the workflow the agent follows.  ``reference_files`` is an
        optional mapping ``{relative_path: contents}`` for additional files
        in the skill's directory (templates, schemas).

        Returns "added" or "updated".  A saved skill is written to
        ``<project>/skills/``, where every supported agent discovers it
        natively, so catalog search (``SkillRouter``) covers only skills
        not yet installed.
        """
        name = spec.get("name")
        if not name:
            raise ValueError("save_skill: spec must include 'name'")
        skill_dir = self.skills_dir / name
        action = "updated" if skill_dir.exists() else "added"
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md = skill_dir / "SKILL.md"
        # Preserve a hand-edited body when updating, unless the caller passed
        # an explicit replacement, the same contract as CodeRegistry.save_tool.
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
        """Resolve a skill name to its SKILL.md, or None if not installed."""
        path = self.skills_dir / name / "SKILL.md"
        return path if path.exists() else None

    def get_skill(self, name: str) -> dict | None:
        """Get a skill's frontmatter by name."""
        path = self._skill_md_path(name)
        if path is None:
            return None
        spec = _parse_frontmatter(path)
        return spec if spec.get("name") else None

    def get_skill_content(self, name: str) -> str | None:
        """Get the full SKILL.md content for a skill."""
        path = self._skill_md_path(name)
        return path.read_text() if path is not None else None
