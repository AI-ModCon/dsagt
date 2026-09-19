"""MCP tools for the tool registry, execution, and provenance.

The code-lifecycle surface of ``dsagt-server``: define a code spec
(``save_code_spec``), discover codes (``get_registry`` / ``search_registry``),
install a code's dependencies (``install_dependencies``), read the readiness
reports on record (``readiness_reports``), and reconstruct a reproducible
pipeline from the recorded executions (``reconstruct_pipeline``).  Execution
in the user's environment is ``dsagt-run``'s, from the agent's own shell, and
reads are the agent's own tools, so the server runs nothing for the agent.

Code specs are saved as ``skills/<name>/SKILL.md`` under the project and
indexed into a ChromaDB collection for semantic search.  The embedding
backend's routing (``EMBEDDING_BACKEND``, ``EMBEDDING_MODEL``,
``EMBEDDING_BASE_URL``) comes from the MCP env block; its key comes from the
shell or ``~/.config/dsagt/env``.

These definitions and handlers run inside the merged ``dsagt-server`` (see
:mod:`dsagt.mcp.server`); ``create_registry_server`` is a test-facing
constructor.  Skill tools (``save_skill`` / ``search_skills`` /
``install_skill``) are defined in :mod:`dsagt.mcp.skill_tools`.
"""

import asyncio
import json
import logging
import subprocess
import sys
from functools import partial
from pathlib import Path

import yaml

import mcp.types as types

from dsagt.knowledge import KnowledgeBase
from dsagt.mcp.server import build_dispatch_server
from dsagt.observability import (
    obs,
    registry_install_deps_span,
    registry_reconstruct_pipeline_span,
    registry_save_code_span,
)
from dsagt.provenance import CodeUseIndexer, readiness_reports, reconstruct_pipeline
from dsagt.registry import CODES_COLLECTION, CodeRegistry

logger = logging.getLogger(__name__)


def _install_dependencies(packages: list[str], timeout: int = 120) -> str:
    """Install packages using uv pip install. Returns a status string."""
    cmd = ["uv", "pip", "install", "--python", sys.executable] + packages
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
        return (
            "Error: 'uv' command not found. Install uv: https://github.com/astral-sh/uv"
        )


# ---------------------------------------------------------------------------
# Per-tool handlers (module-level, explicit dependencies)
# ---------------------------------------------------------------------------


def _code_for_same_script(registry: CodeRegistry, spec: dict) -> dict | None:
    """A registered code, under another name, whose executable runs the same
    script file as *spec*; a skill's scripts are registered when the skill
    is saved, and a second registration of one script gives two codes."""
    import shlex

    def script_of(executable: str) -> str | None:
        tokens = shlex.split(executable.split(" -- ")[-1]) if executable else []
        return next((t for t in tokens if t.endswith((".py", ".sh"))), None)

    target = script_of(spec.get("executable", ""))
    if target is None:
        return None
    for code in registry.list_codes_raw():
        if (
            code.get("name") != spec.get("name")
            and script_of(code.get("executable", "")) == target
        ):
            return code
    return None


async def _handle_save_code_spec(
    arguments: dict,
    *,
    registry: CodeRegistry,
) -> str:
    spec = arguments["spec"]
    # Some MCP clients (notably Claude Sonnet/Haiku 4.x) serialize nested
    # object args as JSON strings instead of objects.  Accept both shapes.
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            return f"Error: spec must be a JSON object (or string-encoded JSON object): {e}"
    with registry_save_code_span(spec.get("name")):
        obs.set("language", spec.get("language"))
        obs.set("n_dependencies", len(spec.get("dependencies") or []))
        obs.set("n_tags", len(spec.get("tags") or []))
        try:
            skill_md = registry.codes_dir / str(spec.get("name")) / "SKILL.md"
            if skill_md.exists() and registry.get_code(spec["name"]) is None:
                codes = [
                    c["name"]
                    for c in registry.list_codes_raw()
                    if spec["name"] in (c.get("tags") or [])
                ]
                return (
                    f"'{spec['name']}' is an installed skill, and a code of that name "
                    "would replace its frontmatter. Its scripts are registered as: "
                    f"{', '.join(codes) or 'none'}. Use one of those, or choose "
                    "another name."
                )
            existing = _code_for_same_script(registry, spec)
            if existing is not None:
                return (
                    f"'{spec['name']}' was not saved: {existing['name']} already "
                    f"runs {spec.get('executable', '')!r}. Run it as: "
                    f"{existing['executable']}\nTo give it parameter roles or "
                    f"dependencies, save this spec again with the name "
                    f"'{existing['name']}'."
                )
            action = registry.save_tool(spec)
        except (KeyError, ValueError, OSError) as e:
            obs.event("save_tool_failed", error=str(e)[:256])
            return f"Error saving tool spec: {e}"

        # Codes share the skill envelope, so a fresh registration mirrors into
        # the agent's native skills dir right away (next session discovers it).
        from dsagt.agents import refresh_native_skills

        refresh_native_skills(registry.runtime_dir)
        tool_count = len(registry.list_codes_raw())
        obs.set("action", action)
        obs.set("registry_size", tool_count)
        stored = registry.get_code(spec["name"])
        message = (
            f"Tool '{spec['name']}' {action} successfully. "
            f"Registry now contains {tool_count} tools.\n"
            f"Run it as: {stored['executable']}\n"
            "The dsagt-run prefix writes the execution record; run this line, "
            "not the command you supplied."
        )
        deps = spec.get("dependencies", [])
        if deps:
            with registry_install_deps_span(deps):
                dep_result = await asyncio.to_thread(_install_dependencies, deps)
                if dep_result.startswith("Successfully installed:"):
                    obs.set("status", "ok")
                else:
                    obs.set("status", "failed")
                    obs.event("install_failed", message=dep_result[:256])
            message += f"\n\nDependency installation:\n{dep_result}"
        return message


async def _handle_get_registry(
    arguments: dict,
    *,
    registry: CodeRegistry,
) -> str:
    tools = registry.list_codes_raw()
    if not tools:
        return "Registry is empty. No tools registered yet."
    return yaml.dump({"codes": tools}, default_flow_style=False, sort_keys=False)


async def _handle_search_registry(
    arguments: dict,
    *,
    registry: CodeRegistry,
    kb: KnowledgeBase | None,
) -> str:
    code_name = arguments.get("code_name")
    query = arguments.get("query", "")
    tag = arguments.get("tag")
    top_k = arguments.get("top_k", 10)

    if code_name:
        tool = registry.get_code(code_name)
        if tool:
            return f"Found tool '{code_name}':\n\n" + yaml.dump(
                tool, default_flow_style=False, sort_keys=False
            )
        return f"No tool named '{code_name}'."

    if kb is None:
        return (
            "search_registry requires a configured knowledge base "
            "(set embedding.api_key + embedding.base_url + embedding.model "
            "in .dsagt/config.yaml).  Use search_registry with an exact "
            "code_name for KB-free lookups."
        )

    # One ``codes`` collection holds bundled and registered entries,
    # distinguished by ``metadata.source``.
    results = await asyncio.to_thread(
        kb.search,
        query=query or "tool",
        collection=CODES_COLLECTION,
        top_k=top_k * 3 if tag else top_k,
    )
    if tag and results:
        results = [
            r
            for r in results
            if tag in r.get("chunk", {}).get("metadata", {}).get("tags", "")
        ][:top_k]
    if not results:
        return "No tools found matching the query."

    # The rank is the information; a rank-fusion score is a sum of 1/(60+rank)
    # terms and reads as the same two decimals for every hit.
    summaries = []
    for rank, r in enumerate(results, start=1):
        chunk = r.get("chunk", {})
        meta = chunk.get("metadata", {})
        summaries.append(
            f"{rank}. **{meta.get('code_name', 'unknown')}**\n"
            f"   {chunk.get('text', '')[:200]}"
        )
    return f"Found {len(results)} code(s), best first:\n\n" + "\n\n".join(summaries)


async def _handle_readiness_reports(arguments: dict, *, runtime_dir: Path) -> dict:
    path = arguments["path"]
    reports = await asyncio.to_thread(readiness_reports, runtime_dir, path)
    return {"path": path, "reports": reports}


async def _handle_reconstruct_pipeline(
    arguments: dict,
    *,
    runtime_dir: Path,
    kb: KnowledgeBase | None = None,
) -> str:
    fmt = arguments.get("format", "bash")
    output = arguments.get("output")
    trace_dir = runtime_dir / "trace_archive"
    # Index the session's tool-use first: reconstruct is the moment the pipeline
    # is complete enough to review, so the just-run executions are made
    # searchable here rather than waiting on the periodic pass.  Idempotent and
    # file-locked, so this is safe to run beside the periodic pass's own
    # CodeUseIndexer.
    if kb is not None:
        try:
            await asyncio.to_thread(CodeUseIndexer(kb, runtime_dir).tick)
        except Exception as e:  # noqa: BLE001  indexing is best-effort here
            logger.warning("code_use indexing before reconstruct failed: %s", e)
    with registry_reconstruct_pipeline_span(fmt):
        try:
            script = await asyncio.to_thread(reconstruct_pipeline, trace_dir, fmt=fmt)
        except (FileNotFoundError, ValueError, OSError) as e:
            obs.event("reconstruct_failed", error=str(e)[:256])
            return f"Error reconstructing pipeline: {e}"
        obs.set("output_chars", len(script))
        if output:
            # A project-relative path; the agent otherwise copies the script
            # by hand from the reply, which one run did from its own
            # transcript after a context compaction.
            target = runtime_dir / output
            if not target.resolve().is_relative_to(runtime_dir.resolve()):
                return f"Error: output must be a path under the project, got {output!r}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(script)
            return f"Saved to {output}\n\n{script}"
        return script


async def _handle_install_dependencies(
    arguments: dict,
    *,
    registry: CodeRegistry,
) -> str:
    code_name = arguments.get("code_name")
    tools = registry.list_codes_raw()
    if not tools:
        return "Registry is empty. No tools registered yet."

    all_deps = []
    codes_with_deps = []
    for tool in tools:
        if code_name and tool.get("name") != code_name:
            continue
        code_deps = tool.get("dependencies", [])
        if code_deps:
            all_deps.extend(code_deps)
            codes_with_deps.append(tool["name"])

    if not all_deps:
        scope = f"tool '{code_name}'" if code_name else "registry"
        return f"No dependencies declared in {scope}."

    seen = set()
    unique_deps = [d for d in all_deps if not (d in seen or seen.add(d))]

    with registry_install_deps_span(unique_deps):
        obs.set("scope_code", code_name)
        obs.set("n_tools_with_deps", len(codes_with_deps))
        result = await asyncio.to_thread(_install_dependencies, unique_deps)
        if result.startswith("Successfully installed:"):
            obs.set("status", "ok")
        else:
            obs.set("status", "failed")
            obs.event("install_failed", message=result[:256])
        return f"Installing dependencies for: {', '.join(codes_with_deps)}\n\n{result}"


# ---------------------------------------------------------------------------
# Tool defs and handler map (used by the merged server and the test wrapper)
# ---------------------------------------------------------------------------


def _registry_tools_and_handlers(
    registry: CodeRegistry,
    kb: KnowledgeBase | None = None,
):
    """Build the registry/execution/provenance ``(tool defs, handler map)``.

    Combined with the other concern modules' tools under one MCP ``Server`` by
    :func:`dsagt.mcp.server.create_dsagt_server`.
    """
    runtime_dir = Path(registry.runtime_dir)

    handlers = {
        "save_code_spec": partial(_handle_save_code_spec, registry=registry),
        "get_registry": partial(_handle_get_registry, registry=registry),
        "search_registry": partial(_handle_search_registry, registry=registry, kb=kb),
        "reconstruct_pipeline": partial(
            _handle_reconstruct_pipeline, runtime_dir=runtime_dir, kb=kb
        ),
        "readiness_reports": partial(
            _handle_readiness_reports, runtime_dir=runtime_dir
        ),
        "install_dependencies": partial(
            _handle_install_dependencies, registry=registry
        ),
    }

    tools = [
        types.Tool(
            name="save_code_spec",
            description=(
                "Register a CLI code: writes a skill-standard spec dir "
                "(skills/<name>/SKILL.md), mirrored into the agent's native "
                "skills dir immediately so future sessions auto-discover it"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    # ``anyOf`` accepts both a structured object and a
                    # JSON-encoded string.  Some MCP clients (notably
                    # Claude Sonnet/Haiku 4.x) serialize nested object
                    # arguments as JSON strings instead of objects; the
                    # handler unwraps either shape.
                    "spec": {
                        "description": "Tool specification (object or JSON-encoded string)",
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": (
                                            "Unique code name: lowercase "
                                            "letters, digits, hyphens (e.g. "
                                            "'datacard-introspect')"
                                        ),
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": (
                                            "What the code does and when to "
                                            "use it, phrased as 'Use when "
                                            "…' so native skill routing can "
                                            "match it"
                                        ),
                                    },
                                    "executable": {
                                        "type": "string",
                                        "description": (
                                            "The bare command, for example "
                                            "'python skills/x/scripts/x.py'. "
                                            "The registry prepends "
                                            "'dsagt-run --code <name> --' and, "
                                            "when dependencies are declared, "
                                            "'uv run --with <deps> --'; the "
                                            "reply returns the stored line to "
                                            "run."
                                        ),
                                    },
                                    "parameters": {
                                        "type": "object",
                                        "description": "Parameter definitions",
                                        "additionalProperties": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "description": "Parameter type",
                                                },
                                                "required": {"type": "boolean"},
                                                "description": {"type": "string"},
                                                "default": {
                                                    "description": "Default value"
                                                },
                                                "cli": {
                                                    "type": "string",
                                                    "description": (
                                                        "How to render this parameter on the command line: "
                                                        "'positional[:N]' for positional args, '--name' or '-n' "
                                                        "for spaced flags, '--name=' or '-n=' for glued flags, "
                                                        "'key=' for dd-style key=value. Defaults to '--<param_name>' "
                                                        "if omitted."
                                                    ),
                                                },
                                                "role": {
                                                    "type": "string",
                                                    "enum": ["input", "output"],
                                                    "description": (
                                                        "Set on a parameter whose value is a file the code "
                                                        "reads ('input') or writes ('output'). dsagt-run "
                                                        "records those files on every run, which is what "
                                                        "reconstruct_pipeline's dependency graph reads."
                                                    ),
                                                },
                                            },
                                            "required": ["type", "description"],
                                        },
                                    },
                                    "dependencies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Python packages to install",
                                    },
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Tags for categorizing the tool",
                                    },
                                },
                                "required": [
                                    "name",
                                    "description",
                                    "executable",
                                    "parameters",
                                ],
                            },
                            {"type": "string"},
                        ],
                    },
                },
                "required": ["spec"],
            },
        ),
        types.Tool(
            name="get_registry",
            description="Get all tools from the registry",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="search_registry",
            description="Search for tools by name, tag, or description via semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "tag": {"type": "string", "description": "Filter by tag"},
                    "code_name": {
                        "type": "string",
                        "description": "Exact tool name lookup",
                    },
                    "top_k": {"type": "integer", "default": 10},
                },
            },
        ),
        types.Tool(
            name="reconstruct_pipeline",
            description=(
                "Reconstruct a reproducible pipeline script from the execution "
                "records in trace_archive/, in the order they ran; a run that "
                "exited non-zero is kept as a comment, and paths under the "
                "project are relative to it. The script calls each recorded tool "
                "directly, without the dsagt-run wrapper, so it runs outside a "
                "DSAgt project; it creates the recorded output directories first "
                "and writes a recorded stdout file with a redirect. Present it to "
                "the user as returned, and pass output to save it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["bash", "snakemake"],
                        "default": "bash",
                    },
                    "output": {
                        "type": "string",
                        "description": "Project-relative path to save the script "
                        "to, e.g. audit/pipeline.sh; the reply says where it went.",
                    },
                },
            },
        ),
        types.Tool(
            name="readiness_reports",
            description=(
                "The AI-readiness reports on record for a file, newest first: "
                "each aidrin run whose input was the file, with its report path, "
                "start time, and whether the file's content is unchanged since "
                "that run. Call it before running a check; a current report is "
                "the pre report of the next stage."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file, relative to the project directory",
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="install_dependencies",
            description="Install Python dependencies for one or all tools in the registry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_name": {
                        "type": "string",
                        "description": "Install deps for a specific tool (omit for all)",
                    },
                },
            },
        ),
    ]
    return tools, handlers


def create_registry_server(
    registry: CodeRegistry,
    kb: KnowledgeBase | None = None,
):
    """Create a standalone MCP server exposing only the registry and provenance tools.

    Test-facing API: tests call with a mock registry and drive the server via
    ``call_tool_sync()``.  The merged ``dsagt-server`` composes
    :func:`_registry_tools_and_handlers` itself.
    """
    tools, handlers = _registry_tools_and_handlers(registry, kb)
    return build_dispatch_server(
        "registry", tools, handlers, {t: "registry" for t in handlers}
    )
