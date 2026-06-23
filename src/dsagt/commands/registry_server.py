"""
DSAgt Registry MCP Server.

Provides tools for building a tool registry by reading documentation,
fetching web resources, and running commands to extract tool specifications.

Tool specs are saved as skill markdown files in the runtime skills directory
and indexed into a ChromaDB collection for semantic search.

Server configuration (embedding credentials) flows through env vars
(LLM_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL) set by dsagt start.

Usage:
    dsagt-registry-server --runtime-dir ./my_session
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import httpx
import yaml

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from dsagt.knowledge import KnowledgeBase
from dsagt.observability import (
    obs,
    registry_install_deps_span,
    registry_reconstruct_pipeline_span,
    registry_save_tool_span,
)
from dsagt.provenance import reconstruct_pipeline
from dsagt.registry import (
    CATALOG_COLLECTION_PREFIX,
    SKILLS_COLLECTION,
    TOOLS_COLLECTION,
    SkillRegistry,
    ToolRegistry,
    _parse_frontmatter,
)

os.environ["PYTHONUNBUFFERED"] = "1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP server helpers
# ---------------------------------------------------------------------------


async def _run_stdio(server: Server, name: str):
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=name,
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


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


async def _handle_read_file(arguments: dict) -> str:
    path = Path(arguments["path"])
    try:
        return path.read_text()
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        OSError,
        UnicodeDecodeError,
    ) as e:
        return f"Error reading file: {e}"


async def _handle_http_request(arguments: dict) -> str:
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
            return f"Status: {response.status_code}\n\n{response.text}"
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        return f"Error making request: {e}"


async def _handle_run_command(arguments: dict) -> str:
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
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except FileNotFoundError:
        return f"Command '{command}' not found"

    output = ""
    if result.stdout:
        output += f"STDOUT:\n{result.stdout}\n"
    if result.stderr:
        output += f"STDERR:\n{result.stderr}\n"
    output += f"\nReturn code: {result.returncode}"
    return output


async def _handle_save_tool_spec(
    arguments: dict,
    *,
    registry: ToolRegistry,
) -> str:
    spec = arguments["spec"]
    # Some MCP clients (notably Claude Sonnet/Haiku 4.x) serialize nested
    # object args as JSON strings instead of objects.  Accept both shapes.
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            return f"Error: spec must be a JSON object (or string-encoded JSON object): {e}"
    with registry_save_tool_span(spec.get("name")):
        obs.set("language", spec.get("language"))
        obs.set("n_dependencies", len(spec.get("dependencies") or []))
        obs.set("n_tags", len(spec.get("tags") or []))
        try:
            action = registry.save_tool(spec)
        except (KeyError, ValueError, OSError) as e:
            obs.event("save_tool_failed", error=str(e)[:256])
            return f"Error saving tool spec: {e}"

        tool_count = len(registry.list_tools_raw())
        obs.set("action", action)
        obs.set("registry_size", tool_count)
        message = (
            f"Tool '{spec['name']}' {action} successfully. "
            f"Registry now contains {tool_count} tools."
        )
        deps = spec.get("dependencies", [])
        if deps:
            with registry_install_deps_span(deps):
                dep_result = _install_dependencies(deps)
                if dep_result.startswith("Successfully installed:"):
                    obs.set("status", "ok")
                else:
                    obs.set("status", "failed")
                    obs.event("install_failed", message=dep_result[:256])
            message += f"\n\nDependency installation:\n{dep_result}"
        return message


async def _handle_save_skill(
    arguments: dict,
    *,
    skill_registry: SkillRegistry,
) -> str:
    """Register a skill (workflow / agent instructions) for later reuse.

    Symmetric with save_tool_spec — writes SKILL.md to
    ``<project>/skills/<name>/`` and indexes it into
    ``registered_skills`` so future ``search_skills`` calls find it.
    """
    spec = arguments["spec"]
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            return f"Error: spec must be a JSON object (or string-encoded JSON object): {e}"
    body = arguments.get("body")
    reference_files = arguments.get("reference_files")
    if isinstance(reference_files, str):
        try:
            reference_files = json.loads(reference_files)
        except json.JSONDecodeError as e:
            return f"Error: reference_files must be a JSON object: {e}"
    try:
        action = skill_registry.save_skill(
            spec, body=body, reference_files=reference_files
        )
    except (KeyError, ValueError, OSError) as e:
        return f"Error saving skill: {e}"
    skill_count = len(skill_registry.list_skills())
    return (
        f"Skill '{spec['name']}' {action} successfully. "
        f"Registry now contains {skill_count} skills."
    )


async def _handle_get_registry(
    arguments: dict,
    *,
    registry: ToolRegistry,
) -> str:
    tools = registry.list_tools_raw()
    if not tools:
        return "Registry is empty. No tools registered yet."
    return yaml.dump({"tools": tools}, default_flow_style=False, sort_keys=False)


async def _handle_search_registry(
    arguments: dict,
    *,
    registry: ToolRegistry,
    kb: KnowledgeBase | None,
) -> str:
    tool_name = arguments.get("tool_name")
    query = arguments.get("query", "")
    tag = arguments.get("tag")
    top_k = arguments.get("top_k", 10)

    if tool_name:
        tool = registry.get_tool(tool_name)
        if tool:
            return f"Found tool '{tool_name}':\n\n" + yaml.dump(
                tool, default_flow_style=False, sort_keys=False
            )
        return f"No tool named '{tool_name}'."

    if kb is None:
        return (
            "search_registry requires a configured knowledge base "
            "(set embedding.api_key + embedding.base_url + embedding.model "
            "in dsagt_config.yaml).  Use search_registry with an exact "
            "tool_name for KB-free lookups."
        )

    # Single ``tools`` collection — bundled and registered entries
    # coexist, distinguished by ``metadata.source`` if needed.
    results = kb.search(
        query=query or "tool",
        collection=TOOLS_COLLECTION,
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

    summaries = []
    for r in results:
        chunk = r.get("chunk", {})
        meta = chunk.get("metadata", {})
        summaries.append(
            f"- **{meta.get('tool_name', 'unknown')}** "
            f"(score: {r.get('score', 0):.2f})\n"
            f"  {chunk.get('text', '')[:200]}"
        )
    return f"Found {len(results)} tool(s):\n\n" + "\n\n".join(summaries)


async def _handle_search_skills(
    arguments: dict,
    *,
    kb: KnowledgeBase | None,
    skill_registry: SkillRegistry | None,
) -> str:
    skill_name = arguments.get("skill_name")
    query = arguments.get("query", "")
    tag = arguments.get("tag")
    top_k = arguments.get("top_k", 10)

    if skill_name and skill_registry:
        skill = skill_registry.get_skill(skill_name)
        if skill:
            return f"Found skill '{skill_name}':\n\n" + yaml.dump(
                skill, default_flow_style=False, sort_keys=False
            )
        return f"No skill named '{skill_name}'."

    if kb is None:
        return (
            "search_skills requires a configured knowledge base "
            "(set embedding.api_key + embedding.base_url + embedding.model "
            "in dsagt_config.yaml).  Use search_skills with an exact "
            "skill_name for KB-free lookups."
        )

    # Search the installed/bundled ``skills`` collection AND every external
    # ``skills_catalog__<slug>`` collection, then merge by score.  Installed
    # skills are also natively discovered by the agent; the catalog is the
    # part native discovery can't do (it isn't loaded into context).
    catalog_collections = [
        c for c in kb.collections if c.startswith(CATALOG_COLLECTION_PREFIX)
    ]
    collections = [SKILLS_COLLECTION] + catalog_collections
    fetch_k = top_k * 3 if tag else top_k
    results: list[dict] = []
    for coll in collections:
        try:
            results.extend(
                kb.search(query=query or "skill", collection=coll, top_k=fetch_k)
            )
        except (FileNotFoundError, KeyError, ValueError):
            continue  # collection absent/empty on this KB — skip
    if tag:
        results = [
            r
            for r in results
            if tag in r.get("chunk", {}).get("metadata", {}).get("tags", "")
        ]
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    results = results[:top_k]
    if not results:
        if not catalog_collections:
            return (
                "No skills found matching the query. Note: no external skill "
                "catalog is synced yet, so only installed skills were searched. "
                "Call list_skill_sources() to see available sources, then "
                "add_skill_source(source=...) (e.g. 'scientific' for "
                "materials/chem/bio) to sync one before searching again."
            )
        return "No skills found matching the query."

    summaries = []
    for r in results:
        chunk = r.get("chunk", {})
        meta = chunk.get("metadata", {})
        src = meta.get("source", "")
        where = (
            " [installed]"
            if src in ("bundled", "registered")
            else (
                " [catalog · install_skill to add]"
                if src.startswith("catalog:")
                else ""
            )
        )
        summaries.append(
            f"- **{meta.get('skill_name', 'unknown')}**{where} "
            f"(score: {r.get('score', 0):.2f})\n"
            f"  {chunk.get('text', '')[:200]}"
        )
    return f"Found {len(results)} skill(s):\n\n" + "\n\n".join(summaries)


async def _handle_install_skill(
    arguments: dict,
    *,
    skill_registry: SkillRegistry | None,
    runtime_dir: Path,
) -> str:
    """Install a catalog skill into ``<project>/skills/<name>/``.

    The skill's files land on disk immediately, so the agent can use it in the
    current session by reading its SKILL.md.  *Native* auto-invocation requires
    the next ``dsagt start`` (which mirrors installed skills into
    ``.claude/skills/`` before launch) plus an agent restart.
    """
    from dsagt.commands.skills_catalog import install_into_project

    name = arguments.get("skill_name")
    if not name:
        return "install_skill requires 'skill_name'."
    try:
        info = install_into_project(name, runtime_dir)
    except LookupError as e:
        return f"Error: {e}"

    # Index the now-installed skill as a project ('registered') skill too, so
    # non-native agents can still find it via search_skills after install.
    if skill_registry is not None and skill_registry._kb is not None:
        skill_md = Path(info["dest_dir"]) / "SKILL.md"
        spec = _parse_frontmatter(skill_md)
        if spec.get("name"):
            skill_registry._index_skill(spec, skill_md)

    return (
        f"{info['action'].capitalize()} skill '{info['name']}' at "
        f"{info['dest_dir']}.\n\n"
        f"Usable now — its SKILL.md and any scripts/references are already on "
        f"disk in this project. To use it this session, read "
        f"{info['dest_dir']}/SKILL.md and follow it; you don't need to restart.\n"
        f"Restart is only for hands-free auto-invocation: the next `dsagt start` "
        f"mirrors it into the platform's native skill dir (.claude/skills/), and "
        f"after relaunch the agent discovers and auto-invokes it without this tool."
    )


async def _handle_reconstruct_pipeline(
    arguments: dict,
    *,
    runtime_dir: Path,
) -> str:
    fmt = arguments.get("format", "bash")
    trace_dir = runtime_dir / "trace_archive"
    with registry_reconstruct_pipeline_span(fmt):
        try:
            script = reconstruct_pipeline(trace_dir, fmt=fmt)
        except (FileNotFoundError, ValueError, OSError) as e:
            obs.event("reconstruct_failed", error=str(e)[:256])
            return f"Error reconstructing pipeline: {e}"
        obs.set("output_chars", len(script))
        return script


async def _handle_install_dependencies(
    arguments: dict,
    *,
    registry: ToolRegistry,
) -> str:
    tool_name = arguments.get("tool_name")
    tools = registry.list_tools_raw()
    if not tools:
        return "Registry is empty. No tools registered yet."

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
        return f"No dependencies declared in {scope}."

    seen = set()
    unique_deps = [d for d in all_deps if not (d in seen or seen.add(d))]

    with registry_install_deps_span(unique_deps):
        obs.set("scope_tool", tool_name)
        obs.set("n_tools_with_deps", len(tools_with_deps))
        result = _install_dependencies(unique_deps)
        if result.startswith("Successfully installed:"):
            obs.set("status", "ok")
        else:
            obs.set("status", "failed")
            obs.event("install_failed", message=result[:256])
        return f"Installing dependencies for: {', '.join(tools_with_deps)}\n\n{result}"


# ---------------------------------------------------------------------------
# Server factory (thin wiring — used by main() and tests)
# ---------------------------------------------------------------------------


def create_registry_server(
    registry: ToolRegistry,
    kb: KnowledgeBase | None = None,
    skill_registry: SkillRegistry | None = None,
):
    """Create and configure the MCP registry server.

    Test-facing API: tests call with a mock registry and get back a server
    they can drive via call_tool_sync().  main() constructs the registry
    and KB from config before calling this.
    """
    server = Server("registry")
    runtime_dir = Path(registry.runtime_dir)

    # Dispatch table — maps MCP tool names to handler functions with
    # dependencies bound via functools.partial.
    handlers = {
        "read_file": _handle_read_file,
        "http_request": _handle_http_request,
        "run_command": _handle_run_command,
        "save_tool_spec": partial(_handle_save_tool_spec, registry=registry),
        "save_skill": partial(_handle_save_skill, skill_registry=skill_registry),
        "get_registry": partial(_handle_get_registry, registry=registry),
        "search_registry": partial(_handle_search_registry, registry=registry, kb=kb),
        "search_skills": partial(
            _handle_search_skills, kb=kb, skill_registry=skill_registry
        ),
        "install_skill": partial(
            _handle_install_skill,
            skill_registry=skill_registry,
            runtime_dir=runtime_dir,
        ),
        "reconstruct_pipeline": partial(
            _handle_reconstruct_pipeline, runtime_dir=runtime_dir
        ),
        "install_dependencies": partial(
            _handle_install_dependencies, registry=registry
        ),
    }

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="read_file",
                description="Read contents of a text file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
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
                        "method": {
                            "type": "string",
                            "description": "HTTP method",
                            "default": "GET",
                        },
                        "headers": {
                            "type": "object",
                            "description": "Optional headers",
                        },
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
                        "command": {
                            "type": "string",
                            "description": "Command to execute",
                        },
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
                                            "description": "Unique tool identifier",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "What the tool does",
                                        },
                                        "executable": {
                                            "type": "string",
                                            "description": "Command to execute",
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
                name="save_skill",
                description=(
                    "Register a skill (agent workflow / instructions) into "
                    "<project>/skills/<name>/SKILL.md and index it into the "
                    "registered_skills KB collection.  Symmetric with "
                    "save_tool_spec — use this when you've designed a "
                    "reusable instruction set you want future sessions to "
                    "discover via search_skills."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        # ``anyOf`` for spec mirrors save_tool_spec — accept
                        # both structured object and JSON-encoded string for
                        # MCP clients that serialize nested args.
                        "spec": {
                            "description": "Skill spec (object or JSON-encoded string)",
                            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Unique skill identifier (becomes the directory name)",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "What the skill does / when to use it",
                                        },
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags for categorizing the skill",
                                        },
                                    },
                                    "required": ["name", "description"],
                                },
                                {"type": "string"},
                            ],
                        },
                        "body": {
                            "type": "string",
                            "description": (
                                "Markdown body of the SKILL.md (workflow / "
                                "instructions the agent will follow).  When "
                                "updating an existing skill, omit to preserve "
                                "the existing body."
                            ),
                        },
                        "reference_files": {
                            "description": (
                                "Optional additional files to write into the "
                                "skill directory.  Object mapping relative "
                                "path -> file contents, or JSON-encoded string."
                            ),
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
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
                        "tool_name": {
                            "type": "string",
                            "description": "Exact tool name lookup",
                        },
                        "top_k": {"type": "integer", "default": 10},
                    },
                },
            ),
            types.Tool(
                name="search_skills",
                description=(
                    "Search agent skills by name, tag, or description. Spans installed "
                    "skills and the external installable catalog. Catalog hits are marked "
                    "'[catalog]' — use install_skill to add one to this project."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "tag": {"type": "string", "description": "Filter by tag"},
                        "skill_name": {
                            "type": "string",
                            "description": "Exact skill name lookup",
                        },
                        "top_k": {"type": "integer", "default": 10},
                    },
                },
            ),
            types.Tool(
                name="install_skill",
                description=(
                    "Install a skill from the external catalog (found via search_skills) "
                    "into this project so the agent can use it natively. Copies SKILL.md "
                    "+ scripts/references; available natively after the next restart."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Catalog skill name to install",
                        },
                    },
                    "required": ["skill_name"],
                },
            ),
            types.Tool(
                name="reconstruct_pipeline",
                description="Reconstruct a reproducible pipeline script from tool execution records.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["bash", "snakemake"],
                            "default": "bash",
                        },
                    },
                },
            ),
            types.Tool(
                name="install_dependencies",
                description="Install Python dependencies for one or all tools in the registry.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Install deps for a specific tool (omit for all)",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        handler = handlers[name]  # KeyError = bug in list_tools schema
        text = await handler(arguments)
        return [types.TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for dsagt-registry-server.

    All configuration comes from the project directory:
    - ``./dsagt_config.yaml`` → project path + name
    - ``LLM_API_KEY``, ``OPENAI_BASE_URL`` env vars → embedding credentials

    No CLI arguments.  By contract the agent's launch one-liner is
    ``cd <pdir> && <agent>``, so cwd is project_dir for the MCP
    children it spawns.
    """
    import logging as _logging
    from dsagt.observability import find_project_config

    project_dir, _cfg = find_project_config()
    if project_dir is None:
        raise RuntimeError(
            "dsagt-registry-server: no dsagt_config.yaml in cwd "
            f"({Path.cwd()}).  Launch the agent from the project "
            "directory (`cd <pdir> && <agent>`)."
        )

    log_file = project_dir / "dsagt_registry_server.log"
    # Default INFO; users opt into DEBUG via DSAGT_LOG_LEVEL=DEBUG.  At DEBUG,
    # transitive libraries (httpcore, urllib3, llama_index, chromadb) flood
    # stderr with one line per network operation — when an agent like roo
    # pipes the MCP server's stderr into its own debug stream, the human
    # output gets buried under thousands of low-value lines.
    _level_name = os.environ.get("DSAGT_LOG_LEVEL", "INFO").upper()
    _level = getattr(_logging, _level_name, _logging.INFO)
    _logging.basicConfig(
        level=_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            _logging.FileHandler(log_file, mode="a"),
            _logging.StreamHandler(),
        ],
    )
    log = _logging.getLogger(__name__)
    log.info("Server starting — log file: %s", log_file)

    config_path = project_dir / "dsagt_config.yaml"
    from dsagt.session import resolve_env_vars

    config = resolve_env_vars(yaml.safe_load(config_path.read_text()))

    emb_config = config["embedding"]

    from dsagt.observability import init_tracing, configure_litellm_retries

    init_tracing(
        "dsagt-registry-server"
    )  # session_id picked up from DSAGT_SESSION_ID env
    configure_litellm_retries()

    # The KB is optional for the registry server — most tools (save_tool_spec,
    # get_registry, read_file, run_command, etc.) work without it.  Only
    # search_registry and search_skills need the KB for semantic search;
    # they return a clear error if the KB is None.
    backend = (emb_config.get("backend") or "local").lower()
    # Cross-backend leakage guard: see the matching block in
    # knowledge_server.py for the rationale.  In short: when
    # ``embedding.backend`` is ``local`` but the resolved model name is
    # an OpenAI-style alias (no slash), drop the override so we fall
    # back to the LocalEmbeddingClient default rather than 404 from HF.
    raw_model = (emb_config.get("model") or "").strip()
    embedder_kwargs: dict = {}
    if raw_model and not raw_model.startswith("${"):
        looks_hf = "/" in raw_model
        if backend == "local" and not looks_hf:
            log.warning(
                "Ignoring embedding.model=%r for backend=local (does not "
                "look like a HuggingFace identifier).  Falling back to the "
                "LocalEmbeddingClient default.",
                raw_model,
            )
        else:
            embedder_kwargs["model"] = raw_model
    if backend == "local":
        kb_available = True
    else:  # backend == "api"
        api_key = emb_config.get("api_key") or ""
        kb_available = (
            api_key and not api_key.startswith("${") and emb_config.get("base_url")
        )
        embedder_kwargs.update(
            {
                "base_url": emb_config.get("base_url") or "",
                "api_key": api_key,
            }
        )

    kb = None
    if kb_available:
        kb = KnowledgeBase(
            index_dir=project_dir / "kb_index",
            default_embedder=backend,
            default_index=config["knowledge"]["vector_db"],
            embedder_kwargs=embedder_kwargs,
        )
        # Background-load the embedder so the model is ready when the
        # agent's first search_registry / save_tool_spec call lands.
        # See knowledge_server for the same rationale.
        kb.preload_default_embedder()

    registry = ToolRegistry(
        source_tools_dir=None,
        runtime_dir=str(project_dir),
        kb=kb,
    )

    skill_reg = SkillRegistry(
        source_skills_dir=None,
        runtime_dir=str(project_dir),
        kb=kb,
    )

    # Bundled tools/skills are pre-embedded in the shared
    # ~/dsagt-projects/kb_index/ by ``dsagt setup-kb`` (or by the auto-bootstrap
    # in ``dsagt start``) and COPIED into the project's kb_index by
    # ``setup_runtime_kb`` before either MCP server spawns.  This server
    # does no embedding work for bundled content at startup; agent's
    # save_tool_spec / save_skill incur a single embed at save time.

    server = create_registry_server(registry, kb, skill_reg)
    asyncio.run(_run_stdio(server, "registry"))


if __name__ == "__main__":
    main()
