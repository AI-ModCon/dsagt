"""MCP tools for the tool registry, execution, and provenance.

The "tool lifecycle" surface of ``dsagt-server``: define a tool spec
(``save_code_spec``), discover tools (``get_registry`` / ``search_registry``),
execute / gather (``read_file`` / ``http_request`` / ``run_command`` /
``install_dependencies``), and reconstruct a reproducible pipeline from the
recorded executions (``reconstruct_pipeline``).

Tool specs are saved as markdown files in the runtime tools directory and
indexed into a ChromaDB collection for semantic search.  Server configuration
(embedding credentials) flows through env vars (LLM_API_KEY, OPENAI_BASE_URL,
EMBEDDING_MODEL) set by ``dsagt start``.

These definitions + handlers run inside the merged ``dsagt-server`` (see
:mod:`dsagt.mcp.server`); ``create_registry_server`` is retained only as a
test-facing constructor.  Skill tools (``save_skill`` / ``search_skills`` /
``install_skill``) live in :mod:`dsagt.mcp.skill_tools`.
"""

import json
import logging
import subprocess
import sys
from functools import partial
from pathlib import Path

import httpx
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
from dsagt.provenance import CodeUseIndexer, reconstruct_pipeline
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
            action = registry.save_tool(spec)
        except (KeyError, ValueError, OSError) as e:
            obs.event("save_tool_failed", error=str(e)[:256])
            return f"Error saving tool spec: {e}"

        tool_count = len(registry.list_codes_raw())
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

    # Single ``tools`` collection — bundled and registered entries
    # coexist, distinguished by ``metadata.source`` if needed.
    results = kb.search(
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

    summaries = []
    for r in results:
        chunk = r.get("chunk", {})
        meta = chunk.get("metadata", {})
        summaries.append(
            f"- **{meta.get('code_name', 'unknown')}** "
            f"(score: {r.get('score', 0):.2f})\n"
            f"  {chunk.get('text', '')[:200]}"
        )
    return f"Found {len(results)} tool(s):\n\n" + "\n\n".join(summaries)


async def _handle_reconstruct_pipeline(
    arguments: dict,
    *,
    runtime_dir: Path,
    kb: KnowledgeBase | None = None,
) -> str:
    fmt = arguments.get("format", "bash")
    trace_dir = runtime_dir / "trace_archive"
    # Index the session's tool-use first: reconstruct is the moment the pipeline
    # is "done enough" to review, so make the just-run executions searchable now
    # rather than waiting on the heartbeat.  Idempotent + file-locked, so this
    # is safe to fire alongside the heartbeat's own CodeUseIndexer.
    if kb is not None:
        try:
            CodeUseIndexer(kb, runtime_dir).tick()
        except Exception as e:  # noqa: BLE001 — indexing is best-effort here
            logger.warning("tool_use indexing before reconstruct failed: %s", e)
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
        result = _install_dependencies(unique_deps)
        if result.startswith("Successfully installed:"):
            obs.set("status", "ok")
        else:
            obs.set("status", "failed")
            obs.event("install_failed", message=result[:256])
        return f"Installing dependencies for: {', '.join(codes_with_deps)}\n\n{result}"


# ---------------------------------------------------------------------------
# Tool defs + handler map (used by the merged server and the test wrapper)
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
        "read_file": _handle_read_file,
        "http_request": _handle_http_request,
        "run_command": _handle_run_command,
        "save_code_spec": partial(_handle_save_code_spec, registry=registry),
        "get_registry": partial(_handle_get_registry, registry=registry),
        "search_registry": partial(_handle_search_registry, registry=registry, kb=kb),
        "reconstruct_pipeline": partial(
            _handle_reconstruct_pipeline, runtime_dir=runtime_dir, kb=kb
        ),
        "install_dependencies": partial(
            _handle_install_dependencies, registry=registry
        ),
    }

    tools = [
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
            name="save_code_spec",
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
    """Create a standalone MCP server exposing only the registry/exec/provenance tools.

    Test-facing API: tests call with a mock registry and drive the server via
    ``call_tool_sync()``.  The merged ``dsagt-server`` uses
    :func:`_registry_tools_and_handlers` directly instead of this wrapper.
    """
    tools, handlers = _registry_tools_and_handlers(registry, kb)
    return build_dispatch_server(
        "registry", tools, handlers, {t: "registry" for t in handlers}
    )
