"""
DSAgt Registry MCP Server.

Provides tools for building a tool registry by reading documentation,
fetching web resources, and running commands to extract tool specifications.

Tool specs are saved as skill markdown files in the runtime skills directory
and indexed into a ChromaDB collection for semantic search.

Usage:
    dsagt-registry-server
    dsagt-registry-server --runtime-dir ./my_session
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

import json as _json
import os as _os

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

_os.environ["PYTHONUNBUFFERED"] = "1"


def _create_server(name: str) -> Server:
    return Server(name)


async def _run_stdio(server: Server, name: str):
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name=name, server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def _text_result(data) -> list[types.TextContent]:
    text = data if isinstance(data, str) else _json.dumps(data, indent=2)
    return [types.TextContent(type="text", text=text)]
from dsagt.registry import (
    SKILL_REGISTRY_COLLECTION,
    TOOL_REGISTRY_COLLECTION,
    SkillRegistry,
    ToolRegistry,
)


def _install_dependencies(packages: list[str], timeout: int = 120) -> str:
    """Install packages using uv pip install. Returns a status string."""
    cmd = ["uv", "pip", "install", "--python", sys.executable] + packages
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
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
        return "Error: 'uv' command not found. Install uv: https://github.com/astral-sh/uv"


def create_registry_server(
    registry: ToolRegistry,
    kb: KnowledgeBase | None = None,
    skill_registry: SkillRegistry | None = None,
):
    """Create and configure the MCP server."""
    server = _create_server("registry")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="read_file",
                description="Read contents of a text file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
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
                        "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                        "headers": {"type": "object", "description": "Optional headers"},
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
                        "command": {"type": "string", "description": "Command to execute"},
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
                        "spec": {
                            "type": "object",
                            "description": "Tool specification",
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
                                    "description": "Command to execute (e.g., 'python script.py')",
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Parameter definitions keyed by parameter name",
                                    "additionalProperties": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "description": "Parameter type (string, integer, number, boolean, array, object)",
                                            },
                                            "required": {
                                                "type": "boolean",
                                                "description": "Whether this parameter is required",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Parameter description",
                                            },
                                            "default": {
                                                "description": "Default value if not provided",
                                            },
                                        },
                                        "required": ["type", "description"],
                                    },
                                },
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Python packages to install (e.g., ['pandas>=2.0', 'numpy'])",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Tags for categorizing the tool (e.g., ['data_processing', 'genomics'])",
                                },
                            },
                            "required": ["name", "description", "executable", "parameters"],
                        },
                    },
                    "required": ["spec"],
                },
            ),
            types.Tool(
                name="get_registry",
                description="Get all tools from the registry",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="search_registry",
                description="Search for tools by name, tag, or description. Uses semantic search to find tools even when exact keywords don't match.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (natural language)"},
                        "tag": {"type": "string", "description": "Filter by tag (e.g., data_processing, genomics)"},
                        "tool_name": {"type": "string", "description": "Exact tool name lookup"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)", "default": 10},
                    },
                },
            ),
            types.Tool(
                name="search_skills",
                description="Search for agent skills (workflows, templates, procedures) by name, tag, or description. Skills are instruction sets the agent follows, not CLI tools.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (natural language)"},
                        "tag": {"type": "string", "description": "Filter by tag"},
                        "skill_name": {"type": "string", "description": "Exact skill name lookup"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)", "default": 10},
                    },
                },
            ),
            types.Tool(
                name="reconstruct_pipeline",
                description="Reconstruct a reproducible pipeline script from tool execution records in the trace archive. Returns a bash script or Snakemake workflow showing the sequence of tool calls with dependencies.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["bash", "snakemake"],
                            "description": "Output format (default: bash)",
                            "default": "bash",
                        },
                    },
                },
            ),
            types.Tool(
                name="install_dependencies",
                description="Install Python dependencies for one or all tools in the registry using uv pip install",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Install deps for a specific tool (omit for all tools)",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "read_file":
            from pathlib import Path
            path = Path(arguments["path"])
            try:
                content = path.read_text()
                return _text_result(content)
            except Exception as e:
                return _text_result(f"Error reading file: {e}")

        elif name == "http_request":
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
                    return _text_result(f"Status: {response.status_code}\n\n{response.text}")
            except Exception as e:
                return _text_result(f"Error making request: {e}")

        elif name == "run_command":
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
                output = ""
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n"
                output += f"\nReturn code: {result.returncode}"
                return _text_result(output)
            except subprocess.TimeoutExpired:
                return _text_result(f"Command timed out after {timeout} seconds")
            except FileNotFoundError:
                return _text_result(f"Command '{command}' not found")
            except Exception as e:
                return _text_result(f"Error executing command: {e}")

        elif name == "save_tool_spec":
            spec = arguments["spec"]
            with registry_save_tool_span(spec.get("name")):
                obs.set("language", spec.get("language"))
                obs.set("n_dependencies", len(spec.get("dependencies") or []))
                obs.set("n_tags", len(spec.get("tags") or []))
                try:
                    action = registry.save_tool(spec)
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
                    return _text_result(message)
                except Exception as e:
                    obs.event("save_tool_failed", error=str(e)[:256])
                    return _text_result(f"Error saving tool spec: {e}")

        elif name == "get_registry":
            tools = registry.list_tools_raw()
            if not tools:
                return _text_result("Registry is empty. No tools registered yet.")
            return _text_result(
                yaml.dump({"tools": tools}, default_flow_style=False, sort_keys=False)
            )

        elif name == "search_registry":
            tool_name = arguments.get("tool_name")
            query = arguments.get("query", "")
            tag = arguments.get("tag")
            top_k = arguments.get("top_k", 10)

            # Exact name lookup — fastest path
            if tool_name:
                tool = registry.get_tool(tool_name)
                if tool:
                    return _text_result(
                        f"Found tool '{tool_name}':\n\n"
                        + yaml.dump(tool, default_flow_style=False, sort_keys=False)
                    )
                return _text_result(f"No tool named '{tool_name}'.")

            # Semantic search + post-filter by tag via KB
            if kb:
                try:
                    results = kb.search(
                        query=query or "tool",
                        collection=TOOL_REGISTRY_COLLECTION,
                        top_k=top_k * 3 if tag else top_k,  # over-fetch if filtering
                    )
                    # Post-filter by tag (comma-separated string in metadata)
                    if tag and results:
                        results = [
                            r for r in results
                            if tag in r.get("chunk", {}).get("metadata", {}).get("tags", "")
                        ][:top_k]
                    if results:
                        summaries = []
                        for r in results:
                            chunk = r.get("chunk", {})
                            meta = chunk.get("metadata", {})
                            summaries.append(
                                f"- **{meta.get('tool_name', 'unknown')}** "
                                f"(score: {r.get('score', 0):.2f})\n"
                                f"  {chunk.get('text', '')[:200]}"
                            )
                        return _text_result(
                            f"Found {len(results)} tool(s):\n\n" + "\n\n".join(summaries)
                        )
                    return _text_result("No tools found matching the query.")
                except Exception:
                    pass  # Fall through to string matching

            # Fallback: string matching (no KB or KB search failed)
            tools = registry.list_tools_raw()
            if not tools:
                return _text_result("Registry is empty.")

            query_lower = query.lower()
            tag_lower = (tag or "").lower()
            results = []
            for tool in tools:
                if query_lower and query_lower not in tool.get("name", "").lower() and query_lower not in tool.get("description", "").lower():
                    continue
                if tag_lower and tag_lower not in ",".join(tool.get("tags", [])).lower():
                    continue
                results.append(tool)

            if results:
                return _text_result(
                    f"Found {len(results)} tool(s):\n\n"
                    + yaml.dump(results, default_flow_style=False, sort_keys=False)
                )
            return _text_result("No tools found matching the criteria.")

        elif name == "search_skills":
            skill_name = arguments.get("skill_name")
            query = arguments.get("query", "")
            tag = arguments.get("tag")
            top_k = arguments.get("top_k", 10)

            # Exact name lookup
            if skill_name and skill_registry:
                skill = skill_registry.get_skill(skill_name)
                if skill:
                    return _text_result(
                        f"Found skill '{skill_name}':\n\n"
                        + yaml.dump(skill, default_flow_style=False, sort_keys=False)
                    )
                return _text_result(f"No skill named '{skill_name}'.")

            # Semantic search via KB
            if kb:
                try:
                    results = kb.search(
                        query=query or "skill",
                        collection=SKILL_REGISTRY_COLLECTION,
                        top_k=top_k * 3 if tag else top_k,
                    )
                    if tag and results:
                        results = [
                            r for r in results
                            if tag in r.get("chunk", {}).get("metadata", {}).get("tags", "")
                        ][:top_k]
                    if results:
                        summaries = []
                        for r in results:
                            chunk = r.get("chunk", {})
                            meta = chunk.get("metadata", {})
                            summaries.append(
                                f"- **{meta.get('skill_name', 'unknown')}** "
                                f"(score: {r.get('score', 0):.2f})\n"
                                f"  {chunk.get('text', '')[:200]}"
                            )
                        return _text_result(
                            f"Found {len(results)} skill(s):\n\n" + "\n\n".join(summaries)
                        )
                    return _text_result("No skills found matching the query.")
                except Exception:
                    pass

            # Fallback: list all skills with string matching
            if skill_registry:
                skills = skill_registry.list_skills()
                if not skills:
                    return _text_result("No skills registered.")
                query_lower = query.lower()
                tag_lower = (tag or "").lower()
                results = []
                for s in skills:
                    if query_lower and query_lower not in s.get("name", "").lower() and query_lower not in s.get("description", "").lower():
                        continue
                    if tag_lower and tag_lower not in ",".join(s.get("tags", [])).lower():
                        continue
                    results.append(s)
                if results:
                    return _text_result(
                        f"Found {len(results)} skill(s):\n\n"
                        + yaml.dump(results, default_flow_style=False, sort_keys=False)
                    )
            return _text_result("No skills found matching the criteria.")

        elif name == "reconstruct_pipeline":
            fmt = arguments.get("format", "bash")
            trace_dir = registry.runtime_dir / "trace_archive"
            with registry_reconstruct_pipeline_span(fmt):
                try:
                    script = reconstruct_pipeline(trace_dir, fmt=fmt)
                    obs.set("output_chars", len(script))
                    return _text_result(script)
                except Exception as e:
                    obs.event("reconstruct_failed", error=str(e)[:256])
                    return _text_result(f"Error reconstructing pipeline: {e}")

        elif name == "install_dependencies":
            tool_name = arguments.get("tool_name")
            tools = registry.list_tools_raw()
            if not tools:
                return _text_result("Registry is empty. No tools registered yet.")

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
                return _text_result(f"No dependencies declared in {scope}.")

            seen = set()
            unique_deps = [d for d in all_deps if not (d in seen or seen.add(d))]

            with registry_install_deps_span(unique_deps):
                obs.set("scope_tool", tool_name)
                obs.set("n_tools_with_deps", len(tools_with_deps))
                result = _install_dependencies(unique_deps)
                # Heuristic: _install_dependencies returns "Successfully installed:"
                # on success and "Installation failed" / "timed out" / "Error:" on
                # failure paths.  We surface the status as an attribute so the UI
                # can filter without parsing the full result string.
                if result.startswith("Successfully installed:"):
                    obs.set("status", "ok")
                else:
                    obs.set("status", "failed")
                    obs.event("install_failed", message=result[:256])
                return _text_result(
                    f"Installing dependencies for: {', '.join(tools_with_deps)}\n\n{result}"
                )

        raise ValueError(f"Unknown tool: {name}")

    return server


def main():
    """Entry point for the registry builder server."""
    import logging as _logging

    parser = argparse.ArgumentParser(description="DSAgt Registry Builder MCP Server")
    parser.add_argument("--runtime-dir", default="./runtime")
    parser.add_argument("--source-tools-dir", default=None, help="Source directory for CLI tool specs")
    parser.add_argument("--source-skills-dir", default=None, help="Source directory for agent skills")
    parser.add_argument("--embedding-backend", default="api")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--otel-endpoint", default=None,
        help="OTLP HTTP base URL for tracing (default: $OTEL_EXPORTER_OTLP_ENDPOINT).")
    parser.add_argument("--session-id", default=None,
        help="DSAgt session id, attached as session.id on every emitted span.")
    args = parser.parse_args()

    from dsagt.observability import init_tracing, install_litellm_otel_callback
    init_tracing("dsagt-registry-server", args.otel_endpoint, args.session_id)
    install_litellm_otel_callback()

    runtime_dir = Path(args.runtime_dir)
    log = _logging.getLogger(__name__)

    # Create KB for semantic search
    embedder_kwargs = {}
    if args.embedding_model:
        embedder_kwargs["model"] = args.embedding_model
    if args.embedding_base_url:
        embedder_kwargs["base_url"] = args.embedding_base_url
    if args.embedding_api_key:
        embedder_kwargs["api_key"] = args.embedding_api_key

    kb = None
    if embedder_kwargs:
        kb = KnowledgeBase(
            index_dir=runtime_dir / "kb_index",
            default_embedder=args.embedding_backend,
            default_index="chroma",
            embedder_kwargs=embedder_kwargs,
        )

    registry = ToolRegistry(
        source_tools_dir=args.source_tools_dir,
        runtime_dir=str(runtime_dir),
        kb=kb,
    )

    skill_reg = SkillRegistry(
        source_skills_dir=args.source_skills_dir,
        runtime_dir=str(runtime_dir),
        kb=kb,
    )

    # Index on startup
    if kb:
        tool_count = registry.reindex_all()
        skill_count = skill_reg.reindex_all()
        if tool_count:
            log.info("Indexed %d tools into KB", tool_count)
        if skill_count:
            log.info("Indexed %d skills into KB", skill_count)

    server = create_registry_server(registry, kb, skill_reg)
    asyncio.run(_run_stdio(server, "registry"))


if __name__ == "__main__":
    main()
