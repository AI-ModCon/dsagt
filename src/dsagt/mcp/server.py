"""DSAGT MCP Server — the single merged registry + knowledge server.

Supersedes the two former servers (``dsagt-registry-server`` +
``dsagt-knowledge-server``).  Both previously constructed their own
:class:`~dsagt.knowledge.KnowledgeBase` — two embedders, two Chroma accesses,
and a write-here/read-there hazard on the ``skills_catalog__*`` collections
(synced by knowledge, searched by registry).  Merging into one process gives one
embedder, one Chroma owner, one ``init_tracing``, and one MCP server per agent.

The heavy/risky work is already offloaded out of the event loop (``run_command``
→ ``dsagt-run`` subprocess; ``kb_ingest`` → background job thread), so collapsing
the two processes costs little isolation.

Tool *definitions* and *handlers* live in their concern modules
(:mod:`~dsagt.mcp.registry_tools` / :mod:`~dsagt.mcp.knowledge_tools` /
:mod:`~dsagt.mcp.memory_tools` / :mod:`~dsagt.mcp.skill_tools`); this module only
composes their ``(tools, handlers)`` under one dispatch shell
(:func:`build_dispatch_server`) and owns the shared-KB startup.  The factory
imports are *lazy* (inside :func:`create_dsagt_server` / :func:`main`) so the
concern modules can import :func:`build_dispatch_server` from here without a
cycle.

See ``design-notes/skills-catalog-server-merge.md`` §2.

Backward compatibility is **rebuild-not-migrate**: a project created against the
old two-server layout adopts this by re-running ``dsagt start`` (which
regenerates the per-agent MCP config to a single ``dsagt`` server).  See the
upgrade note in the README.
"""

import os

# Set before any import that may pull in FAISS / PyTorch / sentence-transformers
# (e.g. ``dsagt.knowledge`` below): prevents a fatal OpenMP crash when multiple
# libraries each bundle their own libomp.
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

import mcp.server.stdio  # noqa: E402
import mcp.types as types  # noqa: E402
from mcp.server.lowlevel import Server, NotificationOptions  # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402

from dsagt.knowledge import KnowledgeBase  # noqa: E402
from dsagt.registry import SkillRegistry, ToolRegistry  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared dispatch shell (used by the merged server *and* the per-concern
# test-facing ``create_*_server`` wrappers)
# ---------------------------------------------------------------------------


def build_dispatch_server(name: str, tools, handlers) -> Server:
    """Wrap a ``(tools, handlers)`` pair in a configured MCP ``Server``.

    One dispatch contract for every concern module: catch + wrap errors, then
    format by return type — a handler that returns ``str`` passes through, one
    that returns ``dict`` is JSON-encoded.  This is a superset of the old
    per-server behavior (registry handlers returned ``str`` and never raised;
    knowledge handlers returned ``dict`` and raised ``ValueError`` on bad
    input), so it is behavior-preserving for both.
    """
    server = Server(name)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict) -> list[types.TextContent]:
        handler = handlers[tool_name]  # KeyError = bug in list_tools schema
        try:
            result = await handler(arguments)
        except ValueError as e:
            result = {"status": "error", "error": str(e)}
        except Exception as e:
            logger.exception("Unexpected error in tool '%s'", tool_name)
            result = {"status": "error", "error": f"Unexpected error: {e}"}
        text = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )
        return [types.TextContent(type="text", text=text)]

    return server


async def _run_stdio(server: Server, name: str) -> None:
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


# ---------------------------------------------------------------------------
# Composition — merge the four concern modules' tools under one Server
# ---------------------------------------------------------------------------


def create_dsagt_server(
    registry: ToolRegistry,
    kb: KnowledgeBase | None,
    skill_registry: SkillRegistry | None,
    runtime_dir: str | Path | None = None,
):
    """Compose the registry + knowledge + memory + skill tools under one ``Server``.

    Test-facing API: build the registries + a (mock) KB, then drive the
    returned server via ``call_tool_sync()``.  ``main()`` constructs the real
    deps from project config before calling this.  Factory imports are lazy to
    keep the concern modules' top-level import of :func:`build_dispatch_server`
    cycle-free.
    """
    from dsagt.mcp.knowledge_tools import _knowledge_tools_and_handlers
    from dsagt.mcp.memory_tools import _memory_tools_and_handlers
    from dsagt.mcp.registry_tools import _registry_tools_and_handlers
    from dsagt.mcp.skill_tools import _skill_tools_and_handlers

    groups = [
        _registry_tools_and_handlers(registry, kb),
        _knowledge_tools_and_handlers(kb),
        _memory_tools_and_handlers(kb, runtime_dir),
        _skill_tools_and_handlers(skill_registry, kb, runtime_dir),
    ]

    tools: list[types.Tool] = []
    handlers: dict = {}
    for g_tools, g_handlers in groups:
        overlap = set(handlers) & set(g_handlers)
        if overlap:
            raise RuntimeError(
                f"dsagt-server tool-name collision across modules: {overlap}"
            )
        tools += g_tools
        handlers.update(g_handlers)

    return build_dispatch_server("dsagt", tools, handlers)


def _build_kb_from_config(config: dict, project_dir: Path) -> KnowledgeBase:
    """Construct the one shared KnowledgeBase from project config.

    The single home for embedding-backend selection + the cross-backend
    leakage guard that the two former server mains duplicated near-verbatim.
    """
    from dsagt.session import REGISTRY_DIR, setup_runtime_kb

    kb_config = config["knowledge"]
    emb_config = config["embedding"]

    backend = (emb_config.get("backend") or "local").lower()
    if backend not in ("local", "api"):
        raise ValueError(
            f"embedding.backend must be 'local' or 'api' (got {backend!r})"
        )

    # Cross-backend leakage guard: HuggingFace identifiers ("org/repo") and
    # OpenAI-style aliases ("text-embedding-3-small") share the same
    # EMBEDDING_MODEL env var in most setups.  When backend=local but the
    # resolved model is an OpenAI-style alias (no slash), drop the override so
    # we fall back to the LocalEmbeddingClient default rather than 404 from HF.
    raw_model = (emb_config.get("model") or "").strip()
    embedder_kwargs: dict = {}
    if raw_model and not raw_model.startswith("${"):
        looks_hf = "/" in raw_model
        if backend == "local" and not looks_hf:
            logger.warning(
                "Ignoring embedding.model=%r for backend=local (does not look "
                "like a HuggingFace identifier).  Falling back to the "
                "LocalEmbeddingClient default.",
                raw_model,
            )
        else:
            embedder_kwargs["model"] = raw_model
    if backend == "api":
        base_url = emb_config.get("base_url") or ""
        api_key = emb_config.get("api_key") or ""
        if not base_url:
            raise ValueError(
                "embedding.backend='api' requires embedding.base_url in "
                "dsagt_config.yaml.  Either set it to your OpenAI-compatible "
                "endpoint, or change backend to 'local'."
            )
        if not api_key or api_key.startswith("${"):
            raise ValueError(
                "embedding.backend='api' requires embedding.api_key in "
                "dsagt_config.yaml.  Either fill it in (or export the "
                "${EMBEDDING_API_KEY} env var), or change backend to 'local'."
            )
        embedder_kwargs.update({"base_url": base_url, "api_key": api_key})

    runtime_kb_dir = setup_runtime_kb(REGISTRY_DIR / "kb_index", project_dir)
    logger.info("Knowledge backend: %s", backend)
    kb = KnowledgeBase(
        index_dir=runtime_kb_dir,
        chunk_size=kb_config["chunk_size"],
        default_rerank=kb_config["rerank"],
        default_embedder=backend,
        default_index=kb_config["vector_db"],
        embedder_kwargs=embedder_kwargs,
    )
    # Background-load the embedder so the model is ready when the agent's first
    # search / kb call lands (otherwise the first call pays the ~5-10s
    # sentence-transformers import + construction, which looks like a hang).
    kb.preload_default_embedder()
    return kb


def main():
    """Entry point for ``dsagt-server``.

    All configuration comes from the project directory:
    - ``./dsagt_config.yaml`` → project path + non-secret settings
    - ``EMBEDDING_*`` env vars → embedding credentials

    No CLI arguments.  By contract the agent's launch one-liner is
    ``cd <pdir> && <agent>``, so cwd is project_dir for the MCP children it
    spawns.
    """
    from dsagt.observability import (
        configure_litellm_retries,
        find_project_config,
        init_tracing,
    )
    from dsagt.session import resolve_env_vars

    project_dir, _cfg = find_project_config()
    if project_dir is None:
        raise RuntimeError(
            "dsagt-server: no dsagt_config.yaml in cwd "
            f"({Path.cwd()}).  Launch the agent from the project "
            "directory (`cd <pdir> && <agent>`)."
        )

    log_file = project_dir / "dsagt_server.log"
    # Default INFO; users opt into DEBUG via DSAGT_LOG_LEVEL=DEBUG.  At DEBUG,
    # transitive libraries (httpcore, urllib3, llama_index, chromadb) flood
    # stderr with one line per network op — when an agent pipes the MCP
    # server's stderr into its own debug stream, human output gets buried.
    _level_name = os.environ.get("DSAGT_LOG_LEVEL", "INFO").upper()
    _level = getattr(logging, _level_name, logging.INFO)
    logging.basicConfig(
        level=_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Server starting — project_dir: %s, log: %s", project_dir, log_file)

    config_path = project_dir / "dsagt_config.yaml"
    config = resolve_env_vars(yaml.safe_load(config_path.read_text()))

    init_tracing("dsagt-server")  # session_id picked up from DSAGT_SESSION_ID env
    configure_litellm_retries()

    kb = _build_kb_from_config(config, project_dir)

    # Bundled tools are pre-embedded in the shared ~/dsagt-projects/kb_index/
    # by ``dsagt setup-kb`` (or the auto-bootstrap in ``dsagt start``) and
    # copied into the project's kb_index by ``setup_runtime_kb`` above.  No
    # bundled embedding work happens here; save_tool_spec incurs a single
    # embed at save time.
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

    server = create_dsagt_server(registry, kb, skill_reg, runtime_dir=str(project_dir))
    try:
        asyncio.run(_run_stdio(server, "dsagt"))
    finally:
        kb.close()


if __name__ == "__main__":
    main()
