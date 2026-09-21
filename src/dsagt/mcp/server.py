"""DSAGT MCP Server: the single ``dsagt-server``.

One process, one :class:`~dsagt.knowledge.KnowledgeBase`, one ``init_tracing``,
one MCP server per agent: a single embedder and a single Chroma owner back
every concern.  Single ownership matters for the ``skills_catalog__*``
collections, which are written under the skill concern and read under the
registry concern: one owner removes any write-here/read-there hazard across
them.  Heavy work runs off the event loop (``kb_ingest`` in a background job
task; the collectors in worker threads), so one process costs little
isolation.

Tool definitions and handlers are defined in their concern modules
(:mod:`~dsagt.mcp.registry_tools` / :mod:`~dsagt.mcp.knowledge_tools` /
:mod:`~dsagt.mcp.memory_tools` / :mod:`~dsagt.mcp.skill_tools`); this module
composes their ``(tools, handlers)`` under one dispatch shell
(:func:`build_dispatch_server`) and owns the shared-KB startup.  The factory
imports are lazy (inside :func:`create_dsagt_server` / :func:`main`) so the
concern modules can import :func:`build_dispatch_server` from here without a
cycle.
"""

import os

# Set before any import that may load a native runtime (e.g.
# ``dsagt.knowledge`` below): prevents a fatal OpenMP crash when multiple
# libraries each include their own libomp.
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jsonschema  # noqa: E402
import yaml  # noqa: E402

import mcp.server.stdio  # noqa: E402
import mcp.types as types  # noqa: E402
from mcp.server.lowlevel import Server, NotificationOptions  # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402

from dsagt.knowledge import KnowledgeBase  # noqa: E402
from dsagt.observability import bound, open_span  # noqa: E402
from dsagt.registry import SkillRegistry, CodeRegistry  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared dispatch shell (used by the merged server *and* the per-concern
# test-facing ``create_*_server`` wrappers)
# ---------------------------------------------------------------------------


def build_dispatch_server(
    name: str,
    tools,
    handlers,
    tool_category: dict[str, str] | None = None,
    ready: asyncio.Event | None = None,
) -> Server:
    """Wrap a ``(tools, handlers)`` pair in a configured MCP ``Server``.

    One dispatch contract for every concern module: reject arguments outside
    the tool's own ``input_schema``, run the handler, catch and wrap what it
    raises, then format by return type: a handler that returns ``str`` passes
    through, one that returns ``dict`` is JSON-encoded.  Registry handlers
    return ``str`` and never raise; knowledge handlers return ``dict`` and raise
    ``ValueError`` on bad input; both are covered.

    Argument validation and the outer error boundary are here because the
    mcp v2 lowlevel server provides neither.  Two consequences shape the code
    below.  Nothing may escape this function, because the v2 runner converts
    an exception into a JSON-RPC protocol error, which tears down the request
    instead of handing the agent something it can read and retry from; and
    every rejection carries ``is_error``, the only signal on the wire that a
    call failed.  The tool name is client-controlled, so an unknown one is a
    rejection, not a bug.

    ``tool_category`` maps tool name to concern (``memory`` / ``skill`` /
    ``knowledge`` / ``registry``).  Each call opens one categorization-root span
    (named for the tool) tagged ``dsagt.source=<category>``; the subsystem spans
    the handler opens (``kb.*`` / ``registry.*``) nest under it and inherit the
    category, so the source reflects the tool the agent called rather than
    the subsystem that did the work.  Tracing is inert outside a project, so
    this has no effect in the single-concern test servers or one-shot tools.

    ``ready`` is the startup gate: ``initialize`` and ``tools/list`` are answered
    at once, and a tool call waits on it, so the session, the trace store, and
    the catch-up can be set up after the client already sees the tools.  Codex
    starts its model turn without waiting for MCP servers, and a ``tool_search``
    that ran before the server had answered ``initialize`` found nothing.
    """
    tool_category = tool_category or {}
    schemas = {tool.name: tool.input_schema for tool in tools}

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    def rejected(message: str) -> types.CallToolResult:
        """A rejection the agent can read and retry from.

        ``is_error`` is what marks a result as failed on the wire; without it a
        client renders a rejection as a successful call and the agent has no
        signal to correct itself.
        """
        error = {"status": "error", "error": message}
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text", text=json.dumps(error, ensure_ascii=False)
                )
            ],
            is_error=True,
        )

    async def on_call_tool(
        ctx, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        tool_name = params.name
        # ``arguments`` is optional in the protocol (None when omitted), and the
        # mcp server dispatches without validating against input_schema, so
        # malformed calls are rejected here and handlers assume valid input.
        arguments = params.arguments or {}
        # The tool name is client-controlled: an agent inventing one, or holding
        # a stale name across a restart, must get a rejection back rather than an
        # escaping KeyError, which the runner turns into a JSON-RPC protocol
        # error that tears down the request instead of informing the agent.
        if tool_name not in handlers:
            return rejected(f"Unknown tool: {tool_name}")
        handler = handlers[tool_name]
        if ready is not None:
            await ready.wait()
        # Validation runs inside the span so a malformed call is traced like any
        # other; an agent looping on bad arguments is what the debug view
        # exists to show.  An unknown name has no category to tag and stays
        # untraced.
        rejection: str | None = None
        with open_span(tool_name, source=tool_category.get(tool_name)) as span:
            try:
                jsonschema.validate(instance=arguments, schema=schemas[tool_name])
                result = await handler(arguments)
            except jsonschema.ValidationError as e:
                rejection = f"Input validation error: {e.message}"
                result = {"status": "error", "error": rejection}
            except ValueError as e:
                result = {"status": "error", "error": str(e)}
            except Exception as e:
                logger.exception("Unexpected error in tool '%s'", tool_name)
                result = {"status": "error", "error": f"Unexpected error: {e}"}
            if span is not None:
                # The trace-level Request/Inputs/Outputs are read from this
                # categorization root; record the call's arguments and result
                # so the MLflow UI shows them as the trace's request and
                # response.  Both are agent-controlled and are written verbatim
                # to mlflow.db, so they go through ``bound``: credential keys
                # redacted, leaves truncated.
                span.set_inputs(bound(arguments))
                span.set_outputs(bound(result))
                if isinstance(result, dict) and result.get("status") == "error":
                    span.set_status("ERROR")
        if rejection is not None:
            return rejected(rejection)
        try:
            text = (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False)
            )
        except (TypeError, ValueError) as e:
            logger.exception("Tool '%s' returned an unserializable result", tool_name)
            return rejected(f"Unserializable result from {tool_name}: {e}")
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    return Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


PASS_INTERVAL_S = 45.0
TRACE_SOURCE_POLL_S = 2.0
# Import time is process start: the earliest point that is certainly before
# the agent's first message, which is what makes the mtime test below sound.
_SERVER_STARTED_AT = time.time()


async def _pin_trace_source(collector, project_dir, interval: float) -> None:
    """Record this session's trace-source token into ``state.yaml`` as soon as it exists.

    The token is what the next session's startup catch-up re-reads, so turns
    lost to an ungraceful kill still reach the store.  It cannot be taken at
    startup: the reader resolves "newest transcript", and until the agent's
    first message that is the previous session's.  Nor on the periodic pass:
    its first tick runs about 50 s in, after the KB build and the embedder
    load, and a scripted session is over by then, which loses the whole
    session.

    So poll fast, and accept a source only once it is provably this session's:
    a path modified since the process started (a transcript's mtime advances on
    every append, so a late first look only delays the pin to the next turn),
    or a non-path token (a DB session id, a session-dir name) that differs
    from the previous session's.  Exits once recorded; failure is logged, never
    fatal.
    """
    from dsagt.session import read_state, record_trace_source

    sessions = read_state(project_dir).get("sessions") or []
    previous = sessions[-2].get("trace_source") if len(sessions) >= 2 else None
    while True:
        await asyncio.sleep(interval)
        try:
            source = collector.active_source()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not resolve trace source: %s", e)
            continue
        if source is None:
            continue
        path = Path(source)
        try:
            fresh = path.stat().st_mtime >= _SERVER_STARTED_AT
        except OSError:  # not a path (DB session id, dir name), or gone
            fresh = None
        # A path is judged by freshness alone: a resumed session keeps writing
        # the previous session's transcript, and that file is this session's
        # source too.  A non-path token has no freshness to check, so it must
        # at least differ from the previous session's.
        if fresh is False or (fresh is None and source == previous):
            continue
        try:
            record_trace_source(project_dir, source)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record trace source: %s", e)
            return
        logger.info("Trace source pinned: %s", source)
        return


async def _periodic_pass(collector, tool_indexer, interval: float, project_dir) -> None:
    """Periodically run the trace collector and tool-use indexer on wall-clock time.

    Runs regardless of tool traffic, so a quiet session (the agent thinking,
    editing with its own tools, plain chat) is still captured.  Both block on
    disk, MLflow, and embedding, so they run in a worker thread to keep
    handlers responsive; a failure is logged, never fatal.
    """
    while True:
        await asyncio.sleep(interval)
        if collector is not None:
            try:
                n = await asyncio.to_thread(collector.collect)
                if n:
                    logger.info("Trace pass: logged %d trace(s)", n)
            except Exception as e:  # noqa: BLE001
                logger.warning("Trace pass failed: %s", e)
        if tool_indexer is not None:
            try:
                # tick_traced opens a code_use categorization root on the
                # worker thread so the indexer's kb.* writes nest under it as
                # tagged traces.
                n = await asyncio.to_thread(tool_indexer.tick_traced)
                if n:
                    logger.info("Tool-use pass: indexed %d record(s)", n)
            except Exception as e:  # noqa: BLE001
                logger.warning("Tool-use pass failed: %s", e)


async def _run_stdio(
    server: Server,
    name: str,
    *,
    startup=None,
    ready: asyncio.Event | None = None,
    project_dir=None,
) -> None:
    """Serve *server* over stdio.

    ``startup`` is a zero-argument callable run in a thread once the transport
    is up; it returns ``(collector, tool_indexer)`` and does the work that
    takes seconds on a fresh project (minting the session, ``init_tracing``
    with its sqlite schema, the catch-up).  The client's ``initialize`` and
    ``tools/list`` are answered meanwhile; ``ready`` opens the tool-call gate
    when startup returns, and the periodic pass and the trace-source poller
    start then.  A startup failure stops the server.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        run_task = asyncio.create_task(
            server.run(
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
        )
        tasks = []
        collector = tool_indexer = None
        try:
            if startup is not None:
                collector, tool_indexer = await asyncio.to_thread(startup)
                if collector is not None or tool_indexer is not None:
                    tasks.append(
                        asyncio.create_task(
                            _periodic_pass(
                                collector, tool_indexer, PASS_INTERVAL_S, project_dir
                            )
                        )
                    )
                if collector is not None:
                    tasks.append(
                        asyncio.create_task(
                            _pin_trace_source(
                                collector, project_dir, TRACE_SOURCE_POLL_S
                            )
                        )
                    )
            if ready is not None:
                ready.set()
            await run_task
        finally:
            if not run_task.done():
                run_task.cancel()
            for task in [run_task, *tasks]:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            # Best-effort end-of-session flush of the deferred final turn +
            # any unindexed tool-use — covers the graceful-exit case.  If
            # this is killed before it runs, the next session's startup
            # catch-up re-collects the previous session's transcript
            # (session.catch_up_extraction → _catch_up_traces, pinned to the
            # recorded transcript path); transcript-qualified acks make both
            # paths idempotent.  Tool-use likewise re-indexes via its ack set.
            if collector is not None:
                try:
                    await asyncio.to_thread(collector.collect, include_last=True)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Final trace pass failed: %s", e)
            if tool_indexer is not None:
                try:
                    await asyncio.to_thread(tool_indexer.tick)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Tool-use final flush failed: %s", e)


# ---------------------------------------------------------------------------
# Composition: merge the four concern modules' tools under one Server
# ---------------------------------------------------------------------------


def create_dsagt_server(
    registry: CodeRegistry,
    kb: KnowledgeBase | None,
    skill_registry: SkillRegistry | None,
    runtime_dir: str | Path | None = None,
    ready: asyncio.Event | None = None,
):
    """Compose the registry, knowledge, memory, and skill tools under one ``Server``.

    Test-facing API: build the registries and a (mock) KB, then drive the
    returned server via ``call_tool_sync()``.  ``main()`` constructs the real
    deps from project config before calling this.  Factory imports are lazy to
    keep the concern modules' top-level import of :func:`build_dispatch_server`
    cycle-free.
    """
    from dsagt.mcp.knowledge_tools import _knowledge_tools_and_handlers
    from dsagt.mcp.memory_tools import _memory_tools_and_handlers
    from dsagt.mcp.registry_tools import _registry_tools_and_handlers
    from dsagt.mcp.skill_tools import _skill_tools_and_handlers

    # (category, group): the category is the dsagt.source bucket stamped on
    # every trace rooted at one of that group's tools.
    groups = [
        ("registry", _registry_tools_and_handlers(registry, kb)),
        ("knowledge", _knowledge_tools_and_handlers(kb)),
        ("memory", _memory_tools_and_handlers(kb, runtime_dir)),
        ("skill", _skill_tools_and_handlers(skill_registry, kb, runtime_dir)),
    ]

    tools: list[types.Tool] = []
    handlers: dict = {}
    tool_category: dict[str, str] = {}
    for category, (g_tools, g_handlers) in groups:
        overlap = set(handlers) & set(g_handlers)
        if overlap:
            raise RuntimeError(
                f"dsagt-server tool-name collision across modules: {overlap}"
            )
        tools += g_tools
        handlers.update(g_handlers)
        for tool_name in g_handlers:
            tool_category[tool_name] = category

    return build_dispatch_server("dsagt", tools, handlers, tool_category, ready)


def _build_kb_from_config(config: dict, project_dir: Path) -> KnowledgeBase:
    """Construct the one shared KnowledgeBase from project config.

    The one place embedding-backend selection and the cross-backend leakage
    guard are defined.
    """
    # embedding is a code default filled in from DEFAULTS; the chunk_size
    # default is in KnowledgeBase itself.
    emb_config = config.get("embedding", {})

    backend = (emb_config.get("backend") or "local").lower()
    if backend not in ("local", "api"):
        raise ValueError(
            f"embedding.backend must be 'local' or 'api' (got {backend!r})"
        )

    # Cross-backend leakage guard: HuggingFace identifiers ("org/repo") and
    # OpenAI-style aliases ("text-embedding-3-small") share the same
    # EMBEDDING_MODEL env var in most setups.  When backend=local but the
    # resolved model is an OpenAI-style alias (no slash), drop the override so
    # we fall back to the LocalEmbedder default rather than 404 from HF.
    raw_model = (emb_config.get("model") or "").strip()
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    if raw_model and not raw_model.startswith("${"):
        looks_hf = "/" in raw_model
        if backend == "local" and not looks_hf:
            logger.warning(
                "Ignoring embedding.model=%r for backend=local (does not look "
                "like a HuggingFace identifier).  Falling back to the "
                "LocalEmbedder default.",
                raw_model,
            )
        else:
            model = raw_model
    if backend == "api":
        base_url = emb_config.get("base_url") or ""
        # Credentials are never on disk: the api key comes from the shell env
        # (EMBEDDING_API_KEY), threaded into MCP children via the env block.
        api_key = os.environ.get("EMBEDDING_API_KEY") or ""
        if not base_url:
            raise ValueError(
                "embedding.backend='api' requires embedding.base_url in "
                ".dsagt/config.yaml.  Either set it to your OpenAI-compatible "
                "endpoint, or change backend to 'local'."
            )
        if not api_key or api_key.startswith("${"):
            raise ValueError(
                "embedding.backend='api' requires the EMBEDDING_API_KEY env "
                "var (export it in your shell), or change backend to 'local'."
            )

    from dsagt.session import _recency_half_life

    # ``dsagt init`` provisions the project's kb_index with exactly the asset
    # set the project chose; the server only opens it.  Copying shared
    # collections here would add catalogs the project excluded.
    runtime_kb_dir = project_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Knowledge backend: %s", backend)
    kb = KnowledgeBase(
        index_dir=runtime_kb_dir,
        default_embedder=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        recency_half_life_days=_recency_half_life(config),
    )
    # Background-load the embedder so the model is ready at the agent's first
    # search or kb call, which otherwise pays the model load.
    kb.preload_default_embedder()
    return kb


def _spawn_catch_up(project_dir: Path, config: dict, kb=None) -> None:
    """Run :func:`dsagt.session.catch_up_extraction` in a daemon thread.

    Best-effort background catch-up of the previous session's post-session
    work.  Daemon so it never holds the server open; exceptions are logged,
    never propagated.
    """

    def _run() -> None:
        try:
            from dsagt.session import catch_up_extraction

            result = catch_up_extraction(project_dir, config, kb=kb)
            logger.info("Background catch-up complete: %s", result)
        except Exception as e:  # noqa: BLE001
            logger.warning("Background catch-up failed: %s", e)

    threading.Thread(target=_run, name="dsagt-catch-up", daemon=True).start()


def main():
    """Entry point for ``dsagt-server``.

    All configuration comes from the project directory:
    - ``./.dsagt/config.yaml``: project path and non-secret settings
    - ``EMBEDDING_*`` env vars: embedding credentials

    The agent's launch one-liner is ``cd <pdir> && <agent>``, so cwd is
    project_dir for the MCP children it spawns, and the server takes no
    CLI arguments.

    The server owns the session lifecycle: it appends a new entry to
    ``.dsagt/state.yaml`` (minting the session id) and spawns a background
    thread that runs the catch-up for the previous session, which needs no
    session-end trigger.
    """
    from dsagt.observability import (
        find_project_config,
        init_tracing,
    )
    from dsagt.session import (
        DEFAULTS,
        _deep_merge,
        append_session,
        load_user_env,
        resolve_env_vars,
        session_tag,
    )

    # First: for an agent that starts MCP children from the env block alone
    # (codex, cline), this file is the one source of a shared-store key or
    # an API embedder key.
    load_user_env()

    project_dir, _cfg = find_project_config()
    if project_dir is None:
        raise RuntimeError(
            "dsagt-server: no .dsagt/config.yaml in cwd "
            f"({Path.cwd()}).  Launch the agent from the project "
            "directory (`cd <pdir> && <agent>`)."
        )

    log_file = project_dir / "dsagt_server.log"
    # Default INFO; users opt into DEBUG via DSAGT_LOG_LEVEL=DEBUG.  At DEBUG,
    # transitive libraries (httpcore, urllib3, llama_index, chromadb) write
    # one stderr line per network op, and an agent that pipes the MCP
    # server's stderr into its own debug stream then buries the human output.
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
    logger.info("Server starting: project_dir: %s, log: %s", project_dir, log_file)

    cfg_file = project_dir / ".dsagt" / "config.yaml"
    # Fill in code defaults (embedding, etc.) the same way ``load_config``
    # does; the written config carries only the user's init choices.
    config = resolve_env_vars(
        _deep_merge(DEFAULTS, yaml.safe_load(cfg_file.read_text()) or {})
    )

    # A KB misconfig (e.g. embedding.backend='api' with no base_url/API key)
    # must not take down the whole server: the tool surface accepts kb=None and
    # only KB-backed tools degrade, so fall back rather than crash every tool.
    try:
        kb = _build_kb_from_config(config, project_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Knowledge base unavailable (%s); KB tools will degrade, "
            "non-KB tools are unaffected.",
            e,
        )
        kb = None

    # Bundled tools are embedded into the shared ~/dsagt-projects/kb_index/
    # by ``dsagt init`` (shared cache, once per machine) and copied into
    # the project's kb_index at init, so the server embeds nothing at
    # startup; save_code_spec embeds one entry at save time.
    registry = CodeRegistry(
        runtime_dir=str(project_dir),
        kb=kb,
    )
    skill_reg = SkillRegistry(
        runtime_dir=str(project_dir),
        kb=kb,
    )

    ready = asyncio.Event()
    server = create_dsagt_server(
        registry, kb, skill_reg, runtime_dir=str(project_dir), ready=ready
    )

    def startup():
        """Mint the session, start tracing, catch up, build the pass consumers.

        Runs in a thread after the transport is up (see ``_run_stdio``): on a
        fresh project ``init_tracing`` creates the sqlite schema, which took
        four seconds in one measured start, longer than a client waits before
        its first tool lookup.
        """
        # Own the session lifecycle: mint this session's id into state.yaml
        # and tag traces with it.  Best-effort — never block startup on state
        # I/O.
        session_id = None
        try:
            entry = append_session(project_dir)
            session_id = session_tag(config.get("project", ""), entry["id"])
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not mint session into state.yaml: %s", e)

        init_tracing("dsagt-server", session_id=session_id)

        # Run the catch-up for the previous session in the background.
        # Daemon thread: best-effort, never fails startup.
        _spawn_catch_up(project_dir, config, kb=kb)

        # The periodic pass: read the live transcript into MLflow.  The
        # loop is agent-agnostic; ``make_trace_collector`` returns a collector
        # for any agent with a registered (reader, translator) pair and
        # ``None`` otherwise.  Best-effort — a collector that can't be built
        # never blocks the server.
        collector = None
        try:
            from dsagt.memory import episodic_consumers
            from dsagt.observability import experiment_name, resolve_tracking_uri
            from dsagt.traces import make_trace_collector

            resolve_cfg = dict(config)
            resolve_cfg["project_dir"] = str(project_dir)
            collector = make_trace_collector(
                config.get("agent"),
                project_dir,
                config.get("project", ""),
                session_id or "",
                resolve_tracking_uri(resolve_cfg),
                experiment=experiment_name(resolve_cfg),
                extra_consumers=episodic_consumers(config, kb, project_dir, session_id),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not start the periodic trace pass: %s", e)

        # Tool-use indexer: incremental, idempotent embedding of dsagt-run
        # records into the ``code_use`` collection on the same periodic pass.
        tool_indexer = None
        try:
            from dsagt.provenance import CodeUseIndexer

            tool_indexer = CodeUseIndexer(kb, project_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not start tool-use indexer: %s", e)
        return collector, tool_indexer

    try:
        asyncio.run(
            _run_stdio(
                server, "dsagt", startup=startup, ready=ready, project_dir=project_dir
            )
        )
    finally:
        kb.close()


if __name__ == "__main__":
    main()
