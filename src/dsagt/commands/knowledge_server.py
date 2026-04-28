"""
DSAgt Knowledge Base MCP Server.

Provides semantic search over document collections for MCP-compatible agents.

At startup, symlinks base indexes into a session-specific runtime directory.
All modifications (ingestion, append) happen in the runtime copy.

Long-running operations (ingest, append) run in the background and return
immediately with a job_id. Use kb_job_status to poll for completion.

Server configuration (chunk_size, vector_db, rerank) is read from the
project's dsagt_config.yaml.  Embedding credentials flow through env vars
(LLM_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL) set by dsagt start.

Usage:
    dsagt-knowledge-server --base-index-dir ./kb_index --runtime-dir ./runtime
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial

# Prevent fatal OpenMP crash when multiple libraries (FAISS, PyTorch/
# sentence-transformers) each bundle their own libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
import yaml
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from dsagt.knowledge import EMBEDDER_REGISTRY, VECTORINDEX_REGISTRY, CollectionRoute, KnowledgeBase
from dsagt.memory import EPISODIC_MEMORY_ROUTE, SuggestionQueue
from dsagt.memory import ExplicitMemory
from dsagt.session import REGISTRY_DIR, _collection_exists, setup_runtime_kb  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP server helpers
# ---------------------------------------------------------------------------

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




# _collection_exists and setup_runtime_kb live in dsagt.session (imported above).


def _register_external_collection(
    kb: KnowledgeBase,
    collection_name: str,
    vector_db: str,
    connection_params: dict,
    embedding_model: str,
    description: str,
) -> None:
    """Wire an already-built external vector store into the routing registry."""
    coll_dir = kb.index_dir / collection_name
    coll_dir.mkdir(exist_ok=True)

    if description:
        (coll_dir / "DESCRIPTION.md").write_text(description)

    if vector_db == "chroma":
        index_kwargs = {
            "collection_name": connection_params.get("collection", collection_name),
            "persist_dir": None,
            "host": connection_params.get("host", "localhost"),
            "port": connection_params.get("port", 8000),
        }
    elif vector_db == "lancedb":
        index_kwargs = {
            "uri": connection_params["uri"],
            "table": connection_params.get("table", collection_name),
        }
    elif vector_db == "qdrant":
        index_kwargs = {
            "url": connection_params["url"],
            "collection": connection_params.get("collection", collection_name),
            "api_key": connection_params.get("api_key"),
        }
    else:
        raise ValueError(
            f"Unsupported vector DB '{vector_db}'. "
            f"Choose from: chroma, lancedb, qdrant"
        )

    route = CollectionRoute(
        embedding_backend="api",
        vector_db=vector_db,
        embedder_kwargs={"model": embedding_model},
        index_kwargs=index_kwargs,
        description=description,
    )
    kb.register_route(collection_name, route)




# ---------------------------------------------------------------------------
# Background job tracker
# ---------------------------------------------------------------------------

@dataclass
class _JobTracker:
    """Tracks background ingest/append jobs and their completion state."""

    jobs: dict[str, dict] = field(default_factory=dict)
    active_collections: set[str] = field(default_factory=set)

    def start(self, coro, collection: str | None = None) -> str:
        job_id = uuid.uuid4().hex[:8]
        self.jobs[job_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "collection": collection,
            "started_at": time.monotonic(),
            "message": "Starting -- embedding documents via API...",
        }
        if collection:
            self.active_collections.add(collection)

        tracker = self  # capture for the closure

        async def _run():
            try:
                tracker.jobs[job_id]["message"] = "Embedding and indexing documents..."
                result = await coro
                tracker.jobs[job_id]["status"] = "complete"
                tracker.jobs[job_id]["result"] = result
                tracker.jobs[job_id]["message"] = "Done."
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                tracker.jobs[job_id]["status"] = "error"
                tracker.jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
                tracker.jobs[job_id]["message"] = f"Failed: {type(e).__name__}: {e}"
                tracker.jobs[job_id]["traceback"] = tb
                logger.error("Job %s failed: %s\n%s", job_id, e, tb)
            finally:
                if collection:
                    tracker.active_collections.discard(collection)

        asyncio.get_event_loop().create_task(_run())
        return job_id


# ---------------------------------------------------------------------------
# Per-tool handlers (module-level, explicit dependencies)
#
# Each handler takes ``arguments: dict`` plus its dependencies as keyword
# args bound via functools.partial in create_knowledge_server().  Handlers
# return a result dict; the outer call_tool wrapper JSON-serializes it.
# ---------------------------------------------------------------------------

async def _handle_kb_list_collections(arguments: dict, *, kb: KnowledgeBase) -> dict:
    collections = await asyncio.to_thread(kb.list_collections)
    return {"status": "ok", "collections": collections, "count": len(collections)}


async def _handle_kb_search(
    arguments: dict, *, kb: KnowledgeBase,
) -> dict:
    query = arguments["query"]
    top_k = arguments.get("top_k", 5)
    rerank = arguments.get("rerank")  # None → kb.default_rerank

    collection_arg = arguments.get("collection")
    collections_arg = arguments.get("collections")

    if not collection_arg and not collections_arg:
        return {"status": "error", "error": "Provide 'collection' or 'collections'"}

    # Build ChromaDB where clause from the filter arguments.  ChromaDB
    # requires single-filter dicts or $and-wrapped lists; an empty dict
    # would be invalid, so we only pass where when there are real filters.
    where = {
        key: arguments[key]
        for key in ("category", "session_id", "source_type", "tool_name")
        if arguments.get(key) is not None
    }
    return_code = arguments.get("return_code")
    if return_code is not None:
        where["return_code"] = int(return_code)
    if len(where) > 1:
        where = {"$and": [{k: v} for k, v in where.items()]}

    target_collections = collections_arg or [collection_arg]
    all_results = []
    search_errors = []

    for coll_name in target_collections:
        try:
            search_kwargs = dict(query=query, collection=coll_name, top_k=top_k, rerank=rerank)
            if where:
                search_kwargs["where"] = where
            coll_results = await asyncio.to_thread(kb.search, **search_kwargs)
            all_results.extend(coll_results)
        except ValueError as e:
            logger.warning("Search failed for '%s': %s", coll_name, e)
            search_errors.append(str(e))

    if search_errors and not all_results:
        if len(target_collections) == 1:
            return {"status": "error", "error": search_errors[0]}
        return {"status": "error", "error": f"All collections failed: {'; '.join(search_errors)}"}

    score_key = "rerank_score" if rerank else "score"
    all_results.sort(key=lambda r: r.get(score_key, r["score"]), reverse=True)
    all_results = all_results[:top_k]

    result = {
        "status": "ok",
        "query": query,
        "collection": collection_arg or ",".join(collections_arg),
        "result_count": len(all_results),
        "results": [
            {
                "text": r["chunk"]["text"],
                "score": r["score"],
                "rerank_score": r.get("rerank_score"),
                "source_file": r["chunk"]["metadata"].get("source_file", ""),
                "chunk_index": r["chunk"]["metadata"].get("chunk_index", 0),
                "metadata": {
                    k: v for k, v in r["chunk"]["metadata"].items()
                    if k not in ("source_file", "chunk_index", "collection", "file_type")
                },
            }
            for r in all_results
        ],
    }
    if search_errors:
        result["warnings"] = search_errors
    return result


async def _handle_kb_ingest(
    arguments: dict, *, kb: KnowledgeBase, job_tracker: _JobTracker,
) -> dict:
    folder_path = Path(arguments["folder_path"])
    collection_name = arguments.get("collection_name")
    file_types = arguments.get("file_types")
    embedding_backend = arguments.get("embedding_backend")
    embedding_model = arguments.get("embedding_model")
    vector_db = arguments.get("vector_db")

    if not folder_path.exists():
        return {"status": "error", "error": f"Folder not found: {folder_path}"}
    if not folder_path.is_dir():
        return {"status": "error", "error": f"Not a directory: {folder_path}"}

    target_name = collection_name or folder_path.name
    warning = None

    if target_name in job_tracker.active_collections:
        return {
            "status": "error",
            "error": (
                f"Collection '{target_name}' is already being ingested. "
                f"Poll kb_job_status for progress."
            ),
        }

    if _collection_exists(kb.index_dir / target_name):
        source_path = kb.index_dir / target_name / "source.txt"
        existing_source = source_path.read_text().strip() if source_path.exists() else None
        same_source = (
            existing_source is None
            or Path(existing_source).resolve() == folder_path.resolve()
        )
        if not same_source:
            original_name = target_name
            n = 1
            while (
                _collection_exists(kb.index_dir / target_name)
                or target_name in job_tracker.active_collections
            ):
                target_name = f"{original_name}{n}"
                n += 1
            warning = (
                f"Collection '{original_name}' already exists from a "
                f"different folder; using '{target_name}'."
            )

    route = None
    if embedding_backend or embedding_model or vector_db:
        default = kb._default_route
        inherited_model = embedding_model or default.embedder_kwargs.get("model")
        route = CollectionRoute(
            embedding_backend=embedding_backend or default.embedding_backend,
            vector_db=vector_db or default.vector_db,
            embedder_kwargs={"model": inherited_model} if inherited_model else {},
        )

    ingest_kwargs: dict = {"collection_name": target_name}
    if file_types:
        ingest_kwargs["file_types"] = file_types
    if route is not None:
        ingest_kwargs["route"] = route

    async def _ingest_with_logging():
        import traceback as _tb
        logger.info("Ingest starting: collection=%s folder=%s kwargs=%s",
                     target_name, folder_path, ingest_kwargs)
        try:
            result = await asyncio.to_thread(kb.ingest, folder_path, **ingest_kwargs)
            logger.info("Ingest complete: %s", result)
            return result
        except Exception as _e:
            logger.error("Ingest FAILED: %s\n%s", _e, _tb.format_exc())
            raise

    job_id = job_tracker.start(_ingest_with_logging(), collection=target_name)
    result = {
        "status": "started",
        "job_id": job_id,
        "collection": target_name,
        "message": (
            f"Ingestion started. "
            f"Poll kb_job_status(job_id='{job_id}') every 10 seconds. "
            f"DO NOT call ingest again -- the job is running in the "
            f"background. Large folders may take several minutes."
        ),
    }
    if warning:
        result["warning"] = warning
    return result


async def _handle_kb_append(
    arguments: dict, *, kb: KnowledgeBase, job_tracker: _JobTracker,
) -> dict:
    collection = arguments["collection"]
    paths = arguments["paths"]
    if isinstance(paths, str):
        paths = [paths]
    file_types = arguments.get("file_types")

    if not _collection_exists(kb.index_dir / collection):
        return {"status": "error", "error": f"Collection '{collection}' not found"}

    append_kwargs: dict = {}
    if file_types:
        append_kwargs["file_types"] = file_types

    job_id = job_tracker.start(
        asyncio.to_thread(kb.append, collection, paths, **append_kwargs),
        collection=collection,
    )
    return {
        "status": "started",
        "job_id": job_id,
        "collection": collection,
        "message": f"Append started. Poll kb_job_status(job_id='{job_id}') for progress.",
    }


async def _handle_kb_add_vector_db(arguments: dict, *, kb: KnowledgeBase) -> dict:
    collection_name = arguments["collection_name"]
    vector_db = arguments["vector_db"]
    connection_params = arguments["connection_params"]
    embedding_model = arguments["embedding_model"]
    description = arguments.get("description", "")

    if (kb.index_dir / collection_name).exists():
        return {
            "status": "error",
            "error": (
                f"Collection '{collection_name}' already exists. "
                "Choose a different name or delete the existing collection."
            ),
        }

    await asyncio.to_thread(
        _register_external_collection,
        kb, collection_name, vector_db,
        connection_params, embedding_model, description,
    )
    return {
        "status": "ok",
        "collection": collection_name,
        "vector_db": vector_db,
        "embedding_model": embedding_model,
        "message": (
            f"External collection '{collection_name}' registered. "
            "Use search to query it."
        ),
    }


async def _handle_kb_job_status(arguments: dict, *, job_tracker: _JobTracker) -> dict:
    job_id = arguments["job_id"]
    if job_id not in job_tracker.jobs:
        return {"status": "error", "error": f"Unknown job: {job_id}"}

    job = job_tracker.jobs[job_id]
    elapsed = int(time.monotonic() - job["started_at"])
    result = {
        "status": job["status"],
        "elapsed_seconds": elapsed,
        "message": job.get("message", ""),
    }
    if job["status"] == "running":
        result["instruction"] = (
            "Job is still running. DO NOT call ingest again. "
            "Keep polling job_status every 10 seconds until "
            "status is 'complete' or 'error'."
        )
    if job["result"] is not None:
        result["result"] = job["result"]
    if job["error"] is not None:
        result["error"] = job["error"]
    if job.get("traceback") and job["status"] == "error":
        result["traceback"] = job["traceback"]
    return result


async def _handle_kb_remember(
    arguments: dict,
    *,
    kb: KnowledgeBase,
    memory: ExplicitMemory,
    suggestions: SuggestionQueue,
) -> dict:
    text = arguments["text"]
    category = arguments.get("category", "")
    session_id = arguments.get("session_id", "")
    supersedes = arguments.get("supersedes")
    promoted_from = arguments.get("promoted_from")

    store_result = await asyncio.to_thread(
        memory.remember,
        text=text,
        category=category,
        session_id=session_id,
        supersedes=supersedes,
    )

    if not store_result.get("stored"):
        return {
            "status": "error",
            "error": store_result.get("error", "Failed to store memory"),
        }

    await asyncio.to_thread(
        kb.add_entries,
        texts=[text],
        collection="episodic_memory",
        metadatas=[{
            "source_type": "explicit_memory",
            "category": category,
            "session_id": session_id,
        }],
        route=EPISODIC_MEMORY_ROUTE,
    )

    if promoted_from:
        suggestions.dismiss(promoted_from)

    return {
        "status": "ok",
        "entry_id": store_result["entry_id"],
        "superseded_id": store_result.get("superseded_id"),
        "promoted_from": promoted_from,
        "total_memories": await asyncio.to_thread(memory.count),
    }


async def _handle_kb_get_memories(
    arguments: dict, *, memory: ExplicitMemory, suggestions: SuggestionQueue,
) -> dict:
    entries = await asyncio.to_thread(memory.get_all)
    pending = suggestions.get_all()
    result = {"status": "ok", "count": len(entries), "memories": entries}
    if pending:
        result["suggestions"] = pending
        result["suggestion_count"] = len(pending)
    return result


async def _handle_kb_get_suggestions(
    arguments: dict, *, suggestions: SuggestionQueue,
) -> dict:
    pending = suggestions.get_all()
    return {"status": "ok", "count": len(pending), "suggestions": pending}


async def _handle_kb_dismiss_suggestion(
    arguments: dict, *, suggestions: SuggestionQueue,
) -> dict:
    suggestion_id = arguments["suggestion_id"]
    dismissed = suggestions.dismiss(suggestion_id)
    if not dismissed:
        return {"status": "error", "error": f"Suggestion not found: {suggestion_id}"}
    return {"status": "ok", "dismissed": suggestion_id, "remaining": suggestions.count}


# ---------------------------------------------------------------------------
# Server factory (thin wiring — used by main() and tests)
# ---------------------------------------------------------------------------

def create_knowledge_server(
    kb: KnowledgeBase,
    runtime_dir: str | Path | None = None,
):
    """Create and configure the MCP knowledge server.

    This is the test-facing API: tests call it with a mock KB and get back
    a server they can drive via call_tool_sync().  main() reads the project
    config and constructs KB before calling this.

    The rerank default is on ``kb.default_rerank`` (set from
    ``knowledge.rerank`` in dsagt_config.yaml).
    """
    server = Server("knowledge")

    mem_dir = Path(runtime_dir) if runtime_dir else kb.index_dir.parent
    memory = ExplicitMemory(runtime_dir=mem_dir)
    suggestions = SuggestionQueue(mem_dir / "suggestions.json")
    job_tracker = _JobTracker()

    handlers = {
        "kb_list_collections": partial(_handle_kb_list_collections, kb=kb),
        "kb_search": partial(_handle_kb_search, kb=kb),
        "kb_ingest": partial(_handle_kb_ingest, kb=kb, job_tracker=job_tracker),
        "kb_append": partial(_handle_kb_append, kb=kb, job_tracker=job_tracker),
        "kb_add_vector_db": partial(_handle_kb_add_vector_db, kb=kb),
        "kb_job_status": partial(_handle_kb_job_status, job_tracker=job_tracker),
        "kb_remember": partial(_handle_kb_remember, kb=kb, memory=memory, suggestions=suggestions),
        "kb_get_memories": partial(_handle_kb_get_memories, memory=memory, suggestions=suggestions),
        "kb_get_suggestions": partial(_handle_kb_get_suggestions, suggestions=suggestions),
        "kb_dismiss_suggestion": partial(_handle_kb_dismiss_suggestion, suggestions=suggestions),
    }

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="kb_list_collections",
                description=(
                    "List all available knowledge base collections with their "
                    "embedding model and vector DB. Use this to discover what "
                    "documentation is already indexed."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="kb_search",
                description=(
                    "Search knowledge base collections using semantic similarity. "
                    "Returns relevant chunks with source metadata. "
                    "Supports multi-collection search."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "collection": {
                            "type": "string",
                            "description": "Name of a single collection to search",
                        },
                        "collections": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search multiple collections and merge results (overrides 'collection')",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                        "rerank": {
                            "type": "boolean",
                            "description": "Use cross-encoder reranking (slower but more accurate). Default from config.",
                            "default": kb.default_rerank,
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category tag (ChromaDB collections only)",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Filter by session ID (ChromaDB collections only)",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Filter by tool name (ChromaDB collections only)",
                        },
                        "source_type": {
                            "type": "string",
                            "description": "Filter by source type (ChromaDB collections only)",
                        },
                        "return_code": {
                            "type": "integer",
                            "description": "Filter by tool exit code (ChromaDB collections only)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="kb_ingest",
                description=(
                    "Index a folder as a new knowledge base collection. "
                    "Returns immediately with a job_id. "
                    "IMPORTANT: poll kb_job_status every 10 seconds and wait for "
                    "status='complete'. DO NOT call ingest again for the same "
                    "folder while a job is running."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_path": {
                            "type": "string",
                            "description": "Path to folder containing documents to index",
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Name for the collection (default: folder name)",
                        },
                        "file_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to include, e.g. ['pdf', 'md', 'py']. Defaults to common types.",
                        },
                        "embedding_backend": {
                            "type": "string",
                            "enum": list(EMBEDDER_REGISTRY.keys()),
                            "description": "Embedding backend override for this collection.",
                        },
                        "embedding_model": {
                            "type": "string",
                            "description": "Embedding model override for this collection.",
                        },
                        "vector_db": {
                            "type": "string",
                            "enum": list(VECTORINDEX_REGISTRY.keys()),
                            "description": "Vector database override for this collection.",
                        },
                    },
                    "required": ["folder_path"],
                },
            ),
            types.Tool(
                name="kb_append",
                description=(
                    "Add documents to an existing collection. Uses the same embedding "
                    "model and vector DB the collection was created with. "
                    "Returns immediately with a job_id -- poll kb_job_status for progress."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Name of the existing collection to append to",
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file or folder paths to add",
                        },
                        "file_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to include when expanding folders.",
                        },
                    },
                    "required": ["collection", "paths"],
                },
            ),
            types.Tool(
                name="kb_add_vector_db",
                description=(
                    "Register an already-built external vector store as a collection. "
                    "Queries will be embedded via the API using the specified model."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {"type": "string", "description": "Unique name for this collection"},
                        "vector_db": {"type": "string", "enum": ["chroma", "lancedb", "qdrant"], "description": "Vector store backend type"},
                        "connection_params": {"type": "object", "description": "Backend-specific connection parameters."},
                        "embedding_model": {"type": "string", "description": "The API model used to build this index"},
                        "description": {"type": "string", "description": "Human-readable description for agent discovery"},
                    },
                    "required": ["collection_name", "vector_db", "connection_params", "embedding_model"],
                },
            ),
            types.Tool(
                name="kb_job_status",
                description="Check the status of a background ingest or append job.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID returned by kb_ingest or kb_append"},
                    },
                    "required": ["job_id"],
                },
            ),
            types.Tool(
                name="kb_remember",
                description=(
                    "Store a user-confirmed fact as an explicit memory. "
                    "These persist across sessions. Use 'supersedes' to replace an outdated memory."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The fact to remember"},
                        "category": {"type": "string", "description": "Classification tag"},
                        "session_id": {"type": "string", "description": "Current session identifier"},
                        "supersedes": {"type": "string", "description": "entry_id of an existing memory this replaces"},
                        "promoted_from": {"type": "string", "description": "suggestion_id if promoted from outlier suggestion"},
                    },
                    "required": ["text"],
                },
            ),
            types.Tool(
                name="kb_get_memories",
                description=(
                    "Get all active explicit memories for this project. "
                    "Call at session start to load project context."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="kb_get_suggestions",
                description=(
                    "Get pending memory suggestions flagged by outlier detection. "
                    "Present to user for confirmation or dismissal."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="kb_dismiss_suggestion",
                description="Dismiss a pending memory suggestion.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "suggestion_id": {"type": "string", "description": "ID of the suggestion to dismiss"},
                    },
                    "required": ["suggestion_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        handler = handlers[name]
        try:
            result = await handler(arguments)
        except ValueError as e:
            result = {"status": "error", "error": str(e)}
        except Exception as e:
            logger.exception("Unexpected error in tool '%s'", name)
            result = {"status": "error", "error": f"Unexpected error: {e}"}
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for dsagt-knowledge-server.

    All configuration comes from the project directory:
    - ``DSAGT_PROJECT_DIR`` env var (set by ``dsagt start``) → project path
    - ``<project>/dsagt_config.yaml`` → non-secret settings (chunk_size, vector_db, rerank)
    - ``LLM_API_KEY``, ``OPENAI_BASE_URL``, ``EMBEDDING_MODEL`` env vars → embedding credentials

    No CLI arguments — the server derives everything from its environment.
    """
    project_dir = Path(os.environ.get("DSAGT_PROJECT_DIR", "."))

    log_file = project_dir / "dsagt_knowledge_server.log"
    # Default INFO; users opt into DEBUG via DSAGT_LOG_LEVEL=DEBUG.  See
    # registry_server.py main() for rationale (httpcore/urllib3/llama_index
    # at DEBUG floods agent debug output).
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

    # Read project config.  Required — this server runs inside a project
    # created by dsagt init.  Every section must be present with all fields
    # filled.  dsagt init generates complete defaults; if anything is missing,
    # the config is broken and the server fails fast.
    config_path = project_dir / "dsagt_config.yaml"
    from dsagt.session import resolve_env_vars
    config = resolve_env_vars(yaml.safe_load(config_path.read_text()))

    kb_config = config["knowledge"]
    emb_config = config["embedding"]

    # Validate embedding credentials — we must not silently fall back to an
    # endpoint the user hasn't explicitly configured.
    model = emb_config["model"]
    base_url = emb_config["base_url"]
    api_key = emb_config["api_key"]

    if not base_url:
        raise ValueError(
            "embedding.base_url is required in dsagt_config.yaml. "
            "Set it to your OpenAI-compatible embedding endpoint."
        )
    if not api_key or api_key.startswith("${"):
        raise ValueError(
            "embedding.api_key is required in dsagt_config.yaml. "
            "Set it to your embedding API key, or export LLM_API_KEY."
        )

    from dsagt.observability import init_tracing, configure_litellm_retries
    init_tracing("dsagt-knowledge-server")  # session_id picked up from DSAGT_SESSION_ID env
    configure_litellm_retries()

    runtime_kb_dir = setup_runtime_kb(REGISTRY_DIR / "kb_index", project_dir)

    kb = KnowledgeBase(
        index_dir=runtime_kb_dir,
        chunk_size=kb_config["chunk_size"],
        default_rerank=kb_config["rerank"],
        default_embedder="api",
        default_index=kb_config["vector_db"],
        embedder_kwargs={"model": model, "base_url": base_url, "api_key": api_key},
    )

    server = create_knowledge_server(kb, runtime_dir=str(project_dir))
    try:
        asyncio.run(_run_stdio(server, "knowledge"))
    finally:
        kb.close()


if __name__ == "__main__":
    main()
