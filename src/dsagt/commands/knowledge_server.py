"""
DSAgt Knowledge Base MCP Server.

Provides semantic search over document collections for MCP-compatible agents.

At startup, symlinks base indexes into a session-specific runtime directory.
All modifications (ingestion, append) happen in the runtime copy.

Long-running operations (ingest, append) run in the background and return
immediately with a job_id. Use kb_job_status to poll for completion.

Embeddings are always generated via the OpenAI-compatible API.
The two things you can configure per-collection are:
  - embedding_model : which API model to use (e.g. "text-embedding-3-small")
  - vector_db       : which vector store to use (faiss, chroma, ...)

Usage:
    dsagt-knowledge-server
    dsagt-knowledge-server --base-index-dir ./kb_index --runtime-dir ./runtime
    dsagt-knowledge-server --rerank
    dsagt-knowledge-server --vector-db chroma --embedding-model text-embedding-3-large
"""

import argparse
import asyncio
import json
import logging
import os
import time

# Prevent fatal OpenMP crash when multiple libraries (FAISS, PyTorch/
# sentence-transformers) each bundle their own libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from dsagt.knowledge import EMBEDDER_REGISTRY, VECTORINDEX_REGISTRY, CollectionRoute, KnowledgeBase
from dsagt.memory import EPISODIC_MEMORY_ROUTE, SuggestionQueue
from dsagt.memory import ExplicitMemory

def create_server(name: str) -> Server:
    return Server(name)


async def run_stdio(server: Server, name: str) -> None:
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

logger = logging.getLogger(__name__)


def _text_result(data: dict) -> list[types.TextContent]:
    """
    Wrap *data* in a TextContent block -- the universally-compatible
    CallToolResult format required by all MCP clients.
    """
    return [types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]


def _collection_exists(path: Path) -> bool:
    """Return True if *path* looks like a persisted collection directory."""
    return (
        path.is_dir()
        and (
            (path / "index.faiss").exists()
            or (path / "chroma_ids.json").exists()
            or (path / "route.json").exists()
        )
    )


def _read_source_folder(coll_dir: Path) -> str | None:
    """
    Return the source folder path recorded in coll_dir/source.txt, or None.
    Written by kb.ingest() so we can detect re-ingests vs. genuine conflicts.
    """
    p = coll_dir / "source.txt"
    return p.read_text().strip() if p.exists() else None


def setup_runtime_kb(base_index_dir: Path, runtime_dir: Path) -> Path:
    """
    Symlink base indexes into a session-specific runtime directory.
    Pre-built collections are symlinked (read-only, zero-cost).
    New collections are written directly into the runtime copy.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)

    if base_index_dir.exists():
        for collection_dir in base_index_dir.iterdir():
            if _collection_exists(collection_dir):
                dest = runtime_kb_dir / collection_dir.name
                if not dest.exists():
                    dest.symlink_to(collection_dir.resolve())

    return runtime_kb_dir


def _register_external_collection(
    kb: KnowledgeBase,
    collection_name: str,
    vector_db: str,
    connection_params: dict,
    embedding_model: str,
    description: str,
) -> None:
    """
    Wire an already-built external vector store into the routing registry.
    A route.json is written so the mapping survives server restarts.
    No local index file is written -- queries are forwarded to the service.
    """
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
        embedding_backend="api",                    # always API
        vector_db=vector_db,
        embedder_kwargs={"model": embedding_model},
        index_kwargs=index_kwargs,
        description=description,
    )
    kb.register_route(collection_name, route)
    kb._save_route(collection_name, route)


def _build_where_clause(arguments: dict) -> dict | None:
    """Build a ChromaDB ``where`` clause from kb_search filter arguments.

    Only includes filters the agent actually passed. Returns None when no
    filters were provided (FAISS collections and unfiltered searches).
    """
    filters = []

    for key in ("category", "session_id", "source_type", "tool_name"):
        value = arguments.get(key)
        if value is not None:
            filters.append({key: value})

    # Numeric filter: return_code
    return_code = arguments.get("return_code")
    if return_code is not None:
        filters.append({"return_code": int(return_code)})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def create_knowledge_server(
    kb: KnowledgeBase,
    use_rerank: bool,
    runtime_dir: str | Path | None = None,
):
    """Create and configure the MCP server."""
    server = create_server("knowledge")

    # Background job tracking: job_id -> {status, result, error}
    jobs: dict[str, dict] = {}
    active_collections: set[str] = set()

    # Explicit memory store (file-backed, not vector-indexed)
    mem_dir = Path(runtime_dir) if runtime_dir else kb.index_dir.parent
    memory = ExplicitMemory(runtime_dir=mem_dir)

    # Outlier suggestion queue (file-backed)
    suggestions = SuggestionQueue(mem_dir / "suggestions.json")

    def _start_job(coro, collection: str | None = None) -> str:
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "collection": collection,
            "started_at": time.monotonic(),
            "message": "Starting -- embedding documents via API...",
        }
        if collection:
            active_collections.add(collection)

        async def _run():
            try:
                jobs[job_id]["message"] = "Embedding and indexing documents..."
                result = await coro
                jobs[job_id]["status"] = "complete"
                jobs[job_id]["result"] = result
                jobs[job_id]["message"] = "Done."
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
                jobs[job_id]["message"] = f"Failed: {type(e).__name__}: {e}"
                jobs[job_id]["traceback"] = tb
                logger.error("Job %s failed: %s\n%s", job_id, e, tb)
            finally:
                if collection:
                    active_collections.discard(collection)

        asyncio.get_event_loop().create_task(_run())
        return job_id

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
                            "description": "Use cross-encoder reranking (slower but more accurate). Only available when server started with --rerank.",
                            "default": use_rerank,
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category tag (ChromaDB collections only, e.g. episodic_memory, tool_executions)",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Filter by session ID (ChromaDB collections only)",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Filter by tool name (ChromaDB collections only, e.g. fastp, megahit)",
                        },
                        "source_type": {
                            "type": "string",
                            "description": "Filter by source type (ChromaDB collections only, e.g. extraction, compression)",
                        },
                        "return_code": {
                            "type": "integer",
                            "description": "Filter by tool exit code (ChromaDB collections only, e.g. 0 for success)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="kb_ingest",
                description=(
                    "Index a folder as a new knowledge base collection. "
                    "Embeddings are always generated via the API. "
                    "Returns immediately with a job_id. "
                    "IMPORTANT: poll kb_job_status every 10 seconds and wait for "
                    "status='complete'. DO NOT call ingest again for the same "
                    "folder while a job is running -- it will be rejected."
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
                            "description": "Embedding backend: 'api' (default) or 'local' (sentence-transformers, no network).",
                        },
                        "embedding_model": {
                            "type": "string",
                            "description": "Embedding model name (default: nomic-embed-text for api, or any sentence-transformers model for local).",
                        },
                        "vector_db": {
                            "type": "string",
                            "enum": list(VECTORINDEX_REGISTRY.keys()),
                            "description": "Vector database backend (default: server default).",
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
                            "description": "File extensions to include when expanding folders (e.g., ['pdf', 'md', 'py']). Defaults to common document types.",
                        },
                    },
                    "required": ["collection", "paths"],
                },
            ),
            types.Tool(
                name="kb_add_vector_db",
                description=(
                    "Register an already-built external vector store (ChromaDB, LanceDB, "
                    "Qdrant) as a collection. The index must already exist and be populated. "
                    "Queries will be embedded via the API using the specified model before "
                    "being dispatched to the backend. The route is persisted across restarts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Unique name for this collection",
                        },
                        "vector_db": {
                            "type": "string",
                            "enum": ["chroma", "lancedb", "qdrant"],
                            "description": "Vector store backend type",
                        },
                        "connection_params": {
                            "type": "object",
                            "description": (
                                "Backend-specific connection parameters. "
                                "chroma: {host, port, collection}. "
                                "lancedb: {uri, table}. "
                                "qdrant: {url, collection, api_key (optional)}."
                            ),
                        },
                        "embedding_model": {
                            "type": "string",
                            "description": "The API model used to build this index (must match exactly)",
                        },
                        "description": {
                            "type": "string",
                            "description": "Human-readable description for agent discovery",
                        },
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
                        "job_id": {
                            "type": "string",
                            "description": "Job ID returned by kb_ingest or kb_append",
                        },
                    },
                    "required": ["job_id"],
                },
            ),
            types.Tool(
                name="kb_remember",
                description=(
                    "Store a user-confirmed fact as an explicit memory. "
                    "These are important facts about the project pipeline "
                    "that persist across sessions. All active memories are "
                    "available at session start via kb_get_memories. "
                    "Use 'supersedes' to replace an outdated memory."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The fact to remember (short declarative statement)",
                        },
                        "category": {
                            "type": "string",
                            "description": "Classification tag (e.g. quality_control, configuration, data_stats)",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Current session identifier",
                        },
                        "supersedes": {
                            "type": "string",
                            "description": "entry_id of an existing memory this fact replaces",
                        },
                        "promoted_from": {
                            "type": "string",
                            "description": "suggestion_id if this memory was promoted from an outlier suggestion",
                        },
                    },
                    "required": ["text"],
                },
            ),
            types.Tool(
                name="kb_get_memories",
                description=(
                    "Get all active explicit memories for this project. "
                    "Call this at the start of each session to load project "
                    "context the user has previously confirmed as important. "
                    "Also returns any pending suggestions from the memory system "
                    "that should be reviewed with the user."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="kb_get_suggestions",
                description=(
                    "Get pending memory suggestions flagged by the outlier "
                    "detection system. These are observations that seem unusual "
                    "or important. Present them to the user and ask if they should "
                    "be remembered. Use kb_remember with promoted_from to store "
                    "confirmed suggestions, or kb_dismiss_suggestion to discard."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="kb_dismiss_suggestion",
                description="Dismiss a pending memory suggestion the user does not want to remember.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "suggestion_id": {
                            "type": "string",
                            "description": "ID of the suggestion to dismiss",
                        },
                    },
                    "required": ["suggestion_id"],
                },
            ),
        ]


    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "kb_list_collections":
                collections = kb.list_collections()
                result = {
                    "status": "ok",
                    "collections": collections,
                    "count": len(collections),
                }

            elif name == "kb_search":
                query = arguments["query"]
                top_k = arguments.get("top_k", 5)
                rerank = arguments.get("rerank", use_rerank)
                where = _build_where_clause(arguments)

                collection_arg = arguments.get("collection")
                collections_arg = arguments.get("collections")

                if not collection_arg and not collections_arg:
                    result = {
                        "status": "error",
                        "error": "Provide 'collection' or 'collections'",
                    }
                else:
                    target_collections = collections_arg or [collection_arg]
                    all_results = []
                    search_errors = []

                    for coll_name in target_collections:
                        try:
                            search_kwargs = dict(
                                query=query,
                                collection=coll_name,
                                top_k=top_k,
                                rerank=rerank,
                            )
                            if where is not None:
                                search_kwargs["where"] = where
                            coll_results = await asyncio.to_thread(
                                kb.search, **search_kwargs,
                            )
                            all_results.extend(coll_results)
                        except ValueError as e:
                            logger.warning("Search failed for '%s': %s", coll_name, e)
                            search_errors.append(str(e))

                    # Report error if every collection failed
                    if search_errors and not all_results:
                        if len(target_collections) == 1:
                            result = {"status": "error", "error": search_errors[0]}
                        else:
                            result = {
                                "status": "error",
                                "error": f"All collections failed: {'; '.join(search_errors)}",
                            }
                    else:
                        # Sort merged results by score, truncate
                        score_key = "rerank_score" if rerank else "score"
                        all_results.sort(
                            key=lambda r: r.get(score_key, r["score"]),
                            reverse=True,
                        )
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

            elif name == "kb_ingest":
                folder_path = Path(arguments["folder_path"])
                collection_name = arguments.get("collection_name")
                file_types = arguments.get("file_types")
                embedding_backend = arguments.get("embedding_backend")
                embedding_model = arguments.get("embedding_model")
                vector_db = arguments.get("vector_db")

                if not folder_path.exists():
                    result = {"status": "error", "error": f"Folder not found: {folder_path}"}
                elif not folder_path.is_dir():
                    result = {"status": "error", "error": f"Not a directory: {folder_path}"}
                else:
                    # Deconflict collection name if it already exists
                    target_name = collection_name or folder_path.name
                    warning = None

                    if target_name in active_collections:
                        result = {
                            "status": "error",
                            "error": (
                                f"Collection '{target_name}' is already being ingested. "
                                f"Poll kb_job_status for progress."
                            ),
                        }
                    else:
                        # Only rename when an existing finished collection came from
                        # a different source folder.
                        if _collection_exists(kb.index_dir / target_name):
                            existing_source = _read_source_folder(kb.index_dir / target_name)
                            same_source = (
                                existing_source is None
                                or Path(existing_source).resolve() == folder_path.resolve()
                            )
                            if not same_source:
                                original_name = target_name
                                n = 1
                                while (
                                    _collection_exists(kb.index_dir / target_name)
                                    or target_name in active_collections
                                ):
                                    target_name = f"{original_name}{n}"
                                    n += 1
                                warning = (
                                    f"Collection '{original_name}' already exists from a "
                                    f"different folder; using '{target_name}'."
                                )

                        # Build a route only when the caller overrides the server default.
                        # Always inherit the default model so embedder_kwargs is never empty —
                        # an empty dict means the APIEmbeddingClient falls back to its own
                        # env-var default instead of the server-configured model.
                        route = None
                        if embedding_backend or embedding_model or vector_db:
                            default = kb._default_route
                            inherited_model = (
                                embedding_model
                                or default.embedder_kwargs.get("model")
                            )
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
                            logger.info(
                                "Ingest starting: collection=%s folder=%s kwargs=%s",
                                target_name, folder_path, ingest_kwargs,
                            )
                            try:
                                result = await asyncio.to_thread(
                                    kb.ingest, folder_path, **ingest_kwargs
                                )
                                logger.info("Ingest complete: %s", result)
                                return result
                            except Exception as _e:
                                logger.error(
                                    "Ingest FAILED: %s\n%s",
                                    _e, _tb.format_exc(),
                                )
                                raise

                        job_id = _start_job(
                            _ingest_with_logging(),
                            collection=target_name,
                        )
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

            elif name == "kb_append":
                collection = arguments["collection"]
                paths = arguments["paths"]
                if isinstance(paths, str):
                    paths = [paths]
                file_types = arguments.get("file_types")

                if not _collection_exists(kb.index_dir / collection):
                    result = {"status": "error", "error": f"Collection '{collection}' not found"}
                else:
                    append_kwargs: dict = {}
                    if file_types:
                        append_kwargs["file_types"] = file_types

                    job_id = _start_job(
                        asyncio.to_thread(kb.append, collection, paths, **append_kwargs),
                        collection=collection,
                    )
                    result = {
                        "status": "started",
                        "job_id": job_id,
                        "collection": collection,
                        "message": f"Append started. Poll kb_job_status(job_id='{job_id}') for progress.",
                    }

            elif name == "kb_add_vector_db":
                collection_name = arguments["collection_name"]
                vector_db = arguments["vector_db"]
                connection_params = arguments["connection_params"]
                embedding_model = arguments["embedding_model"]
                description = arguments.get("description", "")

                if (kb.index_dir / collection_name).exists():
                    result = {
                        "status": "error",
                        "error": (
                            f"Collection '{collection_name}' already exists. "
                            "Choose a different name or delete the existing collection."
                        ),
                    }
                else:
                    await asyncio.to_thread(
                        _register_external_collection,
                        kb, collection_name, vector_db,
                        connection_params, embedding_model, description,
                    )
                    result = {
                        "status": "ok",
                        "collection": collection_name,
                        "vector_db": vector_db,
                        "embedding_model": embedding_model,
                        "message": (
                            f"External collection '{collection_name}' registered. "
                            "Use search to query it."
                        ),
                    }

            # job_status
            elif name == "kb_job_status":
                job_id = arguments["job_id"]
                if job_id not in jobs:
                    result = {"status": "error", "error": f"Unknown job: {job_id}"}
                else:
                    job = jobs[job_id]
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

            elif name == "kb_remember":
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

                if store_result.get("stored"):
                    # Mirror to ChromaDB for future similarity queries
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

                    # Auto-dismiss the suggestion if promoted
                    if promoted_from:
                        suggestions.dismiss(promoted_from)

                    result = {
                        "status": "ok",
                        "entry_id": store_result["entry_id"],
                        "superseded_id": store_result.get("superseded_id"),
                        "promoted_from": promoted_from,
                        "total_memories": await asyncio.to_thread(memory.count),
                    }
                else:
                    result = {
                        "status": "error",
                        "error": store_result.get("error", "Failed to store memory"),
                    }

            elif name == "kb_get_memories":
                entries = await asyncio.to_thread(memory.get_all)
                pending = suggestions.get_all()
                result = {
                    "status": "ok",
                    "count": len(entries),
                    "memories": entries,
                }
                if pending:
                    result["suggestions"] = pending
                    result["suggestion_count"] = len(pending)

            elif name == "kb_get_suggestions":
                pending = suggestions.get_all()
                result = {
                    "status": "ok",
                    "count": len(pending),
                    "suggestions": pending,
                }

            elif name == "kb_dismiss_suggestion":
                suggestion_id = arguments["suggestion_id"]
                dismissed = suggestions.dismiss(suggestion_id)
                if dismissed:
                    result = {
                        "status": "ok",
                        "dismissed": suggestion_id,
                        "remaining": suggestions.count,
                    }
                else:
                    result = {
                        "status": "error",
                        "error": f"Suggestion not found: {suggestion_id}",
                    }

            else:
                result = {"status": "error", "error": f"Unknown tool: {name}"}

        except ValueError as e:
            result = {"status": "error", "error": str(e)}
        except Exception as e:
            logger.exception("Unexpected error in tool '%s'", name)
            result = {"status": "error", "error": f"Unexpected error: {e}"}

        return _text_result(result)

    return server



async def run_server(
    base_index_dir: str,
    runtime_dir: str,
    chunk_size: int,
    use_rerank: bool,
    embedding_backend: str,
    embedding_model: str | None,
    vector_db: str,
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
):
    """Run the MCP server with session-isolated indexes."""
    base_path = Path(base_index_dir)
    runtime_path = Path(runtime_dir)

    runtime_kb_dir = setup_runtime_kb(base_path, runtime_path)

    # Build embedder kwargs from CLI args — config flows here via
    # _mcp_server_args() in session.py, which reads dsagt_config.yaml.
    embedder_kwargs = {}
    if embedding_model:
        embedder_kwargs["model"] = embedding_model
    if embedding_base_url:
        embedder_kwargs["base_url"] = embedding_base_url
    if embedding_api_key:
        embedder_kwargs["api_key"] = embedding_api_key

    kb = KnowledgeBase(
        index_dir=runtime_kb_dir,
        chunk_size=chunk_size,
        default_embedder=embedding_backend,
        default_index=vector_db,
        embedder_kwargs=embedder_kwargs,
    )

    server = create_knowledge_server(kb, use_rerank, runtime_dir=runtime_dir)
    try:
        await run_stdio(server, "knowledge")
    finally:
        kb.close()


def main():
    # File-based logging so background thread errors are visible independently
    # of what the MCP client (Goose) surfaces. Check this file when jobs fail.
    log_file = Path("dsagt_knowledge_server.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),          # also goes to stderr
        ],
    )
    logger.info("Server starting — log file: %s", log_file.resolve())

    parser = argparse.ArgumentParser(description="DSAgt Knowledge Base MCP Server")
    parser.add_argument(
        "--base-index-dir",
        default="./kb_index",
        help="Base directory containing pre-built indexes (default: ./kb_index)",
    )
    parser.add_argument(
        "--runtime-dir",
        default="./runtime",
        help="Runtime directory for session-specific data (default: ./runtime)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for text splitting (default: 1024)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (downloads model on first use)",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=list(EMBEDDER_REGISTRY.keys()),
        default="api",
        help="Default embedding backend: 'api' (default) or 'local' (sentence-transformers).",
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="Default embedding model name (default: nomic-embed-text). Applies to whichever backend is selected.",
    )
    parser.add_argument(
        "--embedding-base-url", default=None,
        help="Base URL for the embedding API (OpenAI-compatible).",
    )
    parser.add_argument(
        "--embedding-api-key", default=None,
        help="API key for the embedding service.",
    )
    parser.add_argument(
        "--vector-db",
        choices=list(VECTORINDEX_REGISTRY.keys()),
        default="faiss",
        help="Default vector database backend for new collections (default: faiss)",
    )
    args = parser.parse_args()

    asyncio.run(run_server(
        args.base_index_dir,
        args.runtime_dir,
        args.chunk_size,
        args.rerank,
        args.embedding_backend,
        args.embedding_model,
        args.vector_db,
        args.embedding_api_key,
        args.embedding_base_url,
    ))


if __name__ == "__main__":
    main()
