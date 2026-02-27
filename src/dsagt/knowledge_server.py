"""
DSAGT Knowledge Base MCP Server

Provides semantic search over document collections for MCP-compatible agents.

At startup, symlinks base indexes into a session-specific runtime directory.
All modifications (ingestion, append) happen in the runtime copy.

Long-running operations (ingest, append) run in the background and return
immediately with a job_id. Use kb_job_status to poll for completion.

Usage:
    dsagt-knowledge-server
    dsagt-knowledge-server --base-index-dir ./kb_index --runtime-dir ./runtime
    dsagt-knowledge-server --rerank
"""

import argparse
import asyncio
import logging
import os

# Prevent fatal OpenMP crash when multiple libraries (FAISS, PyTorch/
# sentence-transformers) each bundle their own libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import uuid
from pathlib import Path

from dsagt.knowledge import KnowledgeBase
from dsagt.mcp_utils import create_server, run_stdio, text_result, types

logger = logging.getLogger(__name__)


def setup_runtime_kb(base_index_dir: Path, runtime_dir: Path) -> Path:
    """
    Symlink base indexes into runtime directory for session isolation.

    Pre-built collections are symlinked (read-only, zero-cost).
    New collections created via kb_ingest are written directly to runtime.

    Returns the runtime index directory path.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)

    if base_index_dir.exists():
        for collection_dir in base_index_dir.iterdir():
            if collection_dir.is_dir() and (collection_dir / "index.faiss").exists():
                dest = runtime_kb_dir / collection_dir.name
                if not dest.exists():
                    dest.symlink_to(collection_dir.resolve())

    return runtime_kb_dir


def create_knowledge_server(kb: KnowledgeBase, use_rerank: bool):
    """Create and configure the MCP server."""
    server = create_server("knowledge")

    # Background job tracking: job_id -> {status, result, error}
    jobs: dict[str, dict] = {}

    def _start_job(coro) -> str:
        """Launch a coroutine as a background task, return job_id."""
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {"status": "running", "result": None, "error": None}

        async def _run():
            try:
                result = await coro
                jobs[job_id]["status"] = "complete"
                jobs[job_id]["result"] = result
            except Exception as e:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
                logger.error("Job %s failed: %s", job_id, e)

        asyncio.get_event_loop().create_task(_run())
        return job_id

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="kb_list_collections",
                description="List all available knowledge base collections with descriptions. Use this to discover what documentation is indexed.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="kb_search",
                description="Search a knowledge base collection using semantic similarity. Returns relevant document chunks with source metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "collection": {
                            "type": "string",
                            "description": "Name of the collection to search (use kb_list_collections to see options)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                        },
                        "rerank": {
                            "type": "boolean",
                            "description": "Use cross-encoder reranking (slower but more accurate). Only available when server started with --rerank.",
                            "default": use_rerank,
                        },
                    },
                    "required": ["query", "collection"],
                },
            ),
            types.Tool(
                name="kb_ingest",
                description=(
                    "Index a folder as a new collection. Returns immediately with a "
                    "job_id -- use kb_job_status to check progress. By default the "
                    "folder name becomes the collection name. Include a DESCRIPTION.md "
                    "file in the folder for agent discovery."
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
                            "description": "File extensions to include (e.g., ['pdf', 'md', 'py']). Defaults to common document types.",
                        },
                    },
                    "required": ["folder_path"],
                },
            ),
            types.Tool(
                name="kb_append",
                description=(
                    "Add documents to an existing collection. Returns immediately "
                    "with a job_id -- use kb_job_status to check progress. Accepts "
                    "file paths and/or folder paths."
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
                collection = arguments["collection"]
                top_k = arguments.get("top_k", 5)
                rerank = arguments.get("rerank", use_rerank)

                results = await asyncio.to_thread(
                    kb.search,
                    query=query,
                    collection=collection,
                    top_k=top_k,
                    rerank=rerank,
                )

                formatted = []
                for r in results:
                    formatted.append({
                        "text": r["chunk"]["text"],
                        "score": r["score"],
                        "rerank_score": r.get("rerank_score"),
                        "source_file": r["chunk"]["metadata"]["source_file"],
                        "chunk_index": r["chunk"]["metadata"]["chunk_index"],
                    })

                result = {
                    "status": "ok",
                    "query": query,
                    "collection": collection,
                    "result_count": len(formatted),
                    "results": formatted,
                }

            elif name == "kb_ingest":
                folder_path = Path(arguments["folder_path"])
                collection_name = arguments.get("collection_name")
                file_types = arguments.get("file_types")

                if not folder_path.exists():
                    result = {"status": "error", "error": f"Folder not found: {folder_path}"}
                elif not folder_path.is_dir():
                    result = {"status": "error", "error": f"Not a directory: {folder_path}"}
                else:
                    # Deconflict collection name if it already exists
                    target_name = collection_name or folder_path.name
                    coll_dest = kb.index_dir / target_name
                    if coll_dest.exists() or coll_dest.is_symlink():
                        original_name = target_name
                        n = 1
                        while coll_dest.exists() or coll_dest.is_symlink():
                            target_name = f"{original_name}{n}"
                            coll_dest = kb.index_dir / target_name
                            n += 1
                        warning = f"Collection '{original_name}' already exists, using '{target_name}'"
                    else:
                        warning = None

                    kwargs = {}
                    if target_name != folder_path.name:
                        kwargs["collection_name"] = target_name
                    if file_types:
                        kwargs["file_types"] = file_types

                    job_id = _start_job(
                        asyncio.to_thread(kb.ingest, folder_path, **kwargs)
                    )
                    result = {
                        "status": "started",
                        "job_id": job_id,
                        "collection": target_name,
                        "message": f"Ingestion started. Use kb_job_status(job_id='{job_id}') to check progress.",
                    }
                    if warning:
                        result["warning"] = warning

            elif name == "kb_append":
                collection = arguments["collection"]
                paths = arguments["paths"]
                if isinstance(paths, str):
                    paths = [paths]
                file_types = arguments.get("file_types")

                coll_dir = kb.index_dir / collection
                if not (coll_dir / "index.faiss").exists():
                    result = {"status": "error",
                              "error": f"Collection '{collection}' not found"}
                else:
                    kwargs = {}
                    if file_types:
                        kwargs["file_types"] = file_types

                    job_id = _start_job(
                        asyncio.to_thread(kb.append, collection, paths, **kwargs)
                    )
                    result = {
                        "status": "started",
                        "job_id": job_id,
                        "collection": collection,
                        "message": f"Append started. Use kb_job_status(job_id='{job_id}') to check progress.",
                    }

            elif name == "kb_job_status":
                job_id = arguments["job_id"]
                if job_id not in jobs:
                    result = {"status": "error", "error": f"Unknown job: {job_id}"}
                else:
                    job = jobs[job_id]
                    result = {"status": job["status"]}
                    if job["result"] is not None:
                        result["result"] = job["result"]
                    if job["error"] is not None:
                        result["error"] = job["error"]

            else:
                result = {"status": "error", "error": f"Unknown tool: {name}"}

        except ValueError as e:
            result = {"status": "error", "error": str(e)}
        except Exception as e:
            result = {"status": "error", "error": f"Unexpected error: {e}"}

        return text_result(result)

    return server


async def run_server(
    base_index_dir: str,
    runtime_dir: str,
    chunk_size: int,
    use_rerank: bool,
    embedding_backend: str,
    embedding_model: str | None,
):
    """Run the MCP server with session-isolated indexes."""
    base_path = Path(base_index_dir)
    runtime_path = Path(runtime_dir)

    runtime_kb_dir = setup_runtime_kb(base_path, runtime_path)

    kb = KnowledgeBase(
        index_dir=runtime_kb_dir,
        chunk_size=chunk_size,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )
    server = create_knowledge_server(kb, use_rerank)

    try:
        await run_stdio(server, "knowledge")
    finally:
        kb.close()


def main():
    parser = argparse.ArgumentParser(description="DSAGT Knowledge Base MCP Server")
    parser.add_argument(
        "--base-index-dir",
        default="./kb_index",
        help="Base directory containing source FAISS indexes (default: ./kb_index)",
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
        choices=["local", "api"],
        default="api",
        help="Embedding backend: 'local' (sentence-transformers) or 'api' (OpenAI-compatible). Default: api",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model name (default: BAAI/bge-base-en-v1.5 for local, text-embedding-3-small-project for api)",
    )
    args = parser.parse_args()

    asyncio.run(run_server(
        args.base_index_dir,
        args.runtime_dir,
        args.chunk_size,
        args.rerank,
        args.embedding_backend,
        args.embedding_model,
    ))


if __name__ == "__main__":
    main()
