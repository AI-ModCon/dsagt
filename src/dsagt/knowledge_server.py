"""
DSAGT Knowledge Base MCP Server

Provides semantic search over document collections for MCP-compatible agents.

At startup, copies base indexes to a session-specific runtime directory.
All modifications (ingestion) happen in the runtime copy.

Usage:
    dsagt-knowledge-server
    dsagt-knowledge-server --base-index-dir ./kb_index --runtime-dir ./runtime
    dsagt-knowledge-server --no-rerank
"""

import argparse
import asyncio
from pathlib import Path

from dsagt.knowledge import KnowledgeBase
from dsagt.mcp_utils import create_server, run_stdio, text_result, types


def setup_runtime_kb(base_index_dir: Path, runtime_dir: Path) -> Path:
    """
    Symlink base indexes into runtime directory for session isolation.
    
    Pre-built collections are symlinked (read-only, zero-cost).
    New collections created via kb_ingest are written directly to runtime.
    
    Returns the runtime index directory path.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)
    
    # Symlink existing collections from base to runtime
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
                            "description": "Use cross-encoder reranking (slower but more accurate)",
                            "default": True,
                        },
                    },
                    "required": ["query", "collection"],
                },
            ),
            types.Tool(
                name="kb_ingest",
                description="Index a folder as a new collection. By default the folder name becomes the collection name. Include a DESCRIPTION.md file in the folder for agent discovery.",
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

                # Run in thread — kb.search makes blocking HTTP calls
                results = await asyncio.to_thread(
                    kb.search,
                    query=query,
                    collection=collection,
                    top_k=top_k,
                    rerank=rerank,
                )

                # Format results for readability
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

                    # Build kwargs — only pass collection_name if overridden
                    kwargs = {}
                    if target_name != folder_path.name:
                        kwargs["collection_name"] = target_name
                    if file_types:
                        kwargs["file_types"] = file_types

                    # Run in thread — kb.ingest makes blocking HTTP calls
                    ingest_result = await asyncio.to_thread(
                        kb.ingest, folder_path, **kwargs,
                    )
                    result = {
                        "status": "ok",
                        "collection": ingest_result["collection"],
                        "files_indexed": ingest_result["files"],
                        "chunks_created": ingest_result["chunks"],
                    }
                    if warning:
                        result["warning"] = warning

            else:
                result = {"status": "error", "error": f"Unknown tool: {name}"}

        except ValueError as e:
            result = {"status": "error", "error": str(e)}
        except Exception as e:
            result = {"status": "error", "error": f"Unexpected error: {e}"}

        return text_result(result)

    return server


async def run_server(base_index_dir: str, runtime_dir: str, chunk_size: int, use_rerank: bool):
    """Run the MCP server with session-isolated indexes."""
    base_path = Path(base_index_dir)
    runtime_path = Path(runtime_dir)
    
    # Copy base indexes to runtime directory
    runtime_kb_dir = setup_runtime_kb(base_path, runtime_path)
    
    # Create KB pointing to runtime directory
    kb = KnowledgeBase(index_dir=runtime_kb_dir, chunk_size=chunk_size)
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
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking by default",
    )
    args = parser.parse_args()

    asyncio.run(run_server(
        args.base_index_dir,
        args.runtime_dir,
        args.chunk_size,
        not args.no_rerank,
    ))


if __name__ == "__main__":
    main()
