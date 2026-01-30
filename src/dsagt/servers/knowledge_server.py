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
import json
import shutil
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from dsagt.knowledge import KnowledgeBase


def setup_runtime_kb(base_index_dir: Path, runtime_dir: Path) -> Path:
    """
    Copy base indexes to runtime directory for session isolation.
    
    Returns the runtime index directory path.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy existing collections from base to runtime
    if base_index_dir.exists():
        for collection_dir in base_index_dir.iterdir():
            if collection_dir.is_dir() and (collection_dir / "index.faiss").exists():
                dest = runtime_kb_dir / collection_dir.name
                if not dest.exists():
                    shutil.copytree(collection_dir, dest)
    
    return runtime_kb_dir


def create_server(kb: KnowledgeBase, use_rerank: bool) -> Server:
    """Create and configure the MCP server."""
    server = Server("dsagt-knowledge")

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
                description="Index a folder as a new collection. The folder name becomes the collection name. Include a DESCRIPTION.md file in the folder for agent discovery.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_path": {
                            "type": "string",
                            "description": "Path to folder containing documents to index",
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

                results = kb.search(
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
                from pathlib import Path

                folder_path = Path(arguments["folder_path"])
                file_types = arguments.get("file_types")

                if not folder_path.exists():
                    result = {"status": "error", "error": f"Folder not found: {folder_path}"}
                elif not folder_path.is_dir():
                    result = {"status": "error", "error": f"Not a directory: {folder_path}"}
                else:
                    ingest_result = kb.ingest(folder_path, file_types=file_types)
                    result = {
                        "status": "ok",
                        "collection": ingest_result["collection"],
                        "files_indexed": ingest_result["files"],
                        "chunks_created": ingest_result["chunks"],
                    }

            else:
                result = {"status": "error", "error": f"Unknown tool: {name}"}

        except ValueError as e:
            result = {"status": "error", "error": str(e)}
        except Exception as e:
            result = {"status": "error", "error": f"Unexpected error: {e}"}

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


async def run_server(base_index_dir: str, runtime_dir: str, chunk_size: int, use_rerank: bool):
    """Run the MCP server with session-isolated indexes."""
    base_path = Path(base_index_dir)
    runtime_path = Path(runtime_dir)
    
    # Copy base indexes to runtime directory
    runtime_kb_dir = setup_runtime_kb(base_path, runtime_path)
    
    # Create KB pointing to runtime directory
    kb = KnowledgeBase(index_dir=runtime_kb_dir, chunk_size=chunk_size)
    server = create_server(kb, use_rerank)

    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="dsagt-knowledge",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
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
        not args.no_rerank
    ))


if __name__ == "__main__":
    main()
