"""MCP tools for explicit memory.

User-confirmed facts that persist across sessions (``kb_remember`` /
``kb_get_memories``).  These front :mod:`dsagt.memory` (``ExplicitMemory``); the
``kb_`` tool-name prefix is historical (the tools were born in the knowledge
server) and is kept for agent-facing backward compatibility.

These definitions + handlers run inside the merged ``dsagt-server`` (see
:mod:`dsagt.mcp.server`); ``create_memory_server`` is retained only as a
test-facing constructor.
"""

import asyncio
import logging
from functools import partial
from pathlib import Path

import mcp.types as types

from dsagt.knowledge import KnowledgeBase
from dsagt.mcp.server import build_dispatch_server
from dsagt.memory import ExplicitMemory

logger = logging.getLogger(__name__)


async def _handle_kb_remember(
    arguments: dict,
    *,
    kb: KnowledgeBase,
    memory: ExplicitMemory,
) -> dict:
    text = arguments["text"]
    category = arguments.get("category", "")
    session_id = arguments.get("session_id", "")
    supersedes = arguments.get("supersedes")

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

    # Mirror into the VectorStore for semantic recall — optional infra.
    # The durable YAML write above already succeeded, so a mirror failure
    # degrades to pure-YAML explicit memory rather than failing the tool.
    try:
        await asyncio.to_thread(
            kb.add_entries,
            texts=[text],
            collection="session_memory",
            metadatas=[
                {
                    "source_type": "explicit_memory",
                    "category": category,
                    "session_id": session_id,
                }
            ],
        )
    except Exception as e:
        logger.warning(
            "kb_remember: stored YAML memory %s but vector mirror failed: %s",
            store_result["entry_id"],
            e,
        )

    return {
        "status": "ok",
        "entry_id": store_result["entry_id"],
        "superseded_id": store_result.get("superseded_id"),
        "total_memories": await asyncio.to_thread(memory.count),
    }


async def _handle_kb_get_memories(
    arguments: dict,
    *,
    memory: ExplicitMemory,
) -> dict:
    entries = await asyncio.to_thread(memory.get_all)
    return {"status": "ok", "count": len(entries), "memories": entries}


# ---------------------------------------------------------------------------
# Tool defs + handler map (used by the merged server and the test wrapper)
# ---------------------------------------------------------------------------


def _memory_tools_and_handlers(
    kb: KnowledgeBase,
    runtime_dir: str | Path | None = None,
):
    """Build the explicit-memory ``(tool defs, handler map)``.

    Combined with the other concern modules' tools under one MCP ``Server`` by
    :func:`dsagt.mcp.server.create_dsagt_server`.  ``ExplicitMemory`` is rooted
    at ``runtime_dir`` (falling back to the KB index's parent), matching the
    project's ``.dsagt`` memory location.
    """
    mem_dir = Path(runtime_dir) if runtime_dir else kb.index_dir.parent
    memory = ExplicitMemory(runtime_dir=mem_dir)

    handlers = {
        "kb_remember": partial(_handle_kb_remember, kb=kb, memory=memory),
        "kb_get_memories": partial(_handle_kb_get_memories, memory=memory),
    }

    tools = [
        types.Tool(
            name="kb_remember",
            description=(
                "Store a user-confirmed fact as an explicit memory. "
                "These persist across sessions. Use 'supersedes' to replace an outdated memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The fact to remember",
                    },
                    "category": {
                        "type": "string",
                        "description": "Classification tag",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Current session identifier",
                    },
                    "supersedes": {
                        "type": "string",
                        "description": "entry_id of an existing memory this replaces",
                    },
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
    ]
    return tools, handlers


def create_memory_server(
    kb: KnowledgeBase,
    runtime_dir: str | Path | None = None,
):
    """Create a standalone MCP server exposing only the explicit-memory tools.

    Test-facing API: tests call it with a mock KB and drive the server via
    ``call_tool_sync()``.  The merged ``dsagt-server`` uses
    :func:`_memory_tools_and_handlers` directly instead of this wrapper.
    """
    tools, handlers = _memory_tools_and_handlers(kb, runtime_dir)
    return build_dispatch_server(
        "memory", tools, handlers, {t: "memory" for t in handlers}
    )
