"""
Tool execution record indexing for semantic search.

Reads tool execution records (JSON files in trace_archive/) and indexes
them into a ChromaDB-backed ``tool_executions`` collection in the knowledge
base.  Each record becomes a searchable entry with metadata for filtering
by session, tool, outcome, and time.

Two record types exist on disk:

  **Proxy records** (from proxy_callback.py):
    ``{record_id, tool_name, session_id, intent, execution: None, report}``
    Have intent + report layers.  execution is None (wrapper wasn't involved).

  **Wrapper records** (from run.py / dsagt-run):
    ``{record_id, tool_name, session_id, execution}``
    Have execution layer only.  No intent or report keys.

Both share top-level ``record_id``, ``tool_name``, ``session_id`` fields.
The text representation uses whichever layers are present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dsagt.knowledge import CollectionRoute, KnowledgeBase

logger = logging.getLogger(__name__)

COLLECTION_NAME = "tool_executions"

# ChromaDB route for the tool_executions collection
TOOL_EXECUTIONS_ROUTE = CollectionRoute(
    embedding_backend="api",
    vector_db="chroma",
    description="Indexed tool execution records from trace_archive.",
)


def render_execution_text(record: dict) -> str:
    """Convert a tool execution record into embeddable natural-language text.

    Handles both proxy records (intent + report) and wrapper records
    (execution only).  Uses whichever layers are present.
    """
    # Top-level fields are the canonical source for tool_name/session_id
    tool_name = record.get("tool_name", "unknown")
    intent = record.get("intent") or {}
    execution = record.get("execution")
    report = record.get("report") or {}

    parts = [f"Tool: {tool_name}"]

    # Command: prefer exact command from wrapper, fall back to intent parameters
    if execution and execution.get("exact_command"):
        cmd = execution["exact_command"]
        # exact_command is a list from subprocess — join for display
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        parts.append(f"Command: {cmd}")
    elif intent.get("parameters"):
        parts.append(f"Parameters: {json.dumps(intent['parameters'])}")

    # Outcome: prefer wrapper return code, fall back to agent report
    if execution and execution.get("return_code") is not None:
        rc = execution["return_code"]
        status = "succeeded" if rc == 0 else f"failed (exit code {rc})"
        parts.append(f"Outcome: {status}")
    elif report.get("agent_output"):
        output = report["agent_output"]
        if len(output) > 500:
            output = output[:500] + "..."
        parts.append(f"Agent report: {output}")

    # Timing from wrapper
    if execution:
        start = execution.get("timestamp_start", "")
        end = execution.get("timestamp_end", "")
        if start and end:
            parts.append(f"Duration: {start} to {end}")

    # Files from wrapper
    if execution and execution.get("input_files"):
        parts.append(f"Input files: {', '.join(execution['input_files'])}")
    if execution and execution.get("output_files"):
        parts.append(f"Output files: {', '.join(execution['output_files'])}")

    # Stderr summary (useful for search — errors, warnings)
    if execution and execution.get("stderr"):
        stderr = execution["stderr"].strip()
        if stderr:
            if len(stderr) > 300:
                stderr = stderr[:300] + "..."
            parts.append(f"Stderr: {stderr}")

    return "\n".join(parts)


def execution_metadata(record: dict) -> dict:
    """Extract ChromaDB-filterable metadata from a tool execution record.

    All values are strings or ints (ChromaDB metadata constraints).
    Uses top-level fields as canonical source, with layer fields as fallback.
    """
    execution = record.get("execution")
    intent = record.get("intent") or {}

    meta: dict = {}

    # Top-level fields are canonical (both proxy and wrapper records have them)
    meta["tool_name"] = record.get("tool_name", intent.get("command", "unknown"))
    meta["session_id"] = record.get("session_id", intent.get("session_id", "unknown"))

    if execution and execution.get("return_code") is not None:
        meta["return_code"] = execution["return_code"]

    meta["wrapper_used"] = 1 if execution is not None else 0

    # Timestamp: prefer execution start, fall back to intent timestamp
    timestamp = None
    if execution and execution.get("timestamp_start"):
        timestamp = execution["timestamp_start"]
    elif intent.get("timestamp_requested"):
        timestamp = intent["timestamp_requested"]
    if timestamp:
        meta["timestamp"] = timestamp

    # Record ID for idempotent indexing
    record_id = record.get("record_id", "")
    if record_id:
        meta["record_id"] = record_id

    return meta


def index_execution_record(
    record: dict,
    kb: KnowledgeBase,
) -> dict:
    """Render and store a single tool execution record in the knowledge base."""
    text = render_execution_text(record)
    metadata = execution_metadata(record)

    return kb.add_entries(
        texts=[text],
        collection=COLLECTION_NAME,
        metadatas=[metadata],
        route=TOOL_EXECUTIONS_ROUTE,
    )


def index_trace_archive(
    trace_dir: Path,
    kb: KnowledgeBase,
    indexed_ids: set[str] | None = None,
) -> dict:
    """Batch-index all tool execution records in a trace archive directory.

    Parameters
    ----------
    trace_dir : Path
        Directory containing tool execution JSON files.
    kb : KnowledgeBase
        Knowledge base to index into.
    indexed_ids : set[str], optional
        Set of record IDs already indexed.  Records with matching
        record_id are skipped.  Updated in-place with newly indexed IDs.

    Returns
    -------
    dict
        ``{indexed, skipped, errors, total_files}``
    """
    if indexed_ids is None:
        indexed_ids = set()

    if not trace_dir.is_dir():
        return {"indexed": 0, "skipped": 0, "errors": 0, "total_files": 0}

    json_files = sorted(trace_dir.glob("*.json"))
    if not json_files:
        return {"indexed": 0, "skipped": 0, "errors": 0, "total_files": 0}

    texts = []
    metadatas = []
    skipped = 0
    errors = 0

    for path in json_files:
        raw = path.read_text()
        record = json.loads(raw)

        record_id = record.get("record_id", "")
        if record_id and record_id in indexed_ids:
            skipped += 1
            continue

        # A valid record has at least an intent layer (proxy) or execution layer (wrapper)
        has_intent = "intent" in record
        has_execution = record.get("execution") is not None
        if not has_intent and not has_execution:
            logger.warning("Skipping %s: no intent or execution layer", path.name)
            errors += 1
            continue

        texts.append(render_execution_text(record))
        metadatas.append(execution_metadata(record))

        if record_id:
            indexed_ids.add(record_id)

    if texts:
        kb.add_entries(
            texts=texts,
            collection=COLLECTION_NAME,
            metadatas=metadatas,
            route=TOOL_EXECUTIONS_ROUTE,
        )

    return {
        "indexed": len(texts),
        "skipped": skipped,
        "errors": errors,
        "total_files": len(json_files),
    }
