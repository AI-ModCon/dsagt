"""
Memory management for DSAgt projects.

Two memory types:

**Explicit memory** (ExplicitMemory class):
    User-confirmed facts stored in a YAML file and mirrored into the
    ``explicit_memory`` ChromaDB collection.  Not auto-loaded into the
    agent's context at session start — the agent retrieves entries on
    demand via the ``kb_get_memories`` / ``kb_search`` MCP tools.
    Supports remember, supersede, remove, and retrieval.

**Episodic memory** (MemoryExtractor):
    A trace-pipeline *consumer* (``MemoryExtractor``) that consumes the
    in-process ``Trace`` the heartbeat produces and writes per-block chunks into
    the ``session_memory`` collection (producer/tool/turn_id metadata, no LLM).

Files on disk (in project directory):
  explicit_memories.yaml       — active user-confirmed facts
  explicit_memories_history.yaml — superseded/removed entries
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from dsagt.knowledge import KnowledgeBase

if TYPE_CHECKING:
    from dsagt.traces import Trace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explicit memory (YAML store)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id(text: str, timestamp: str) -> str:
    return hashlib.sha256(f"{text}:{timestamp}".encode()).hexdigest()[:16]


class ExplicitMemory:
    """File-backed store for explicit (user-confirmed) facts."""

    FILENAME = "explicit_memories.yaml"
    HISTORY_FILENAME = "explicit_memories_history.yaml"

    def __init__(self, runtime_dir: str | Path):
        self._dir = Path(runtime_dir)
        self._path = self._dir / self.FILENAME
        self._history_path = self._dir / self.HISTORY_FILENAME

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        text = self._path.read_text()
        if not text.strip():
            return []
        data = yaml.safe_load(text)
        if not isinstance(data, list):
            # A non-list payload means the file is corrupt or hand-edited.
            # Raising keeps the next _save from silently overwriting (and
            # losing) its real contents.
            raise ValueError(
                f"{self._path} is not a list of memories "
                f"(got {type(data).__name__}); inspect and fix the file."
            )
        return data

    def _save(self, entries: list[dict]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            yaml.dump(entries, default_flow_style=False, sort_keys=False)
            if entries
            else ""
        )

    def _append_history(self, entry: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        history = []
        if self._history_path.exists():
            text = self._history_path.read_text()
            if text.strip():
                loaded = yaml.safe_load(text)
                if isinstance(loaded, list):
                    history = loaded
        history.append(entry)
        self._history_path.write_text(
            yaml.dump(history, default_flow_style=False, sort_keys=False)
        )

    def remember(
        self,
        text: str,
        category: str = "",
        session_id: str = "",
        supersedes: str | None = None,
    ) -> dict:
        """Store a fact. Optionally supersede an existing entry."""
        entries = self._load()
        now = _now_iso()
        entry_id = _make_id(text, now)

        superseded_id = None
        if supersedes:
            remaining = []
            for e in entries:
                if e.get("id") == supersedes:
                    superseded_id = supersedes
                    e["superseded_by"] = entry_id
                    e["superseded_at"] = now
                    self._append_history(e)
                else:
                    remaining.append(e)
            if superseded_id is None:
                return {"stored": False, "error": f"Entry '{supersedes}' not found"}
            entries = remaining

        entry = {
            "id": entry_id,
            "text": text,
            "category": category,
            "session_id": session_id,
            "timestamp": now,
        }
        entries.append(entry)
        self._save(entries)

        logger.info("Stored explicit memory %s: %s", entry_id, text[:80])
        return {"stored": True, "entry_id": entry_id, "superseded_id": superseded_id}

    def get_all(self) -> list[dict]:
        return self._load()

    def get_by_id(self, entry_id: str) -> dict | None:
        for e in self._load():
            if e.get("id") == entry_id:
                return e
        return None

    def remove(self, entry_id: str) -> dict:
        entries = self._load()
        remaining = []
        removed = None
        for e in entries:
            if e.get("id") == entry_id:
                removed = e
                e["removed_at"] = _now_iso()
                self._append_history(e)
            else:
                remaining.append(e)

        if removed is None:
            return {"removed": False, "error": f"Entry '{entry_id}' not found"}

        self._save(remaining)
        return {"removed": True, "entry_id": entry_id}

    def count(self) -> int:
        return len(self._load())

    def render_context(self) -> str:
        entries = self._load()
        if not entries:
            return ""
        lines = ["# Explicit Memories", ""]
        for e in entries:
            cat = f" [{e['category']}]" if e.get("category") else ""
            lines.append(f"- {e['text']}{cat}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episodic memory: constants
# ---------------------------------------------------------------------------

#: Project-local collection holding the mechanically-indexed session turns.
#: Named ``session_memory`` to match the user-facing terminology in
#: ``dsagt info`` ("session memory").
SESSION_MEMORY_COLLECTION = "session_memory"


# ---------------------------------------------------------------------------
# Trace chunking — split a turn's blocks into embeddable chunks
#
# The trace already carries mechanical boundaries (messages → content blocks),
# so a "chunk" is one block: a user/assistant text block or a tool_result, each
# embedded separately with a ``producer`` (user/llm/tool) label.  tool_use blocks
# aren't embedded — they only supply the ``tool`` name resolved onto the matching
# tool_result.  Over-long blocks split on newlines (cheap, boundary-meaningful).
# ---------------------------------------------------------------------------

#: Char budget above which a block is split on newlines before embedding.
_MAX_CHUNK_CHARS = 1200


def _extract_block_text(block: dict) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _split_oversized(text: str) -> list[str]:
    """Split an over-long block at meaningful boundaries — no tokenizer.

    Paragraph (``\\n\\n``) first, then line (``\\n``), greedily packing pieces up
    to :data:`_MAX_CHUNK_CHARS`; a no-newline blob is hard-sliced as a last
    resort.  Short blocks pass through untouched (stripped).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]
    sep = "\n\n" if "\n\n" in text else ("\n" if "\n" in text else None)
    if sep is None:
        return [
            text[i : i + _MAX_CHUNK_CHARS]
            for i in range(0, len(text), _MAX_CHUNK_CHARS)
        ]
    chunks: list[str] = []
    cur = ""
    for piece in (p.strip() for p in text.split(sep)):
        if not piece:
            continue
        cur = f"{cur}{sep}{piece}" if cur else piece
        while len(cur) > _MAX_CHUNK_CHARS:
            chunks.append(cur[:_MAX_CHUNK_CHARS])
            cur = cur[_MAX_CHUNK_CHARS:].lstrip()
    if cur:
        chunks.append(cur)
    return chunks


def _resolve_tool_names(exchanges: list[dict]) -> dict[str, str]:
    """Map ``tool_use_id`` → tool name across a batch.

    A tool call and its result land in *different* turns (the result is the next
    turn's input), so the lookup is built over the whole delivered batch, not a
    single exchange.
    """
    names: dict[str, str] = {}
    for ex in exchanges:
        blocks = list(ex.get("response", []))
        for msg in ex.get("new_messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                blocks.extend(content)
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                names[b.get("id", "")] = b.get("name", "")
    return names


def _turn_chunks(exchange: dict, tool_names: dict[str, str]) -> list[dict]:
    """One turn's blocks → ``[{text, producer, tool}]`` chunks.

    text blocks carry the message's producer (``user``/``llm``); tool_result
    blocks are ``tool`` with the name resolved via ``tool_use_id``; tool_use
    blocks are skipped (captured in ``tool_names``, not embedded as prose).
    """
    chunks: list[dict] = []

    def emit(text: str, producer: str, tool_name: str) -> None:
        for piece in _split_oversized(text or ""):
            chunks.append({"text": piece, "producer": producer, "tool_name": tool_name})

    def emit_block(block: dict, text_producer: str) -> None:
        bt = block.get("type")
        if bt == "text":
            emit(block.get("text", ""), text_producer, "")
        elif bt == "tool_result":
            emit(
                _extract_block_text(block),
                "tool",
                tool_names.get(block.get("tool_use_id", ""), ""),
            )

    for msg in exchange.get("new_messages", []):
        producer = "user" if msg.get("role", "user") == "user" else "llm"
        content = msg.get("content", "")
        if isinstance(content, str):
            emit(content, producer, "")
        else:
            for block in content:
                emit_block(block, producer)

    for block in exchange.get("response", []):
        emit_block(block, "llm")

    return chunks


# ---------------------------------------------------------------------------
# Episodic memory: the trace-pipeline consumer
# ---------------------------------------------------------------------------


def _epoch_or_now(ts: object) -> float:
    """An exchange timestamp (epoch seconds) → float, else wall-clock now.

    ``Trace.to_exchanges`` carries the span ``start_time`` (epoch
    seconds) when the transcript recorded it, ``None`` otherwise — recency
    weighting needs a number, so fall back to now for unstamped turns.
    """
    return float(ts) if isinstance(ts, (int, float)) else time.time()


def episodic_consumers(config: dict, kb, runtime_dir, session_id) -> list:
    """Build the episodic-memory consumer list from config (empty when off).

    Shared by the live heartbeat (current session) and the startup catch-up
    (previous session) so both feed the same :class:`MemoryExtractor` shape.
    Episodic memory is a compute/storage opt-in (``episodic.enabled``).
    Best-effort: a build failure returns ``[]`` rather than raising.
    """
    epi = config.get("episodic", {}) or {}
    if not epi.get("enabled"):
        return []
    try:
        return [
            MemoryExtractor(
                kb,
                runtime_dir=str(runtime_dir),
                session_id=session_id or "",
            )
        ]
    except Exception as e:  # noqa: BLE001 — memory is best-effort, never fatal
        logger.warning("Could not build episodic-memory consumer: %s", e)
        return []


class MemoryExtractor:
    """Trace-pipeline consumer: ``Trace`` → ``session_memory`` chunks.

    Plugged into :class:`~dsagt.traces.TraceCollector` alongside the MLflow sink;
    each ``write`` receives the (subset of) just-completed turns and indexes them.
    Idempotency is the heartbeat's job — it only delivers turns this consumer
    hasn't acked — so ``write`` just does the work.

    Chunks each turn per-block and embeds it — no LLM, nothing lost, the agent
    is never blocked.
    """

    #: Subscriber name → its own ack file (``.dsagt/trace_acks_memory.json``).
    name = "memory"

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        runtime_dir: str | Path,
        session_id: str = "",
    ):
        self._kb = kb
        self._runtime_dir = Path(runtime_dir)
        self._session_id = session_id

    def write(self, trace: "Trace") -> None:
        exchanges = trace.to_exchanges()
        if not exchanges:
            return
        # Categorization root: this runs on the heartbeat, outside any MCP
        # dispatch, so tag the whole extraction ``dsagt.source=episodic`` — the
        # nested kb.* writes inherit it (otherwise they'd land uncategorized).
        # ``episodic``, NOT ``memory``: this is per-turn internal embedding, and
        # must filter apart from the user-facing memory tools (kb_remember /
        # kb_get_memories) that carry ``dsagt.source=memory``.
        # (This tags the *observability* span, not the memory chunks.)
        from dsagt.observability import open_span

        with open_span("memory.extract", source="episodic") as span:
            n = self._index_turns(exchanges)
            if span is not None:
                # Trace-level Request/Inputs/Outputs are read from this root, so
                # record the turn count in and chunk count out to keep the trace
                # from showing a null request in the MLflow UI.
                span.set_inputs({"n_turns": len(exchanges)})
                span.set_outputs({"chunks_indexed": n})

    def _index_turns(self, exchanges: list[dict]) -> int:
        """Chunk each turn per-block and embed into session_memory."""
        tool_names = _resolve_tool_names(exchanges)
        texts, metas = [], []
        for ex in exchanges:
            turn_id = ex.get("turn_id", "")
            ts_epoch = _epoch_or_now(ex.get("timestamp"))
            for chunk in _turn_chunks(ex, tool_names):
                texts.append(chunk["text"])
                metas.append(
                    {
                        "session_id": self._session_id,
                        "source_type": "turn",
                        "turn_id": turn_id,
                        "producer": chunk["producer"],
                        "tool_name": chunk["tool_name"],
                        "ts_epoch": ts_epoch,
                    }
                )
        if texts:
            self._kb.add_entries(
                texts=texts, collection=SESSION_MEMORY_COLLECTION, metadatas=metas
            )
        return len(texts)
