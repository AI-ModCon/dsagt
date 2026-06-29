"""
Memory management for DSAgt projects.

Two memory types:

**Explicit memory** (ExplicitMemory class):
    User-confirmed facts stored in a YAML file and mirrored into the
    ``explicit_memory`` ChromaDB collection.  Not auto-loaded into the
    agent's context at session start — the agent retrieves entries on
    demand via the ``kb_get_memories`` / ``kb_search`` MCP tools.
    Supports remember, supersede, remove, and retrieval.

**Episodic memory** (MemoryExtractor — Phase 3):
    A trace-pipeline *consumer* (``MemoryExtractor``) that consumes the
    in-process ``Trace`` the heartbeat produces and writes tagged
    facts into the ``session_memory`` collection.  Two tiers:

      * **Tier-0 (mechanical, always):** chunk + mechanical-tag + embed
        each turn — no LLM, the universal fallback for every agent.
      * **Tier-1 (distilled, opt-in):** a small local LLM (``judge.Judge``
        — ``LocalJudge`` by default) tags + condenses each turn into
        ≤1-sentence facts.  On judge failure it degrades to Tier-0 — never
        lose data, never block.

    The per-category-centroid outlier detection (``CategoryCentroids`` /
    ``SuggestionQueue``) gives principled, user-confirmed novelty review on
    top of the distilled facts.  ``extract_session`` (the old end-of-session
    entry point) remains a no-op stub: the heartbeat consumer is the live
    path; ``extract_session`` is retained only for the deferred cross-session
    N+1 catch-up call site.

Files on disk (in project directory):
  explicit_memories.yaml       — active user-confirmed facts
  explicit_memories_history.yaml — superseded/removed entries
  centroids.json               — per-category centroid vectors and counts
  suggestions.json             — queued outlier facts awaiting user review
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import yaml

from dsagt.knowledge import KnowledgeBase

if TYPE_CHECKING:
    from dsagt.judge import Judge
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
        return data if isinstance(data, list) else []

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

#: Project-local collection holding extracted-at-end-of-session facts,
#: insights, and summaries.  Renamed from ``episodic_memory`` to match
#: the user-facing terminology in ``dsagt info`` ("session memory").
SESSION_MEMORY_COLLECTION = "session_memory"

#: Backwards-compat alias.  New code should use ``SESSION_MEMORY_COLLECTION``.
EPISODIC_COLLECTION = SESSION_MEMORY_COLLECTION

# session_memory inherits the kb's default backend / vector_db — it
# used to hardcode ``embedding_backend="api"`` from when embedding was
# assumed to be API-only, but that forced ``kb_remember`` to retry
# against the (possibly invalid) embedding API even when the project
# was otherwise configured for local embeddings, hanging the agent
# for ~60s per call on the retry backoff.

# The stock "AI-data-ready" taxonomy: a small, closed, domain-neutral label set
# — the *closedness* is what makes the Tier-1 LocalJudge viable (small models
# classify into a fixed set reliably; "invent a category" is what they fail at).
# Init solicits per-project domain tags that merge on top (the genomics-specific
# ``assembly`` that used to live here is now exactly such a user tag, not stock).
STOCK_CATEGORIES = {
    "quality_control": "Assessment or filtering of data quality, QC metrics, thresholds, pass/fail rates",
    "data_management": "File organization, data movement, format conversion, naming conventions",
    "transformation": "Data processing steps, parameter choices, pipeline stage configuration",
    "configuration": "Tool settings, environment setup, resource allocation decisions",
    "performance": "Runtime, memory usage, throughput, resource consumption observations",
    "tool_usage": "Tool selection rationale, parameter tuning, tool-specific behaviors or quirks",
    "results": "Output summaries, key findings, deliverables produced",
}

DEFAULT_SENSITIVITY = 0.35


# ---------------------------------------------------------------------------
# Extraction prompt construction
#
# Prompt/parse building blocks for the Phase-3 Trace-based
# extractor (see ``extract_session``).
# ---------------------------------------------------------------------------


def _render_conversation(exchanges: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(exchanges, 1):
        lines.append(f"=== Exchange {i} ({ex.get('timestamp', '')}) ===")

        for msg in ex.get("new_messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(f"[tool_result: {_extract_block_text(block)}]")
                content = "\n".join(parts)
            lines.append(f"[{role}] {content}")

        for block in ex.get("response", []):
            if block.get("type") == "text":
                lines.append(f"[assistant] {block.get('text', '')}")
            elif block.get("type") == "tool_use":
                name = block.get("name", "")
                inp = json.dumps(block.get("input", {}))
                lines.append(f"[assistant → tool_use: {name}({inp})]")

        lines.append("")
    return "\n".join(lines)


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


def _format_categories(categories: dict[str, str]) -> str:
    lines = []
    for name, description in categories.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def build_extraction_prompt(
    exchanges: list[dict],
    categories: dict[str, str] | None = None,
) -> str:
    """Build the extraction prompt from session log exchanges."""
    cats = {**STOCK_CATEGORIES, **(categories or {})}
    conversation = _render_conversation(exchanges)
    category_list = _format_categories(cats)

    return f"""\
You are a memory extraction system for a data pipeline building assistant. \
Review the following session conversation to:

1) Extract relevant facts — short declarative statements about what happened, \
what was decided, what parameters were used, what succeeded or failed, \
and any observations worth remembering for future sessions.

2) Provide a complete summary of the session — the overall flow, what was \
accomplished, key decisions made, and deliverables produced.

3) Reflect across the facts and summary to synthesize insights — patterns, \
lessons learned, generalizable observations that would help in future sessions \
with similar data or tools.

4) Classify each fact and insight into one of these categories:

{category_list}

Output ONLY a JSON object with this structure (no markdown, no preamble):

{{
  "facts": [
    {{"text": "short declarative fact", "category": "category_name"}},
    ...
  ],
  "summary": "paragraph-length session summary",
  "insights": [
    {{"text": "generalizable insight", "category": "category_name"}},
    ...
  ]
}}

Session conversation:

{conversation}"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_extraction_response(response_text: str) -> dict:
    """Parse the LLM's JSON response into facts, summary, and insights."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    parsed = json.loads(text)

    return {
        "facts": parsed.get("facts", []),
        "summary": parsed.get("summary", ""),
        "insights": parsed.get("insights", []),
    }


# ---------------------------------------------------------------------------
# Outlier detection: category centroids
# ---------------------------------------------------------------------------


class CategoryCentroids:
    """Maintains running centroids per category."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def update(self, category: str, embedding: np.ndarray) -> float:
        """Update centroid, return cosine distance before update."""
        vec = embedding.astype(np.float64)

        if category not in self._data:
            self._data[category] = {"centroid": vec.tolist(), "count": 1}
            return 0.0

        entry = self._data[category]
        old_centroid = np.array(entry["centroid"], dtype=np.float64)
        count = entry["count"]

        distance = self._cosine_distance(vec, old_centroid)

        new_centroid = (old_centroid * count + vec) / (count + 1)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid /= norm

        entry["centroid"] = new_centroid.tolist()
        entry["count"] = count + 1

        return float(distance)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return float(1.0 - dot / (norm_a * norm_b))

    @property
    def categories(self) -> list[str]:
        return list(self._data.keys())

    def count(self, category: str) -> int:
        entry = self._data.get(category)
        return entry["count"] if entry else 0


# ---------------------------------------------------------------------------
# Outlier detection: suggestion queue
# ---------------------------------------------------------------------------


class SuggestionQueue:
    """Manages pending outlier suggestions on disk."""

    def __init__(self, path: Path):
        self._path = path
        self._suggestions: list[dict] = []
        if path.exists():
            self._suggestions = json.loads(path.read_text())

    def add(
        self, text: str, category: str, distance: float, session_id: str = ""
    ) -> str:
        suggestion_id = (
            "sug_"
            + hashlib.sha256(f"{text}:{category}:{session_id}".encode()).hexdigest()[:8]
        )

        self._suggestions.append(
            {
                "id": suggestion_id,
                "text": text,
                "category": category,
                "distance": round(distance, 4),
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save()
        return suggestion_id

    def dismiss(self, suggestion_id: str) -> bool:
        before = len(self._suggestions)
        self._suggestions = [s for s in self._suggestions if s["id"] != suggestion_id]
        if len(self._suggestions) < before:
            self._save()
            return True
        return False

    def get_all(self) -> list[dict]:
        return list(self._suggestions)

    def clear(self) -> int:
        count = len(self._suggestions)
        self._suggestions = []
        self._save()
        return count

    @property
    def count(self) -> int:
        return len(self._suggestions)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._suggestions, indent=2))


# ---------------------------------------------------------------------------
# Outlier detection: check and queue
# ---------------------------------------------------------------------------


def check_and_queue_outliers(
    texts: list[str],
    categories: list[str],
    embeddings: np.ndarray,
    centroids: CategoryCentroids,
    queue: SuggestionQueue,
    threshold: float = DEFAULT_SENSITIVITY,
    session_id: str = "",
) -> list[str]:
    """Check facts against centroids, queue outliers. Returns suggestion IDs."""
    suggestion_ids = []

    for text, category, embedding in zip(texts, categories, embeddings):
        if not category:
            continue

        distance = centroids.update(category, embedding)

        if centroids.count(category) <= 1:
            continue

        if distance > threshold:
            sid = queue.add(
                text=text, category=category, distance=distance, session_id=session_id
            )
            suggestion_ids.append(sid)
            logger.info(
                "Outlier flagged (distance=%.3f, threshold=%.3f): %s",
                distance,
                threshold,
                text[:80],
            )

    centroids.save()
    return suggestion_ids


# ---------------------------------------------------------------------------
# Mechanical (Tier-0) tagging
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z][a-z0-9_]{2,}")


def _mechanical_tag(text: str, tags: dict[str, str]) -> str:
    """Pick the best-matching tag by keyword overlap, or ``""`` for none.

    Tier-0's no-LLM classifier: score each tag by how many of its description
    words appear in the turn text, take the best non-zero.  Coarse by design —
    Tier-1's LocalJudge is the accurate path; this only has to be *useful* as
    the universal fallback, and an empty tag (uncategorized) is an honest result
    when nothing matches rather than a forced wrong label.
    """
    words = set(_WORD_RE.findall(text.lower()))
    if not words:
        return ""
    best_tag, best_score = "", 0
    for name, desc in tags.items():
        score = len(words & set(_WORD_RE.findall(desc.lower())))
        if score > best_score:
            best_tag, best_score = name, score
    return best_tag


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


def _batch_epoch(exchanges: list[dict]) -> float:
    """The most-recent exchange timestamp in a turn batch (for distilled facts)."""
    stamps = [
        e["timestamp"]
        for e in exchanges
        if isinstance(e.get("timestamp"), (int, float))
    ]
    return max(stamps) if stamps else time.time()


class MemoryExtractor:
    """Trace-pipeline consumer: ``Trace`` → tagged ``session_memory`` facts.

    Plugged into :class:`~dsagt.traces.TraceCollector` alongside the MLflow sink;
    each ``write`` receives the (subset of) just-completed turns and indexes them.
    Idempotency is the heartbeat's job — it only delivers turns this consumer
    hasn't acked — so ``write`` just does the work.

    Tier selection: with a ``judge`` it runs Tier-1 (distilled facts) and falls
    back to Tier-0 (mechanical chunk) if the judge raises; without one it runs
    Tier-0 directly.  Either way nothing is lost and the agent is never blocked.
    """

    #: Subscriber name → its own ack file (``.dsagt/trace_acks_memory.json``).
    name = "memory"

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        runtime_dir: str | Path,
        session_id: str = "",
        tags: dict[str, str] | None = None,
        judge: "Judge | None" = None,
        outlier_sensitivity: float = DEFAULT_SENSITIVITY,
    ):
        self._kb = kb
        self._runtime_dir = Path(runtime_dir)
        self._session_id = session_id
        self._tags = {**STOCK_CATEGORIES, **(tags or {})}
        self._judge = judge
        self._sensitivity = outlier_sensitivity

    def write(self, trace: "Trace") -> None:
        exchanges = trace.to_exchanges()
        if not exchanges:
            return
        if self._judge is None:
            self._index_tier0(exchanges)
            return
        try:
            facts = self._judge.distill(exchanges, self._tags)
        except Exception as e:  # designed degradation (plan §5), not a swallow
            # *Judge* failure (not a store error) degrades to Tier-0 so the turn
            # is never lost.  A store-layer failure below is left to propagate —
            # TraceCollector's per-consumer isolation stops it blocking the agent,
            # and Tier-0 would hit the same KB anyway.
            logger.warning("Tier-1 judge failed (%s); falling back to Tier-0", e)
            self._index_tier0(exchanges)
            return
        # A judge that legitimately found nothing durable (most turns) stores
        # nothing — that's the empty escape, not a failure, so no Tier-0 redo.
        self._store_facts(facts, _batch_epoch(exchanges))

    def _index_tier0(self, exchanges: list[dict]) -> None:
        """Mechanical: render each turn, mechanically tag it, embed into session_memory."""
        texts, metas = [], []
        for ex in exchanges:
            text = _render_conversation([ex]).strip()
            if not text:
                continue
            texts.append(text)
            metas.append(
                {
                    "session_id": self._session_id,
                    "source_type": "turn",
                    "category": _mechanical_tag(text, self._tags),
                    "tier": "0",
                    "ts_epoch": _epoch_or_now(ex.get("timestamp")),
                }
            )
        if texts:
            self._kb.add_entries(
                texts=texts, collection=SESSION_MEMORY_COLLECTION, metadatas=metas
            )

    def _store_facts(self, facts: list[dict], ts_epoch: float) -> None:
        """Store Tier-1 distilled ``[{text, tag}]`` facts + outlier-detect."""
        if not facts:
            return
        texts = [f["text"] for f in facts]
        cats = [f.get("tag", "") for f in facts]
        metas = [
            {
                "session_id": self._session_id,
                "source_type": "fact",
                "category": c,
                "tier": "1",
                "ts_epoch": ts_epoch,
            }
            for c in cats
        ]
        need_embeddings = self._sensitivity > 0
        result = self._kb.add_entries(
            texts=texts,
            collection=SESSION_MEMORY_COLLECTION,
            metadatas=metas,
            return_embeddings=need_embeddings,
        )
        if need_embeddings:
            check_and_queue_outliers(
                texts=texts,
                categories=cats,
                embeddings=result["embeddings"],
                centroids=CategoryCentroids(self._runtime_dir / "centroids.json"),
                queue=SuggestionQueue(self._runtime_dir / "suggestions.json"),
                threshold=self._sensitivity,
                session_id=self._session_id,
            )


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------


def extract_session(
    project_name: str,
    kb: KnowledgeBase,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    base_url: str | None = None,
    provider: str | None = None,
    categories: dict[str, str] | None = None,
    session_id: str | None = None,
    runtime_dir: Path | None = None,
    outlier_sensitivity: float = 0.0,
    mlflow_uri: str | None = None,
    exchanges: list[dict] | None = None,
) -> dict:
    """Episodic-memory extraction — a no-op stub until Phase 3.

    Phase 3 builds this over the ``Trace`` pipeline (Tier-0
    mechanical chunk/tag/embed by default, opt-in LLM distillation),
    reusing the prompt/parse helpers and outlier detection in this module.

    The full signature is kept so ``session.catch_up_extraction`` and tests
    keep their call site stable when the pipeline lands.  Returns a status
    dict reporting that extraction is unavailable; tool-execution indexing
    (the other half of catch-up work) is unaffected.
    """
    del (
        project_name,
        kb,
        api_key,
        model,
        base_url,
        provider,
        categories,
        runtime_dir,
        outlier_sensitivity,
        mlflow_uri,
        exchanges,
    )
    return {
        "status": "extraction_unavailable",
        "facts": 0,
        "insights": 0,
        "summary": 0,
        "suggestions": 0,
        "total_entries": 0,
        "session_id": session_id or "",
    }
