"""
Memory management for DSAgt projects.

Two memory types:

**Explicit memory** (ExplicitMemory class):
    User-confirmed facts stored in a YAML file. Loaded into agent context
    at session start. Supports remember, supersede, remove, and retrieval.

**Episodic memory** (extract_session and friends):
    End-of-session LLM extraction of facts, summaries, and insights from
    the session log. Stored in the ``episodic_memory`` ChromaDB collection.
    Includes outlier detection via per-category embedding centroids.

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
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import yaml

from dsagt.knowledge import CollectionRoute, KnowledgeBase
from dsagt.provenance import SESSION_LOG_FILE

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
            if entries else ""
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

EPISODIC_COLLECTION = "episodic_memory"

EPISODIC_MEMORY_ROUTE = CollectionRoute(
    embedding_backend="api",
    vector_db="chroma",
    description="Episodic memory: extracted facts, summaries, and insights from sessions.",
)

STOCK_CATEGORIES = {
    "quality_control": "Assessment or filtering of data quality, QC metrics, thresholds, pass/fail rates",
    "data_management": "File organization, data movement, format conversion, naming conventions",
    "transformation": "Data processing steps, parameter choices, pipeline stage configuration",
    "assembly": "Genome assembly, contig generation, scaffolding, assembly QC metrics",
    "configuration": "Tool settings, environment setup, resource allocation decisions",
    "performance": "Runtime, memory usage, throughput, resource consumption observations",
    "tool_usage": "Tool selection rationale, parameter tuning, tool-specific behaviors or quirks",
    "results": "Output summaries, key findings, deliverables produced",
}

DEFAULT_SENSITIVITY = 0.35


# ---------------------------------------------------------------------------
# Session log reading
# ---------------------------------------------------------------------------

def load_session_log(trace_dir: Path) -> list[dict]:
    """Read the session log JSONL file. Returns list of exchange dicts."""
    log_path = trace_dir / SESSION_LOG_FILE
    if not log_path.exists():
        return []

    entries = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def drain_session_log(trace_dir: Path) -> list[dict]:
    """Atomically read and remove the session log."""
    log_path = trace_dir / SESSION_LOG_FILE
    if not log_path.exists():
        return []

    consumed_path = log_path.with_suffix(".consumed")
    log_path.rename(consumed_path)

    entries = []
    for line in consumed_path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))

    consumed_path.unlink()
    return entries


def delete_session_log(trace_dir: Path) -> bool:
    """Delete the session log. Returns True if a file was deleted."""
    log_path = trace_dir / SESSION_LOG_FILE
    if log_path.exists():
        log_path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Extraction prompt construction
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
            b.get("text", "") for b in content
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
# LLM call
# ---------------------------------------------------------------------------

def call_extraction_llm(
    prompt: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    base_url: str | None = None,
) -> str:
    """Call the LLM with the extraction prompt. Returns raw response text."""
    base_url = base_url or "https://api.anthropic.com/v1"

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]

    return ""


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

    def add(self, text: str, category: str, distance: float, session_id: str = "") -> str:
        suggestion_id = "sug_" + hashlib.sha256(
            f"{text}:{category}:{session_id}".encode()
        ).hexdigest()[:8]

        self._suggestions.append({
            "id": suggestion_id,
            "text": text,
            "category": category,
            "distance": round(distance, 4),
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
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
            sid = queue.add(text=text, category=category, distance=distance, session_id=session_id)
            suggestion_ids.append(sid)
            logger.info("Outlier flagged (distance=%.3f, threshold=%.3f): %s", distance, threshold, text[:80])

    centroids.save()
    return suggestion_ids


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------

def extract_session(
    trace_dir: Path,
    kb: KnowledgeBase,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    base_url: str | None = None,
    categories: dict[str, str] | None = None,
    session_id: str | None = None,
    runtime_dir: Path | None = None,
    outlier_sensitivity: float = 0.0,
) -> dict:
    """Extract memories from the session log and store in episodic_memory."""
    exchanges = drain_session_log(trace_dir)
    if not exchanges:
        return {"status": "empty", "facts": 0, "insights": 0}

    prompt = build_extraction_prompt(exchanges, categories)
    response_text = call_extraction_llm(prompt, api_key, model, base_url)
    extracted = parse_extraction_response(response_text)

    if not session_id:
        first_ts = exchanges[0].get("timestamp", "unknown")
        session_id = f"session_{first_ts[:10]}"

    timestamps = [ex.get("timestamp", "") for ex in exchanges if ex.get("timestamp")]
    timestamp_start = min(timestamps) if timestamps else ""
    timestamp_end = max(timestamps) if timestamps else ""
    call_ids = [ex.get("call_id", "") for ex in exchanges if ex.get("call_id")]
    trace_refs = ",".join(call_ids) if call_ids else ""

    batch_meta = {
        "session_id": session_id,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }
    if trace_refs:
        batch_meta["trace_refs"] = trace_refs

    fact_texts = []
    fact_metas = []
    fact_categories = []
    for fact in extracted["facts"]:
        fact_texts.append(fact["text"])
        fact_categories.append(fact.get("category", ""))
        fact_metas.append({**batch_meta, "source_type": "extraction", "category": fact.get("category", "")})

    if extracted["summary"]:
        fact_texts.append(extracted["summary"])
        fact_categories.append("results")
        fact_metas.append({**batch_meta, "source_type": "summary", "category": "results"})

    for insight in extracted["insights"]:
        fact_texts.append(insight["text"])
        fact_categories.append(insight.get("category", ""))
        fact_metas.append({**batch_meta, "source_type": "insight", "category": insight.get("category", "")})

    stored = {"facts": 0, "insights": 0, "summary": 0, "suggestions": 0}
    if fact_texts:
        kb.add_entries(
            texts=fact_texts,
            collection=EPISODIC_COLLECTION,
            metadatas=fact_metas,
            route=EPISODIC_MEMORY_ROUTE,
        )
        stored["facts"] = len(extracted["facts"])
        stored["insights"] = len(extracted["insights"])
        stored["summary"] = 1 if extracted["summary"] else 0

        if outlier_sensitivity > 0:
            project_dir = runtime_dir or trace_dir.parent
            centroids_obj = CategoryCentroids(project_dir / "centroids.json")
            queue = SuggestionQueue(project_dir / "suggestions.json")
            embeddings = kb.embed_texts(fact_texts, EPISODIC_COLLECTION)

            suggestion_ids = check_and_queue_outliers(
                texts=fact_texts,
                categories=fact_categories,
                embeddings=embeddings,
                centroids=centroids_obj,
                queue=queue,
                threshold=outlier_sensitivity,
                session_id=session_id,
            )
            stored["suggestions"] = len(suggestion_ids)

    return {
        "status": "ok",
        **stored,
        "total_entries": len(fact_texts),
        "session_id": session_id,
    }
