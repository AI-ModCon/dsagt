"""
End-of-session memory extraction and outlier detection.

Reads the session log (written by the proxy callback), sends the
conversation through a single LLM call to extract facts, a session
summary, and cross-fact insights, then stores the results in the
episodic_memory ChromaDB collection and deletes the session log.

After storage, new facts are checked against per-category embedding
centroids.  Facts whose cosine distance exceeds a threshold are flagged
as suggestions for user review.

The session log is a transient buffer.  Each extraction drains it.
MLflow remains the single source of truth for auditing.

Files on disk (in project runtime directory):
  centroids.json      — per-category centroid vectors and counts
  suggestions.json    — queued outlier facts awaiting user review

Usage::

    from dsagt.memory_extraction import run_extraction

    result = run_extraction(Path("runtime/my-project"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from dsagt.knowledge import CollectionRoute, KnowledgeBase
from dsagt.proxy_callback import SESSION_LOG_FILE

logger = logging.getLogger(__name__)

COLLECTION_NAME = "episodic_memory"

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
    """Read the session log JSONL file.  Returns list of exchange dicts."""
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
    """Atomically read and remove the session log.

    Renames the log file before reading so new exchanges from the proxy
    go to a fresh file.  Safe to call while the proxy is still running.
    """
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
    """Delete the session log.  Returns True if a file was deleted."""
    log_path = trace_dir / SESSION_LOG_FILE
    if log_path.exists():
        log_path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _render_conversation(exchanges: list[dict]) -> str:
    """Render session log exchanges as readable conversation text."""
    lines = []
    for i, ex in enumerate(exchanges, 1):
        lines.append(f"=== Exchange {i} ({ex.get('timestamp', '')}) ===")

        for msg in ex.get("new_messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Anthropic format: list of content blocks
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
    """Extract text from a content block (string or list of text blocks)."""
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
    """Format categories as a numbered list for the prompt."""
    lines = []
    for name, description in categories.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def build_extraction_prompt(
    exchanges: list[dict],
    categories: dict[str, str] | None = None,
) -> str:
    """Build the extraction prompt from session log exchanges.

    This is a pure function — no I/O, no LLM calls.  Testable independently.
    """
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
    """Parse the LLM's JSON response into facts, summary, and insights.

    Returns a dict with keys: facts (list), summary (str), insights (list).
    Each fact/insight has 'text' and 'category' fields.
    """
    # Strip markdown fences if present
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
    """Call the LLM with the extraction prompt.  Returns raw response text.

    Goes directly to the API, not through the proxy (avoids logging
    the extraction call itself in the session log).
    """
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

    # Anthropic response: content[0].text
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]

    return ""


# ---------------------------------------------------------------------------
# Outlier detection: category centroids
# ---------------------------------------------------------------------------

class CategoryCentroids:
    """Maintains running centroids per category.

    Centroids are incrementally updated: each new vector shifts the
    centroid proportionally.  No batch recomputation needed.
    """

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self._data = raw

    def update(self, category: str, embedding: np.ndarray) -> float:
        """Update the centroid for a category with a new embedding.

        Returns the cosine distance between the new embedding and the
        centroid BEFORE the update (used for outlier detection).
        """
        vec = embedding.astype(np.float64)

        if category not in self._data:
            self._data[category] = {
                "centroid": vec.tolist(),
                "count": 1,
            }
            return 0.0

        entry = self._data[category]
        old_centroid = np.array(entry["centroid"], dtype=np.float64)
        count = entry["count"]

        # Cosine distance before update
        distance = self._cosine_distance(vec, old_centroid)

        # Incremental centroid update
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
        """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
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
        self,
        text: str,
        category: str,
        distance: float,
        session_id: str = "",
    ) -> str:
        """Add a suggestion.  Returns the suggestion ID."""
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
        """Remove a suggestion by ID.  Returns True if found."""
        before = len(self._suggestions)
        self._suggestions = [s for s in self._suggestions if s["id"] != suggestion_id]
        if len(self._suggestions) < before:
            self._save()
            return True
        return False

    def get_all(self) -> list[dict]:
        return list(self._suggestions)

    def clear(self) -> int:
        """Remove all suggestions.  Returns count removed."""
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
    """Check facts against category centroids, queue outliers.

    Updates centroids incrementally for every fact.  Facts whose distance
    from their category centroid exceeds the threshold are added to the
    suggestion queue.

    Returns list of suggestion IDs for newly queued outliers.
    """
    suggestion_ids = []

    for i, (text, category, embedding) in enumerate(zip(texts, categories, embeddings)):
        if not category:
            continue

        distance = centroids.update(category, embedding)

        # Skip the first fact in a category (distance is 0, no centroid yet)
        if centroids.count(category) <= 1:
            continue

        if distance > threshold:
            sid = queue.add(
                text=text,
                category=category,
                distance=distance,
                session_id=session_id,
            )
            suggestion_ids.append(sid)
            logger.info(
                "Outlier flagged (distance=%.3f, threshold=%.3f): %s",
                distance, threshold, text[:80],
            )

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
    """Extract memories from the session log and store in episodic_memory.

    Reads the session log, sends one LLM call, stores results in
    ChromaDB, runs outlier detection, then deletes the session log.

    Parameters
    ----------
    outlier_sensitivity : float
        Cosine distance threshold for flagging outliers.  0 disables.

    Returns dict with counts of what was stored.
    """
    exchanges = drain_session_log(trace_dir)
    if not exchanges:
        return {"status": "empty", "facts": 0, "insights": 0}

    prompt = build_extraction_prompt(exchanges, categories)
    response_text = call_extraction_llm(prompt, api_key, model, base_url)
    extracted = parse_extraction_response(response_text)

    # Determine session_id from first exchange timestamp if not provided
    if not session_id:
        first_ts = exchanges[0].get("timestamp", "unknown")
        session_id = f"session_{first_ts[:10]}"

    # Collect provenance metadata from the consumed exchanges
    timestamps = [ex.get("timestamp", "") for ex in exchanges if ex.get("timestamp")]
    timestamp_start = min(timestamps) if timestamps else ""
    timestamp_end = max(timestamps) if timestamps else ""
    call_ids = [ex.get("call_id", "") for ex in exchanges if ex.get("call_id")]
    trace_refs = ",".join(call_ids) if call_ids else ""

    # Common metadata for all entries in this extraction batch
    batch_meta = {
        "session_id": session_id,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }
    if trace_refs:
        batch_meta["trace_refs"] = trace_refs

    # Store facts
    fact_texts = []
    fact_metas = []
    fact_categories = []
    for fact in extracted["facts"]:
        fact_texts.append(fact["text"])
        fact_categories.append(fact.get("category", ""))
        fact_metas.append({
            **batch_meta,
            "source_type": "extraction",
            "category": fact.get("category", ""),
        })

    # Store summary as a single entry
    if extracted["summary"]:
        fact_texts.append(extracted["summary"])
        fact_categories.append("results")
        fact_metas.append({
            **batch_meta,
            "source_type": "summary",
            "category": "results",
        })

    # Store insights
    for insight in extracted["insights"]:
        fact_texts.append(insight["text"])
        fact_categories.append(insight.get("category", ""))
        fact_metas.append({
            **batch_meta,
            "source_type": "insight",
            "category": insight.get("category", ""),
        })

    stored = {"facts": 0, "insights": 0, "summary": 0, "suggestions": 0}
    if fact_texts:
        kb.add_entries(
            texts=fact_texts,
            collection=COLLECTION_NAME,
            metadatas=fact_metas,
            route=EPISODIC_MEMORY_ROUTE,
        )
        stored["facts"] = len(extracted["facts"])
        stored["insights"] = len(extracted["insights"])
        stored["summary"] = 1 if extracted["summary"] else 0

        # Outlier detection
        if outlier_sensitivity > 0:
            project_dir = runtime_dir or trace_dir.parent
            centroids = CategoryCentroids(project_dir / "centroids.json")
            queue = SuggestionQueue(project_dir / "suggestions.json")
            embeddings = kb.embed_texts(fact_texts, COLLECTION_NAME)

            suggestion_ids = check_and_queue_outliers(
                texts=fact_texts,
                categories=fact_categories,
                embeddings=embeddings,
                centroids=centroids,
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


# ---------------------------------------------------------------------------
# Orchestration: load config, run extraction, clean up
# ---------------------------------------------------------------------------

def run_extraction(project_dir: str | Path) -> dict:
    """Run memory extraction for a project and clean up the session log.

    Loads the session log, sends one LLM call to extract facts/summary/insights,
    stores results in episodic_memory, then deletes the session log.  If extraction
    fails, the session log is still deleted (transient buffer, MLflow has the truth).

    Returns the extraction result dict, or a status dict on error/empty.
    """
    from dsagt.config import load_config

    project_dir = Path(project_dir)
    trace_dir = project_dir / "trace_archive"

    config = load_config(project_dir.name)
    api_key = config.get("llm", {}).get("api_key", "") or os.environ.get("LLM_API_KEY", "")
    model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    session_id = config.get("project", "")
    categories = config.get("categories", {})

    if not api_key or api_key.startswith("${"):
        logger.warning("No API key available for extraction, skipping")
        delete_session_log(trace_dir)
        return {"status": "skipped", "reason": "no_api_key"}

    kb = KnowledgeBase(index_dir=project_dir / "kb_index")
    try:
        return extract_session(
            trace_dir=trace_dir,
            kb=kb,
            api_key=api_key,
            model=model,
            session_id=session_id,
            categories=categories if categories else None,
            runtime_dir=project_dir,
            outlier_sensitivity=float(
                config.get("extraction", {}).get("outlier_sensitivity", 0)
            ),
        )
    finally:
        kb.close()
        # Safety net: ensure session log files are gone even if extraction raised.
        for suffix in (".jsonl", ".consumed"):
            leftover = trace_dir / f"session_log{suffix}"
            if leftover.exists():
                leftover.unlink()
