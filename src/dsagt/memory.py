"""
Memory management for DSAgt projects.

Two memory types:

**Explicit memory** (ExplicitMemory class):
    User-confirmed facts stored in a YAML file. Loaded into agent context
    at session start. Supports remember, supersede, remove, and retrieval.

**Episodic memory** (extract_session and friends):
    End-of-session LLM extraction of facts, summaries, and insights.
    Conversation history comes from MLflow traces with
    ``service.name = "dsagt-proxy"`` — i.e., LLM calls forwarded
    through ``dsagt-proxy`` and autologged via
    ``mlflow.litellm.autolog()`` into a uniform shape
    (``mlflow.spanInputs`` = request kwargs with ``messages``,
    ``mlflow.spanOutputs`` = provider-specific response).

    Native agent OTel emission (Claude Code, Goose) does NOT feed
    extraction, even though those traces are visible in the MLflow UI.
    The reason is shape divergence: Claude Code puts conversation in
    span events (``api_response_body``), Goose puts tool calls in
    ``dispatch_tool_call`` spans with a domain-specific schema, and
    LiteLLM autolog uses ``mlflow.spanInputs`` / ``mlflow.spanOutputs``.
    Parsing all three would mean three parallel parsers in this module
    and per-agent maintenance forever.  Instead, extraction reads one
    canonical shape (the one ``dsagt-proxy`` emits via autolog) and
    users who want extraction run ``dsagt start --enable-proxy``.

    ``drain_session_traces`` queries proxy-shape traces in the session,
    skips ones already tagged with ``dsagt.memory.extracted = "true"``,
    formats each into the exchange shape the prompt expects, and tags
    consumed traces so re-runs are idempotent.  Stored in the
    ``episodic_memory`` ChromaDB collection.  Includes outlier
    detection via per-category embedding centroids.

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

import numpy as np

import yaml

from dsagt.knowledge import CollectionRoute, KnowledgeBase

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
# Session-trace reading (MLflow)
# ---------------------------------------------------------------------------

#: Trace tag we set after consuming a trace into an extraction run, so
#: re-runs of ``extract_session`` for the same session don't double-feed
#: the same exchange into the LLM.  ``MlflowClient.set_trace_tag`` is the
#: idempotent equivalent of the old ``drain → unlink`` pattern.
DSAGT_MEMORY_PROCESSED_TAG = "dsagt.memory.extracted"


def _safe_parse_json(value):
    """Parse value as JSON; return as-is if already a dict/list."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _trace_to_exchange(row) -> dict | None:
    """Format one MLflow ``search_traces`` row as an extraction-prompt exchange.

    The exchange shape mirrors what the proxy used to write to
    ``session_log.jsonl`` so ``_render_conversation`` doesn't need to
    change:

        {"timestamp": ..., "trace_id": ..., "model": ...,
         "new_messages": [...], "response": [...content blocks...]}

    Returns None when the trace doesn't carry recognisable LLM-call
    request/response shape (e.g. tool-execute spans, kb.* spans) so
    callers can skip silently.
    """
    request = _safe_parse_json(row.get("request"))
    response = _safe_parse_json(row.get("response"))

    messages = request.get("messages") if isinstance(request, dict) else None
    if not messages:
        return None

    response_blocks: list[dict] = []
    if isinstance(response, dict):
        # Anthropic shape: top-level ``content`` already a list of blocks.
        if isinstance(response.get("content"), list):
            response_blocks.extend(response["content"])
        # OpenAI shape: choices[].message.content (str or block list) plus
        # tool_calls (translated to tool_use blocks for prompt consistency).
        for choice in response.get("choices") or []:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content:
                response_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                response_blocks.extend(content)
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function") or {}
                response_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "input": _safe_parse_json(func.get("arguments")) or {},
                })

    return {
        "timestamp": str(row.get("request_time") or ""),
        "trace_id": row.get("trace_id") or "",
        "model": (request.get("model") if isinstance(request, dict) else "") or "",
        "new_messages": messages,
        "response": response_blocks,
    }


#: Service name extraction reads from.  Restricting to ``dsagt-proxy``
#: keeps the parser simple — every trace emitted by the proxy is
#: guaranteed LiteLLM-autolog shape (``mlflow.spanInputs`` /
#: ``mlflow.spanOutputs``), the same shape ``_trace_to_exchange`` parses.
#: Native agent traces (claude-code, goose) skip extraction by design;
#: see module docstring for why.
DSAGT_EXTRACTION_SOURCE_SERVICE_NAME = "dsagt-proxy"


def drain_session_traces(
    project_name: str, session_id: str, mlflow_uri: str | None = None,
) -> list[dict]:
    """Pull untagged proxy-shape session traces, format as exchanges, tag.

    Uses ``mlflow.search_traces`` to find traces in the project's
    experiment whose ``mlflow.trace.session`` metadata matches
    *session_id*, then filters to those emitted by ``dsagt-proxy``
    (the only canonical-shape source — see module docstring).  Skips
    ones already tagged with ``DSAGT_MEMORY_PROCESSED_TAG``; tags each
    consumed trace so a subsequent extraction run on the same session
    is a no-op.

    Returns a list of exchange dicts in chronological order.  Empty
    list when no experiment, no proxy-shape traces, or every trace was
    already processed.  Importantly, an empty result when the user ran
    without ``--enable-proxy`` is *expected*: native-OTel agents
    (Claude Code, Goose) emit traces in shapes this parser doesn't
    handle, and the design choice is to require the proxy for
    extraction rather than maintain N per-agent parsers.
    """
    import os
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        logger.warning("MLFLOW_TRACKING_URI not set; cannot drain session traces")
        return []

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    exp = client.get_experiment_by_name(project_name)
    if exp is None:
        return []

    df = mlflow.search_traces(
        locations=[exp.experiment_id],
        filter_string=f"metadata.`mlflow.trace.session` = '{session_id}'",
        max_results=10000,
        order_by=["timestamp_ms ASC"],
    )
    if df is None or df.empty:
        return []

    exchanges: list[dict] = []
    consumed_ids: list[str] = []
    for _, row in df.iterrows():
        tags = row.get("tags") or {}
        if isinstance(tags, dict) and tags.get(DSAGT_MEMORY_PROCESSED_TAG) == "true":
            continue
        if not _trace_emitted_by(row, DSAGT_EXTRACTION_SOURCE_SERVICE_NAME):
            continue
        ex = _trace_to_exchange(row)
        if ex:
            exchanges.append(ex)
        # Tag every proxy-shape trace we considered (parsed or not), so
        # the next run doesn't re-inspect them.  Non-proxy traces are
        # left untagged on purpose — a future extraction run that
        # broadens the source filter will pick them up.
        trace_id = row.get("trace_id")
        if trace_id:
            consumed_ids.append(trace_id)

    for trace_id in consumed_ids:
        try:
            client.set_trace_tag(trace_id, DSAGT_MEMORY_PROCESSED_TAG, "true")
        except Exception as e:
            # Tag failure is non-fatal — worst case we re-extract that trace
            # next session, which is annoying but not data-destroying.
            logger.debug("set_trace_tag(%s) failed: %s", trace_id, e)

    return exchanges


def _trace_emitted_by(row, service_name: str) -> bool:
    """Return True when *row*'s root span carries ``service.name == name``.

    MLflow's OTLP receiver flows the OTel resource attribute through to
    each span's attributes, so every span in a trace from a given
    process carries the same ``service.name``.  We check the first span
    we find — the shape varies (Span object / dict / pandas Series)
    across MLflow versions, mirroring info.py:_source_from_spans.
    """
    spans = row.get("spans") or []
    try:
        for span in spans:
            attrs = getattr(span, "attributes", None)
            if attrs is None and isinstance(span, dict):
                attrs = span.get("attributes")
            if attrs and attrs.get("service.name") == service_name:
                return True
    except (TypeError, AttributeError):
        pass
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
    provider: str | None = None,
) -> str:
    """Call the LLM with the extraction prompt. Returns raw response text.

    Uses LiteLLM so the call works against any OpenAI- or Anthropic-format
    upstream the user's ``llm.base_url`` points at.  Hand-rolling an
    Anthropic-format request would 404 against an OpenAI-compatible gateway
    (e.g. PNNL's ai-incubator-api), and vice versa.
    """
    import litellm

    # Mirror what the proxy does (commands/proxy_server.py): prefix the model
    # with the configured provider so LiteLLM picks the right request format.
    # Falls back to ``openai`` when provider is unset, preserving the historic
    # behavior for callers that don't yet thread ``llm.provider`` through.
    if base_url:
        completion_model = f"{provider or 'openai'}/{model}"
    else:
        completion_model = model

    response = litellm.completion(
        model=completion_model,
        api_base=base_url,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        timeout=120.0,
    )
    return response.choices[0].message.content or ""


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
    """Extract memories from a session's MLflow traces; store in episodic_memory.

    *project_name* is the MLflow experiment name (matches ``DSAGT_PROJECT``).
    *session_id* selects which traces to consume; required when *exchanges*
    is not supplied.  Tests pass *exchanges* directly to bypass MLflow.
    """
    if not session_id:
        return {"status": "empty", "facts": 0, "insights": 0,
                "reason": "no_session_id"}

    if exchanges is None:
        exchanges = drain_session_traces(project_name, session_id, mlflow_uri)
    if not exchanges:
        return {"status": "empty", "facts": 0, "insights": 0}

    prompt = build_extraction_prompt(exchanges, categories)
    response_text = call_extraction_llm(prompt, api_key, model, base_url, provider)
    extracted = parse_extraction_response(response_text)

    timestamps = [ex.get("timestamp", "") for ex in exchanges if ex.get("timestamp")]
    timestamp_start = min(timestamps) if timestamps else ""
    timestamp_end = max(timestamps) if timestamps else ""
    trace_ids = [ex.get("trace_id", "") for ex in exchanges if ex.get("trace_id")]
    trace_refs = ",".join(trace_ids) if trace_ids else ""

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
        # Ask add_entries to hand back the freshly-computed embeddings so we
        # don't pay a second embedding round-trip for outlier detection below.
        # On API embedders this halves the wall time of memory extraction.
        need_embeddings = outlier_sensitivity > 0
        add_result = kb.add_entries(
            texts=fact_texts,
            collection=EPISODIC_COLLECTION,
            metadatas=fact_metas,
            return_embeddings=need_embeddings,
        )
        stored["facts"] = len(extracted["facts"])
        stored["insights"] = len(extracted["insights"])
        stored["summary"] = 1 if extracted["summary"] else 0

        if need_embeddings:
            if runtime_dir is None:
                raise ValueError(
                    "extract_session: runtime_dir is required when "
                    "outlier_sensitivity > 0 (centroids/suggestions live there)"
                )
            project_dir = runtime_dir
            centroids_obj = CategoryCentroids(project_dir / "centroids.json")
            queue = SuggestionQueue(project_dir / "suggestions.json")

            suggestion_ids = check_and_queue_outliers(
                texts=fact_texts,
                categories=fact_categories,
                embeddings=add_result["embeddings"],
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
