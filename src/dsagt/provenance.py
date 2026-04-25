"""
Provenance and session tracking.

**Execution capture** (dsagt-run wrapper):
    Wraps a shell command, captures exact execution data, writes a JSON record.

**LLM conversation tracking** (LiteLLM callback):
    Tracks tool_use/tool_result blocks from LLM conversations, writes intent +
    report records, maintains the session log for memory extraction.

**Record indexing** (ChromaDB):
    Indexes execution records into a ``tool_executions`` collection for
    semantic search and metadata filtering.

**Pipeline reconstruction**:
    Reads execution records, builds a dependency graph from input/output
    file overlap, and renders as a bash script or Snakemake workflow.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dsagt.knowledge import CollectionRoute, KnowledgeBase

logger = logging.getLogger(__name__)

TOOL_EXECUTIONS_COLLECTION = "tool_executions"
TOOL_EXECUTIONS_ROUTE = CollectionRoute(
    embedding_backend="api",
    vector_db="chroma",
    description="Indexed tool execution records from trace_archive.",
)


# ---------------------------------------------------------------------------
# Execution capture (dsagt-run)
# ---------------------------------------------------------------------------

def _resolve_records_dir(explicit: str | None) -> Path:
    """Determine the records directory from arg, env var, or project dir.

    Priority: explicit flag → $DSAGT_RECORDS_DIR → $DSAGT_PROJECT_DIR/trace_archive.
    Raises ValueError if none are available.
    """
    if explicit:
        return Path(explicit)
    from_env = os.environ.get("DSAGT_RECORDS_DIR")
    if from_env:
        return Path(from_env)
    project_dir = os.environ.get("DSAGT_PROJECT_DIR")
    if project_dir:
        return Path(project_dir) / "trace_archive"
    raise ValueError(
        "No records directory: set --records-dir, $DSAGT_RECORDS_DIR, or $DSAGT_PROJECT_DIR"
    )


def _parse_file_list(raw: str | None) -> list[str]:
    """Split a comma-separated file list, stripping whitespace."""
    if not raw:
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def run_and_record(
    tool_name: str,
    command: list[str],
    records_dir: Path,
    session_id: str | None = None,
    record_id: str | None = None,
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
) -> int:
    """Execute a command, write an execution record, return the exit code."""
    from dsagt.observability import obs, tool_execute_span, truncate

    record_id = record_id or uuid.uuid4().hex[:12]
    session_id = session_id or os.environ.get("DSAGT_SESSION_ID")

    with tool_execute_span(record_id, tool_name):
        timestamp_start = datetime.now(timezone.utc).isoformat()
        start_perf = time.perf_counter()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            return_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except FileNotFoundError:
            return_code = 127
            stdout = ""
            stderr = f"dsagt-run: command not found: {command[0]}"
        except (PermissionError, OSError) as e:
            return_code = 1
            stdout = ""
            stderr = f"dsagt-run: execution error: {e}"

        duration_ms = round((time.perf_counter() - start_perf) * 1000, 3)
        timestamp_end = datetime.now(timezone.utc).isoformat()

        # Attach execution summary to the span. Full payload still goes to
        # trace_archive/<record_id>.json; the span only carries truncated
        # summaries that render usefully in the MLflow UI.
        obs.set_many({
            "exit_code": return_code,
            "duration_ms": duration_ms,
            "n_input_files": len(input_files or []),
            "n_output_files": len(output_files or []),
            "command": truncate(" ".join(command), 256),
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
        })
        if stderr.strip():
            obs.set("stderr_truncated", truncate(stderr, 256))
        if return_code != 0:
            obs.event("tool_failed", exit_code=return_code)

        # Populate the MLflow trace UI's Input/Output tabs.  Truncate to
        # ~4KB per side so big tool results don't bloat the trace store
        # (the full payload is on disk in trace_archive/<record_id>.json).
        obs.set_inputs({
            "tool": tool_name,
            "command": list(command),
            "input_files": input_files or [],
        })
        obs.set_outputs({
            "exit_code": return_code,
            "duration_ms": duration_ms,
            "stdout": truncate(stdout, 4096),
            "stderr": truncate(stderr, 4096) if stderr else "",
            "output_files": output_files or [],
        })

    record = {
        "record_id": record_id,
        "tool_name": tool_name,
        "session_id": session_id,
        "execution": {
            "exact_command": command,
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "input_files": input_files or [],
            "output_files": output_files or [],
        },
    }

    _write_record(record, records_dir)

    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)

    return return_code


def _write_record(record: dict, records_dir: Path) -> Path:
    """Write a JSON execution record. Returns the file path."""
    records_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{record['tool_name']}_{ts}_{record['record_id']}.json"
    path = records_dir / filename

    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Record indexing (ChromaDB)
# ---------------------------------------------------------------------------

def render_execution_text(record: dict) -> str:
    """Convert a tool execution record into embeddable natural-language text."""
    tool_name = record.get("tool_name", "unknown")
    intent = record.get("intent") or {}
    execution = record.get("execution")
    report = record.get("report") or {}

    parts = [f"Tool: {tool_name}"]

    if execution and execution.get("exact_command"):
        cmd = execution["exact_command"]
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        parts.append(f"Command: {cmd}")
    elif intent.get("parameters"):
        parts.append(f"Parameters: {json.dumps(intent['parameters'])}")

    if execution and execution.get("return_code") is not None:
        rc = execution["return_code"]
        status = "succeeded" if rc == 0 else f"failed (exit code {rc})"
        parts.append(f"Outcome: {status}")
    elif report.get("agent_output"):
        output = report["agent_output"]
        if len(output) > 500:
            output = output[:500] + "..."
        parts.append(f"Agent report: {output}")

    if execution:
        start = execution.get("timestamp_start", "")
        end = execution.get("timestamp_end", "")
        if start and end:
            parts.append(f"Duration: {start} to {end}")

    if execution and execution.get("input_files"):
        parts.append(f"Input files: {', '.join(execution['input_files'])}")
    if execution and execution.get("output_files"):
        parts.append(f"Output files: {', '.join(execution['output_files'])}")

    if execution and execution.get("stderr"):
        stderr = execution["stderr"].strip()
        if stderr:
            if len(stderr) > 300:
                stderr = stderr[:300] + "..."
            parts.append(f"Stderr: {stderr}")

    return "\n".join(parts)


def execution_metadata(record: dict) -> dict:
    """Extract ChromaDB-filterable metadata from a tool execution record."""
    execution = record.get("execution")
    intent = record.get("intent") or {}

    meta: dict = {}
    meta["tool_name"] = record.get("tool_name", intent.get("command", "unknown"))
    meta["session_id"] = record.get("session_id", intent.get("session_id", "unknown"))

    if execution and execution.get("return_code") is not None:
        meta["return_code"] = execution["return_code"]

    meta["wrapper_used"] = 1 if execution is not None else 0

    timestamp = None
    if execution and execution.get("timestamp_start"):
        timestamp = execution["timestamp_start"]
    elif intent.get("timestamp_requested"):
        timestamp = intent["timestamp_requested"]
    if timestamp:
        meta["timestamp"] = timestamp

    record_id = record.get("record_id", "")
    if record_id:
        meta["record_id"] = record_id

    return meta


def index_execution_record(record: dict, kb: KnowledgeBase) -> dict:
    """Render and store a single tool execution record in the knowledge base."""
    text = render_execution_text(record)
    metadata = execution_metadata(record)

    return kb.add_entries(
        texts=[text],
        collection=TOOL_EXECUTIONS_COLLECTION,
        metadatas=[metadata],
        route=TOOL_EXECUTIONS_ROUTE,
    )


def index_trace_archive(
    trace_dir: Path,
    kb: KnowledgeBase,
    indexed_ids: set[str] | None = None,
) -> dict:
    """Batch-index all tool execution records in a trace archive directory."""
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
            collection=TOOL_EXECUTIONS_COLLECTION,
            metadatas=metadatas,
            route=TOOL_EXECUTIONS_ROUTE,
        )

    return {
        "indexed": len(texts),
        "skipped": skipped,
        "errors": errors,
        "total_files": len(json_files),
    }


# ---------------------------------------------------------------------------
# Pipeline reconstruction
# ---------------------------------------------------------------------------

def load_pipeline_records(trace_dir: Path, session_id: str | None = None) -> list[dict]:
    """Load execution records that have the wrapper execution layer.

    Returns records sorted by execution start time.
    """
    if not trace_dir.is_dir():
        return []

    records = []
    for path in trace_dir.glob("*.json"):
        raw = json.loads(path.read_text())
        execution = raw.get("execution")
        if not execution:
            continue
        if session_id and raw.get("session_id") != session_id:
            continue
        records.append(raw)

    records.sort(key=lambda r: r["execution"].get("timestamp_start", ""))
    return records


def build_dependency_graph(records: list[dict]) -> dict[int, list[int]]:
    """Build a dependency graph from input/output file overlap."""
    output_to_step: dict[str, int] = {}
    for i, record in enumerate(records):
        for f in record["execution"].get("output_files", []):
            output_to_step[f] = i

    deps: dict[int, list[int]] = {i: [] for i in range(len(records))}
    for i, record in enumerate(records):
        for f in record["execution"].get("input_files", []):
            producer = output_to_step.get(f)
            if producer is not None and producer != i:
                if producer not in deps[i]:
                    deps[i].append(producer)

    return deps


def render_bash(records: list[dict], deps: dict[int, list[int]]) -> str:
    """Render the pipeline as a bash script."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Pipeline reconstructed from DSAgt execution records",
        "",
    ]

    for i, record in enumerate(records):
        tool = record["tool_name"]
        execution = record["execution"]
        cmd = execution["exact_command"]
        rc = execution.get("return_code", 0)
        inputs = execution.get("input_files", [])
        outputs = execution.get("output_files", [])

        lines.append(f"# Step {i + 1}: {tool}")
        if inputs:
            lines.append(f"#   inputs:  {', '.join(inputs)}")
        if outputs:
            lines.append(f"#   outputs: {', '.join(outputs)}")
        if deps[i]:
            dep_names = [records[d]["tool_name"] for d in deps[i]]
            lines.append(f"#   depends: {', '.join(dep_names)}")
        if rc != 0:
            lines.append(f"#   WARNING: original run exited with code {rc}")

        cmd_str = " ".join(_shell_quote(arg) for arg in cmd)
        lines.append(cmd_str)
        lines.append("")

    return "\n".join(lines)


def render_snakemake(records: list[dict], deps: dict[int, list[int]]) -> str:
    """Render the pipeline as a Snakemake workflow."""
    lines = [
        "# Snakemake workflow reconstructed from DSAgt execution records",
        "",
    ]

    rule_names = []
    for i, record in enumerate(records):
        tool = record["tool_name"]
        rule_name = f"{tool}_{i + 1}"
        rule_names.append(rule_name)

    all_outputs = []
    for record in records:
        all_outputs.extend(record["execution"].get("output_files", []))
    if all_outputs:
        lines.append("rule all:")
        lines.append("    input:")
        for f in all_outputs:
            lines.append(f'        "{f}",')
        lines.append("")

    for i, record in enumerate(records):
        execution = record["execution"]
        cmd = execution["exact_command"]
        inputs = execution.get("input_files", [])
        outputs = execution.get("output_files", [])

        lines.append(f"rule {rule_names[i]}:")
        if inputs:
            lines.append("    input:")
            for f in inputs:
                lines.append(f'        "{f}",')
        if outputs:
            lines.append("    output:")
            for f in outputs:
                lines.append(f'        "{f}",')

        cmd_str = " ".join(_shell_quote(arg) for arg in cmd)
        lines.append("    shell:")
        lines.append(f'        "{cmd_str}"')
        lines.append("")

    return "\n".join(lines)


def _shell_quote(s: str) -> str:
    """Quote a string for shell if it contains special characters."""
    if not s:
        return "''"
    safe = all(c.isalnum() or c in "-_./:=@+" for c in s)
    if safe:
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def reconstruct_pipeline(
    trace_dir: Path,
    session_id: str | None = None,
    fmt: str = "bash",
) -> str:
    """Reconstruct a pipeline from execution records.

    Parameters
    ----------
    trace_dir : Path
        Path to the trace_archive directory.
    session_id : str, optional
        Filter records to a specific session.
    fmt : str
        Output format: "bash" or "snakemake".

    Returns
    -------
    str
        The rendered pipeline script.
    """
    records = load_pipeline_records(trace_dir, session_id)
    if not records:
        return f"# No execution records found{' for session ' + session_id if session_id else ''}\n"

    deps = build_dependency_graph(records)

    if fmt == "snakemake":
        return render_snakemake(records, deps)
    return render_bash(records, deps)


# ---------------------------------------------------------------------------
# LLM conversation tracking (LiteLLM callback)
# ---------------------------------------------------------------------------

SESSION_LOG_FILE = "session_log.jsonl"


class ToolRecordStore:
    """Manages tool execution records and session logging on disk.

    Tracks tool_use blocks from LLM responses, matches them against
    tool_result blocks in subsequent requests, and writes the combined
    intent + report record. dsagt-run optionally fills in the execution layer.
    """

    def __init__(self, records_dir: str | Path, session_id: str | None = None):
        self.records_dir = Path(records_dir)
        self.session_id = session_id or os.environ.get("DSAGT_SESSION_ID")

        self._pending: dict[str, dict] = {}
        self._logged_message_count: int = 0
        # Cursor for match_tool_results — only the suffix of the messages list
        # past this index needs to be re-scanned on each call.  Without this,
        # match_tool_results was O(n²) over the session: every callback
        # iterated the entire growing message history.
        self._matched_message_count: int = 0
        self._extraction_threshold: int = int(
            os.environ.get("DSAGT_EXTRACTION_THRESHOLD", "0")
        )
        self._exchange_count: int = 0
        self._extracting: bool = False
        self._last_injected_suggestion_count: int = 0

        # Long-lived session log handle.  Opened lazily on the first write so
        # ToolRecordStore can be constructed in tests / unusual lifecycles
        # without hitting the disk.  Keeping a single handle open across the
        # proxy lifetime saves an open() + close() syscall pair on every LLM
        # exchange — meaningful because log_exchange runs on the proxy
        # callback hot path.
        self._session_log_handle = None

        self.records_dir.mkdir(parents=True, exist_ok=True)

    def track_tool_uses(self, response_data: dict) -> list[dict]:
        """Extract tool_use/tool_calls from a response and track them."""
        tracked = []
        now = datetime.now(timezone.utc).isoformat()

        for choice in response_data.get("choices", []):
            message = choice.get("message") or {}
            for tc in message.get("tool_calls") or []:
                func = tc.get("function", {})
                entry = {
                    "tool_use_id": tc.get("id"),
                    "tool_name": func.get("name"),
                    "parameters": _parse_arguments(func.get("arguments", "{}")),
                    "timestamp_requested": now,
                }
                self._pending[entry["tool_use_id"]] = entry
                tracked.append(entry)

        for block in response_data.get("content", []):
            if block.get("type") == "tool_use":
                entry = {
                    "tool_use_id": block["id"],
                    "tool_name": block["name"],
                    "parameters": block.get("input", {}),
                    "timestamp_requested": now,
                }
                self._pending[entry["tool_use_id"]] = entry
                tracked.append(entry)

        for t in tracked:
            logger.info("Tracking tool_use: %s (%s)", t["tool_name"], t["tool_use_id"][:12])

        return tracked

    def match_tool_results(self, messages: list[dict]) -> list[Path]:
        """Find tool_result/tool-role messages and write execution records.

        Only iterates the suffix of *messages* past the cursor advanced on
        the previous call.  Chat history is append-only, so any tool_result
        not seen yet must live in the new tail.
        """
        paths = []
        new_messages = messages[self._matched_message_count:]
        self._matched_message_count = len(messages)

        for msg in new_messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id in self._pending:
                    intent = self._pending.pop(tool_call_id)
                    path = self._write_proxy_record(intent, msg.get("content", ""))
                    paths.append(path)
                continue

            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if tool_use_id and tool_use_id in self._pending:
                    intent = self._pending.pop(tool_use_id)
                    result_text = _extract_tool_result_text(block)
                    path = self._write_proxy_record(intent, result_text)
                    paths.append(path)

        return paths

    def _write_proxy_record(self, intent: dict, agent_output: str) -> Path:
        """Write a tool execution record with intent + report layers."""
        record = {
            "record_id": intent["tool_use_id"],
            "tool_name": intent["tool_name"],
            "session_id": self.session_id,
            "intent": {
                "command": intent["tool_name"],
                "parameters": intent["parameters"],
                "timestamp_requested": intent["timestamp_requested"],
                "session_id": self.session_id,
            },
            "execution": None,
            "report": {
                "agent_output": agent_output,
                "timestamp_reported": datetime.now(timezone.utc).isoformat(),
                "wrapper_used": False,
            },
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid = intent["tool_use_id"][:12]
        filename = f"{intent['tool_name']}_{ts}_{rid}.json"
        path = self.records_dir / filename
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        logger.info("Execution record: %s", filename)
        return path

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def session_log_path(self) -> Path:
        return self.records_dir / SESSION_LOG_FILE

    def log_exchange(self, kwargs: dict, response_data: dict) -> None:
        """Append this LLM exchange to the session log."""
        messages = kwargs.get("messages", [])
        new_messages = messages[self._logged_message_count:]
        self._logged_message_count = len(messages)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": kwargs.get("litellm_call_id", ""),
            "model": kwargs.get("model", ""),
            "new_messages": new_messages,
            "response": _extract_response_content(response_data),
        }

        # Lazy-open the session log handle on first write.  Line-buffered
        # so each exchange is durable on disk after the trailing "\n" without
        # needing an explicit flush.
        if self._session_log_handle is None:
            self._session_log_handle = open(
                self.session_log_path, "a", buffering=1, encoding="utf-8",
            )
        self._session_log_handle.write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )

        self._exchange_count += 1
        if (
            self._extraction_threshold > 0
            and self._exchange_count >= self._extraction_threshold
            and not self._extracting
        ):
            self._trigger_extraction()

    def close(self) -> None:
        """Close the session log handle if it was opened.

        Idempotent and safe to call from atexit hooks or shutdown paths.
        Line buffering means data is already on disk; this just releases
        the file descriptor.
        """
        if self._session_log_handle is not None:
            try:
                self._session_log_handle.close()
            except Exception as e:
                logger.debug("ToolRecordStore.close: %s", e)
            self._session_log_handle = None

    def _trigger_extraction(self) -> None:
        """Run extraction in a background thread so the proxy isn't blocked."""
        import threading

        project_name = os.environ.get("DSAGT_PROJECT")
        if not project_name:
            logger.warning("DSAGT_PROJECT not set, skipping volume-triggered extraction")
            return

        self._extracting = True
        self._exchange_count = 0

        def _run():
            try:
                from dsagt.session import run_extraction
                result = run_extraction(project_name)
                logger.info("Volume-triggered extraction: %s", result.get("status"))
            finally:
                self._extracting = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def pending_injection(self) -> str | None:
        """Check for pending suggestions and return an injection message."""
        suggestions_path = self.records_dir.parent / "suggestions.json"
        if not suggestions_path.exists():
            return None

        suggestions = json.loads(suggestions_path.read_text())
        if not suggestions or len(suggestions) == self._last_injected_suggestion_count:
            return None

        self._last_injected_suggestion_count = len(suggestions)

        count = len(suggestions)
        previews = []
        for s in suggestions[:3]:
            previews.append(f"  - [{s.get('category', '')}] {s.get('text', '')[:80]}")
        preview_text = "\n".join(previews)
        more = f"\n  ... and {count - 3} more" if count > 3 else ""

        return (
            f"[DSAgt Memory System] {count} new observation(s) were flagged as "
            f"potentially important during this session:\n{preview_text}{more}\n"
            f"Call kb_get_suggestions to review them with the user."
        )


# ---------------------------------------------------------------------------
# LLM callback helpers
# ---------------------------------------------------------------------------

def _parse_arguments(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def _extract_tool_result_text(block: dict) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(texts)
    return str(content)


def _extract_response_content(response_data: dict) -> list[dict]:
    if "content" in response_data and isinstance(response_data["content"], list):
        return response_data["content"]

    for choice in response_data.get("choices", []):
        msg = choice.get("message") or {}
        blocks = []
        if msg.get("content"):
            blocks.append({"type": "text", "text": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            blocks.append({
                "type": "tool_use",
                "name": func.get("name", ""),
                "input": _parse_arguments(func.get("arguments", "{}")),
            })
        if blocks:
            return blocks

    return []


def _response_to_dict(response_obj) -> dict:
    if isinstance(response_obj, dict):
        return response_obj
    for method in ("model_dump", "dict", "to_dict"):
        fn = getattr(response_obj, method, None)
        if fn:
            return fn()
    return {"_repr": str(response_obj)}


# ---------------------------------------------------------------------------
# LiteLLM callback factory
# ---------------------------------------------------------------------------

def create_callback(records_dir: str | Path, session_id: str | None = None):
    """Create a DSAGT callback instance for LiteLLM.

    Imports litellm lazily so the rest of the module is testable without it.
    """
    import atexit

    from litellm.integrations.custom_logger import CustomLogger

    store = ToolRecordStore(records_dir, session_id)
    # Release the session log file descriptor on proxy shutdown.
    atexit.register(store.close)

    class DSAGTCallback(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            injection = store.pending_injection()
            if injection:
                messages.insert(0, {"role": "system", "content": injection})
                logger.info("Injected suggestion prompt (%d chars)", len(injection))
            _inject_cache_breakpoints(messages, kwargs)

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _handle_success(store, kwargs, response_obj, start_time, end_time)
            _tag_mlflow_session(session_id)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            _handle_success(store, kwargs, response_obj, start_time, end_time)
            _tag_mlflow_session(session_id)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            logger.warning("LLM call failed: model=%s", kwargs.get("model"))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            logger.warning("LLM call failed: model=%s", kwargs.get("model"))

    return DSAGTCallback()


# Marker LiteLLM forwards through to Anthropic-family providers (Anthropic
# direct, Bedrock-Claude) as their `cache_control` field on a content block
# or tool definition.  Providers without prompt caching (current OpenAI
# automatic caching ignores explicit markers, Cohere/Mistral/etc just drop
# them) treat the field as a no-op, so this is safe to set unconditionally.
_CACHE_MARKER = {"type": "ephemeral"}


def _inject_cache_breakpoints(messages: list, kwargs: dict) -> None:
    """Mark the largest stable request prefix as cacheable.

    Anthropic prompt caching keys on the prefix UP TO each marked block, so
    one marker at the end of the tools array caches "system + tools", and a
    second on the system message itself ensures the system block is cached
    even on requests with no tools.  Subsequent turns within the 5-minute
    TTL pay 10% on the cached prefix instead of 100%.

    Mutates ``messages`` and ``kwargs`` in place — the proxy reads the same
    objects on its way to the upstream call.
    """
    # 1) Tools: stamp the last tool definition.  In LiteLLM's OpenAI-shape
    #    the tool dict carries cache_control as a top-level key and the
    #    Anthropic translator picks it up.
    tools = kwargs.get("tools") or []
    if tools and isinstance(tools[-1], dict):
        tools[-1]["cache_control"] = _CACHE_MARKER

    # 2) System message: stamp the last text block.  For string content we
    #    promote to block format because cache_control lives on the block,
    #    not the message itself.
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": _CACHE_MARKER,
            }]
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = _CACHE_MARKER
        break


def install_mlflow_logger_with_session_tag() -> None:
    """Inject a MlflowLogger subclass that stamps ``mlflow.trace.session``.

    Why a subclass instead of post-hoc tagging from another callback:
    LiteLLM's MlflowLogger does the entire trace lifecycle (start_trace →
    set_attributes → end_trace) inside one ``_handle_success`` call.  By
    the time any sibling success-callback fires, the trace is already
    exported and ``mlflow.update_current_trace`` has nothing to update.
    Subclassing lets us slot the metadata write between trace creation and
    trace export, where the in-memory trace still exists and is mutable.

    Why register via LiteLLM's ``_in_memory_loggers`` cache: when the
    proxy resolves the string ``"mlflow"`` from ``success_callback``, it
    iterates ``_in_memory_loggers`` looking for any ``isinstance(_,
    MlflowLogger)`` and returns it instead of constructing a fresh one
    (litellm_logging.py ~line 4111).  A subclass passes the isinstance
    check, so pre-seeding our subclass means the existing string-based
    registration in ``success_callback``/``failure_callback`` automatically
    routes through us — no sync-vs-async dispatch quirks to worry about.

    Idempotent: safe to call more than once.
    """
    from litellm.integrations.mlflow import MlflowLogger
    from litellm.litellm_core_utils import litellm_logging as _ll

    from dsagt.observability import _stamp_metadata_on_trace

    class _DSAGTMlflowLogger(MlflowLogger):
        def _start_span_or_trace(self, kwargs, start_time):
            span = super()._start_span_or_trace(kwargs, start_time)
            # span.request_id is MLflow's trace_id; the trace lives in
            # InMemoryTraceManager until the parent _handle_success calls
            # _end_span_or_trace, so we have a window here to mutate
            # trace_metadata before export.
            #
            # Agent-turn LLM calls land here (proxy path).  Stamp:
            #   mlflow.trace.session — session grouping in the UI
            #   dsagt.source=agent   — distinguishes from extraction/embedding
            #   dsagt.agent          — which platform (goose, claude-code, ...)
            # Non-agent LLM calls (memory extraction, embeddings) go through
            # llm_source(...) decorators and never touch this subclass, so
            # hard-coding source="agent" here is safe.
            if span is None:
                return span
            metadata: dict[str, str] = {"dsagt.source": "agent"}
            if session_id := os.environ.get("DSAGT_SESSION_ID"):
                metadata["mlflow.trace.session"] = session_id
            if agent := os.environ.get("DSAGT_AGENT"):
                metadata["dsagt.agent"] = agent
            _stamp_metadata_on_trace(span.request_id, metadata)
            return span

    # Already installed? Leave it.
    for cb in _ll._in_memory_loggers:
        if type(cb).__name__ == "_DSAGTMlflowLogger":
            return
    # Drop any vanilla MlflowLogger that beat us to the cache.
    _ll._in_memory_loggers[:] = [
        cb for cb in _ll._in_memory_loggers
        if not (isinstance(cb, MlflowLogger) and type(cb).__name__ != "_DSAGTMlflowLogger")
    ]
    _ll._in_memory_loggers.append(_DSAGTMlflowLogger())


def _tag_mlflow_session(session_id: str | None) -> None:
    """Stamp the current MLflow trace with the reserved session key.

    Runs per-LLM-call because ``mlflow.litellm.autolog`` creates one trace per
    invocation; ``update_current_trace`` only applies to the active trace, so
    tagging has to happen while that trace is still open (i.e. from the
    LiteLLM success callback, before the autolog span closes).

    Silently no-ops if MLflow isn't installed or no trace is active — the
    proxy still works without MLflow.
    """
    if not session_id:
        return
    try:
        import mlflow
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id},
        )
    except Exception as e:
        logger.debug("mlflow.trace.session tag failed: %s", e)


def _handle_success(store: ToolRecordStore, kwargs, response_obj, start_time, end_time):
    from dsagt.observability import record_sidechannel_call as _record_sidechannel

    messages = kwargs.get("messages", [])
    response_data = _response_to_dict(response_obj)
    store.log_exchange(kwargs, response_data)
    store.match_tool_results(messages)
    store.track_tool_uses(response_data)
    _log_cache_usage(response_data)
    _record_sidechannel(store.records_dir, kwargs)


def _log_cache_usage(response_data: dict) -> None:
    """Emit a one-line summary of prompt-cache hits/misses per completion.

    Anthropic-family responses carry ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` under ``usage``; providers without caching
    omit both fields (log line prints zeros and the user knows the marker
    was dropped).  Cheap to leave on — one logger.info per LLM call.
    """
    usage = response_data.get("usage") or {}
    if not usage:
        return
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    logger.info(
        "tokens: prompt=%d completion=%d  cache: read=%d write=%d",
        prompt, completion, cache_read, cache_write,
    )
