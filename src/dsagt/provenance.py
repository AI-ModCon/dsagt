"""
Provenance for tool executions.

**Execution capture** (dsagt-run wrapper):
    Wraps a shell command, captures exact execution data (command, exit
    code, stdout/stderr, input/output files), and writes a JSON record
    to ``trace_archive/<tool>_<ts>_<id>.json``.

**Record indexing** (ChromaDB):
    Indexes execution records into a ``tool_executions`` collection for
    semantic search and metadata filtering.

**Pipeline reconstruction**:
    Reads execution records, builds a dependency graph from input/output
    file overlap, and renders as a bash script or Snakemake workflow.

LLM-call provenance lives in MLflow (each MCP-server / agent process
autologs LiteLLM calls via ``init_tracing`` post-proxy-removal).
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

#: Project-local collection of indexed tool-execution records.
#: Renamed from ``tool_executions`` to match the user-facing
#: terminology ("tool_use index").
TOOL_USE_COLLECTION = "tool_use"

#: Backwards-compat alias.  New code should use ``TOOL_USE_COLLECTION``.
TOOL_EXECUTIONS_COLLECTION = TOOL_USE_COLLECTION
TOOL_EXECUTIONS_ROUTE = CollectionRoute(
    embedding_backend="api",
    vector_db="chroma",
    description="Indexed tool execution records from trace_archive.",
)


# ---------------------------------------------------------------------------
# Execution capture (dsagt-run)
# ---------------------------------------------------------------------------

def _resolve_records_dir(explicit: str | None) -> Path:
    """Determine the records directory.

    Priority: explicit ``--records-dir`` flag → ``<cwd>/trace_archive``,
    where cwd must contain ``dsagt_config.yaml`` (the project's
    single-source-of-truth config written by ``dsagt init``).  No env-var
    chain, no walking up the tree — if the agent's cwd isn't the project
    dir, that's the bug to fix, not something to recover from silently.
    """
    if explicit:
        return Path(explicit)
    cwd = Path.cwd().resolve()
    if not (cwd / "dsagt_config.yaml").exists():
        raise ValueError(
            f"No dsagt_config.yaml in cwd ({cwd}); pass --records-dir or "
            "run dsagt-run from a project directory."
        )
    return cwd / "trace_archive"


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

    # No ``route=`` — fall through to kb's default route so embedding
    # backend follows the project's ``embedding.backend`` config (BYOA
    # default = local sentence-transformers, no API call).
    return kb.add_entries(
        texts=[text],
        collection=TOOL_EXECUTIONS_COLLECTION,
        metadatas=[metadata],
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

    # No ``route=`` — see ``index_execution_record`` above.
    if texts:
        kb.add_entries(
            texts=texts,
            collection=TOOL_EXECUTIONS_COLLECTION,
            metadatas=metadatas,
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

