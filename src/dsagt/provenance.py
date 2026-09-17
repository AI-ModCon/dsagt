"""
Provenance for code executions.

**Execution capture** (dsagt-run wrapper):
    Wraps a shell command, captures exact execution data (command, exit
    code, stdout/stderr, input/output files), and writes a JSON record
    to ``trace_archive/<code>_<ts>_<id>.json``.

**Record indexing** (ChromaDB):
    Indexes execution records into a ``code_use`` collection for
    semantic search and metadata filtering.

**Pipeline reconstruction**:
    Reads execution records, builds a dependency graph from input/output
    file overlap, and renders as a bash script or Snakemake workflow.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only (this module has ``from __future__ import annotations``, so
    # the hint is a string).  Importing KnowledgeBase at runtime would drag the
    # whole retrieval module into ``dsagt-run``, which only writes provenance
    # records to disk and never touches a KB — the embedding of those records
    # happens later, on the MCP server's periodic pass via ``CodeUseIndexer``.
    from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

#: Project-local collection of indexed code-execution records.
CODE_USE_COLLECTION = "code_use"


# ---------------------------------------------------------------------------
# Execution capture (dsagt-run)
# ---------------------------------------------------------------------------


def _resolve_records_dir(explicit: str | None) -> Path:
    """Determine the records directory.

    Priority: explicit ``--records-dir`` flag → ``$DSAGT_PROJECT_DIR``
    (exported by ``dsagt start`` and the MCP env block) → the cwd.  The
    directory must hold ``.dsagt/config.yaml``, the project config
    ``dsagt init`` writes.  The project is a fixed place, the agent's
    working directory, so there is no walk up the tree: a ``cd`` into a
    subdirectory before the command is the error, and the message names it.
    """
    if explicit:
        return Path(explicit)
    env_dir = os.environ.get("DSAGT_PROJECT_DIR")
    if env_dir:
        project, source = Path(env_dir).resolve(), "DSAGT_PROJECT_DIR"
    else:
        project, source = Path.cwd().resolve(), "the working directory"
    if not (project / ".dsagt" / "config.yaml").exists():
        raise ValueError(
            f"{source} ({project}) is not a dsagt project: no .dsagt/config.yaml. "
            "Run dsagt-run from the project directory, or pass --records-dir."
        )
    return project / "trace_archive"


def _current_session_tag_from_cwd() -> str | None:
    """Read the current session tag from ``<cwd>/.dsagt/state.yaml``.

    ``dsagt-run`` runs with cwd == project dir; the MCP server (also a child
    of the agent) minted the session into ``state.yaml`` at startup.  Lazy
    import of ``session`` avoids a circular import (``session`` imports this
    module for ``index_trace_archive``).
    """
    from dsagt import session

    cwd = Path.cwd().resolve()
    cfg = session.read_config_file(cwd)
    project = cfg.get("project")
    if not project:
        return None
    return session.current_session_tag(cwd, project)


def _parse_file_list(raw: str | None) -> list[str]:
    """Split a comma-separated file list, stripping whitespace."""
    if not raw:
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def _child_env() -> dict[str, str]:
    """Environment for the code's process: the caller's, with the directory
    of dsagt's own interpreter appended to PATH.

    A tool installer (pipx, ``uv tool install``) links only dsagt's commands
    onto PATH; a CLI that is a dsagt dependency, such as ``aidrin``, is in
    that private environment's bin directory.  Appended, not prepended, so
    every command already on PATH resolves as it did before.
    """
    env = dict(os.environ)
    bin_dir = str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        env["PATH"] = f"{path}{os.pathsep}{bin_dir}" if path else bin_dir
    return env


def _pump(source, sink, lines: list[str]) -> None:
    """Copy *source* to *sink* line by line, keeping every line in *lines*."""
    for line in iter(source.readline, ""):
        lines.append(line)
        sink.write(line)
        sink.flush()


def _run_streaming(command: list[str]) -> tuple[int, str, str]:
    """Run *command*, echoing its output as it arrives, and return the exit
    code with the full stdout and stderr.

    The child's two pipes are read on two threads, so a command that fills
    one while the other is being read cannot block.  Undecodable bytes are
    replaced, so a stray byte in a tool's log cannot lose the record of
    the run.  Raises ``FileNotFoundError`` when the executable is absent.
    """
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=_child_env(),
    )
    out_lines: list[str] = []
    err_lines: list[str] = []
    readers = [
        threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_lines)),
        threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_lines)),
    ]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join()
    return proc.wait(), "".join(out_lines), "".join(err_lines)


def run_and_record(
    code_name: str,
    command: list[str],
    records_dir: Path,
    session_id: str | None = None,
    record_id: str | None = None,
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
) -> int:
    """Execute a command, write an execution record, return the exit code.

    The command's output is echoed as it arrives and kept in full for the
    record, so a slow code shows progress and the record still holds
    everything it printed.
    """
    from dsagt.observability import obs, code_execute_span, truncate

    record_id = record_id or uuid.uuid4().hex[:12]
    if session_id is None:
        # The MCP server mints the session at startup and records it in
        # ``.dsagt/state.yaml``; read the current tag from there so this
        # code span buckets with the rest of the session (cwd == project
        # dir by contract).  ``None`` if no session has been minted yet.
        session_id = _current_session_tag_from_cwd()

    with code_execute_span(record_id, code_name):
        timestamp_start = datetime.now(timezone.utc).isoformat()
        start_perf = time.perf_counter()

        try:
            return_code, stdout, stderr = _run_streaming(command)
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
        obs.set_many(
            {
                "exit_code": return_code,
                "duration_ms": duration_ms,
                "n_input_files": len(input_files or []),
                "n_output_files": len(output_files or []),
                "command": truncate(" ".join(command), 256),
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
            }
        )
        if stderr.strip():
            obs.set("stderr_truncated", truncate(stderr, 256))
        if return_code != 0:
            obs.event("code_failed", exit_code=return_code)
            obs.set_status("ERROR")

        # Populate the MLflow trace UI's Input/Output tabs.  Truncate to
        # ~4KB per side so big code results don't bloat the trace store.  The
        # span is a preview by contract: the full stdout/stderr is in
        # trace_archive/<code>_<ts>_<record_id>.json on the machine that ran
        # the code, findable by the span's ``record_id`` attribute — and that
        # file is the only full copy, including when the store is a shared
        # server that other people read.
        obs.set_inputs(
            {
                "code": code_name,
                "command": list(command),
                "input_files": input_files or [],
            }
        )
        obs.set_outputs(
            {
                "exit_code": return_code,
                "duration_ms": duration_ms,
                "stdout": truncate(stdout, 4096),
                "stderr": truncate(stderr, 4096) if stderr else "",
                "output_files": output_files or [],
            }
        )

    record = {
        "record_id": record_id,
        "code_name": code_name,
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
    return return_code


def _write_record(record: dict, records_dir: Path) -> Path:
    """Write a JSON execution record. Returns the file path."""
    records_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{record['code_name']}_{ts}_{record['record_id']}.json"
    path = records_dir / filename

    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Record indexing (ChromaDB)
# ---------------------------------------------------------------------------


def render_execution_text(record: dict) -> str:
    """Convert a code execution record into embeddable natural-language text."""
    code_name = record.get("code_name", "unknown")
    execution = record.get("execution")

    parts = [f"Code: {code_name}"]

    if execution and execution.get("exact_command"):
        cmd = execution["exact_command"]
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        parts.append(f"Command: {cmd}")

    if execution and execution.get("return_code") is not None:
        rc = execution["return_code"]
        status = "succeeded" if rc == 0 else f"failed (exit code {rc})"
        parts.append(f"Outcome: {status}")

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
    """Extract ChromaDB-filterable metadata from a code execution record."""
    execution = record.get("execution")

    meta: dict = {}
    meta["code_name"] = record.get("code_name") or "unknown"
    # A code run outside a minted session stores session_id: null, and ChromaDB
    # rejects a null metadata value — which would fail the whole batch add and
    # re-fail every pass.  Coerce null to "unknown".
    meta["session_id"] = record.get("session_id") or "unknown"

    if execution and execution.get("return_code") is not None:
        meta["return_code"] = execution["return_code"]

    if execution and execution.get("timestamp_start"):
        meta["timestamp"] = execution["timestamp_start"]

    record_id = record.get("record_id", "")
    if record_id:
        meta["record_id"] = record_id

    return meta


def index_trace_archive(
    trace_dir: Path,
    kb: KnowledgeBase,
    indexed_ids: set[str] | None = None,
    *,
    source: str | None = None,
) -> dict:
    """Batch-index all code execution records in a trace archive directory."""
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
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A truncated/corrupt record must not abort the whole batch — it
            # persists on disk and would re-fail every pass.  Skip it,
            # consistent with the missing-execution-layer skip below.
            logger.warning("Skipping %s: unreadable record", path.name)
            errors += 1
            continue

        record_id = record.get("record_id", "")
        if record_id and record_id in indexed_ids:
            skipped += 1
            continue

        if record.get("execution") is None:
            logger.warning("Skipping %s: no execution layer", path.name)
            errors += 1
            continue

        texts.append(render_execution_text(record))
        metadatas.append(execution_metadata(record))

        if record_id:
            indexed_ids.add(record_id)

    # No ``route=`` — see ``index_execution_record`` above.
    if texts:
        from contextlib import nullcontext

        from dsagt.observability import open_span

        # Open a categorization root only when there is work to index — a quiet
        # periodic pass produces no child spans, so wrapping it would just emit an
        # empty, null-request trace.  Only the tagged background triggers pass a
        # ``source``; the reconstruct-pipeline caller passes none and lets its
        # kb.* writes inherit the tool's own trace.
        cm = open_span("code_use.index", source=source) if source else nullcontext(None)
        with cm as span:
            kb.add_entries(
                texts=texts,
                collection=CODE_USE_COLLECTION,
                metadatas=metadatas,
            )
            if span is not None:
                span.set_inputs({"trace_dir": str(trace_dir), "n_records": len(texts)})
                span.set_outputs({"indexed": len(texts)})

    return {
        "indexed": len(texts),
        "skipped": skipped,
        "errors": errors,
        "total_files": len(json_files),
    }


class CodeUseIndexer:
    """Idempotent, incremental indexer of ``dsagt-run`` records into ``code_use``.

    The code-execution counterpart to :class:`~dsagt.trace_scan.TraceScan`:
    ``dsagt-run`` writes one JSON record per call to ``trace_archive/``, and each
    :meth:`tick` embeds only the records not already indexed — tracked by
    ``record_id`` in a persisted ack set — so re-ticks and cross-session
    re-reads can never duplicate (the bug the prior cursor-less batch had).

    One primitive, three triggers, all safe to overlap: the MCP server's periodic pass
    (current-session freshness), startup catch-up (the previous session's tail),
    and the ``reconstruct_pipeline`` code (index-then-reconstruct, so a pipeline
    review reflects the calls just made).  An OS file lock around
    load→index→save serializes those callers — distinct instances in one
    process, or a future cross-process ticker — against the shared ack file.
    """

    def __init__(self, kb: KnowledgeBase, project_dir: str | Path):
        self._kb = kb
        pdir = Path(project_dir)
        self._trace_dir = pdir / "trace_archive"
        self._acks_path = pdir / ".dsagt" / "code_use_acks.json"

    def _load_acks(self) -> set[str]:
        try:
            return set(json.loads(self._acks_path.read_text()))
        except FileNotFoundError:
            return set()
        except (json.JSONDecodeError, ValueError):
            # A truncated/corrupt ack file must not stall indexing every tick;
            # treat it as empty and let the next _save_acks rewrite it.
            logger.warning("Corrupt %s; treating as empty", self._acks_path.name)
            return set()

    def _save_acks(self, acks: set[str]) -> None:
        self._acks_path.parent.mkdir(parents=True, exist_ok=True)
        self._acks_path.write_text(json.dumps(sorted(acks)))

    @contextmanager
    def _lock(self):
        self._acks_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._acks_path.with_suffix(".lock")
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def tick(self, *, source: str | None = None) -> int:
        """Index newly-arrived records; return how many were indexed this tick.

        Passing ``source`` opens a ``dsagt.source=<source>`` categorization root
        around the actual indexing (see :func:`index_trace_archive`) — but only
        when records are indexed.  At the ``reconstruct_pipeline`` call site this
        runs *inside* the registry tool's trace with no source, so its ``kb.*``
        writes correctly inherit ``dsagt.source=registry``.  The background
        callers use :meth:`tick_traced` so their writes don't orphan as untagged
        roots.
        """
        with self._lock():
            acks = self._load_acks()
            before = len(acks)
            # index_trace_archive skips record_ids already in ``acks`` and adds
            # the newly-indexed ones to it (mutates the set we pass).
            result = index_trace_archive(
                self._trace_dir, self._kb, indexed_ids=acks, source=source
            )
            if len(acks) != before:
                self._save_acks(acks)
            return result.get("indexed", 0)

    def tick_traced(self) -> int:
        """:meth:`tick` under a ``dsagt.source=code_use`` categorization root.

        For the background triggers (the periodic pass, startup catch-up) that run off
        any tool-call trace — otherwise the indexer's ``kb.add_entries`` /
        ``kb.embed`` spans start their own untagged top-level traces, landing in
        the ``unknown`` bucket and detached from the executions they index.  The
        root is opened only when a tick actually indexes records, so a pass with
        no new records emits no empty trace.  Runs on the caller's thread (callers
        dispatch *this* to the embedding worker), so the span opens there.
        """
        return self.tick(source="code_use")


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
        code = record["code_name"]
        execution = record["execution"]
        cmd = execution["exact_command"]
        rc = execution.get("return_code", 0)
        inputs = execution.get("input_files", [])
        outputs = execution.get("output_files", [])

        lines.append(f"# Step {i + 1}: {code}")
        if inputs:
            lines.append(f"#   inputs:  {', '.join(inputs)}")
        if outputs:
            lines.append(f"#   outputs: {', '.join(outputs)}")
        if deps[i]:
            dep_names = [records[d]["code_name"] for d in deps[i]]
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
        code = record["code_name"]
        rule_name = f"{code}_{i + 1}"
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
