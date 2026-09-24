"""
Provenance for code executions: the record, its index, and the script it replays.

``dsagt-run`` wraps a registered code's command, and what it leaves behind is
one JSON record per run in ``trace_archive/``: the command as argv, the exit
code, what it printed, and the files it read and wrote with a SHA-256 of each.
The record is the provenance; everything else in this module reads it.  The
MCP server's periodic pass embeds records into the ``code_use`` collection so
past runs are searchable, and ``reconstruct_pipeline`` turns a session's
records into a script that repeats the work on a fresh copy.

Which files a run touched is derived, not declared: the code's spec names its
input and output parameters, and each side the spec leaves empty is filled by
scanning the arguments.  That is why a command's files can be known without
the agent listing them, and why the scan has a section of its own below.

Nothing here imports a heavy dependency at module scope.  ``dsagt-run`` pays
this module's import on every recorded command, so ``KnowledgeBase`` is an
annotation under ``TYPE_CHECKING`` and MLflow is imported inside the two
functions that log a trace.

Sections (top to bottom)
------------------------
  which files      file_roles_from_command  (the spec's roles)
                   files_from_arguments · new_files_from_arguments  (the scan)
                   sha256_of  (what says a file changed since a run)
  running          _run_streaming  (echo as it arrives, keep it all, forward
                   a signal to the child so a killed run still has a record)
  the record       run_and_record ─┬─ _outputs_after_the_run
                                   ├─ _write_record
                                   └─ log_execution_trace_if_tracing
  the index        index_trace_archive · CodeUseIndexer  (the periodic pass)
                   render_execution_text · execution_metadata  (what a
                   record looks like to the embedder and to a filter)
  the script       reconstruct_pipeline ─┬─ load_pipeline_records
                                         ├─ build_dependency_graph
                                         └─ render_bash · render_snakemake
                   readiness_reports  (the aidrin records for one file)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import signal
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
    # the hint is a string).  Importing KnowledgeBase at runtime would load the
    # whole retrieval module into ``dsagt-run``, which writes provenance records
    # to disk; the MCP server's periodic pass embeds them through
    # ``CodeUseIndexer``.
    from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

#: Project-local collection of indexed code-execution records.


CODE_USE_COLLECTION = "code_use"


# -------------------------------------------------------------------------
# Which files a command read and wrote
# -------------------------------------------------------------------------


def file_roles_from_command(
    spec: dict, command: list[str]
) -> tuple[list[str], list[str]]:
    """The input and output files a command names, by the spec's parameter roles.

    A parameter whose ``role`` is ``input`` or ``output`` names a file; its
    value is read off *command* by the parameter's ``cli`` rendering: a
    ``--name``/``-n`` flag takes the next token, a glued ``--name=``/``-n=``
    flag carries its value, and ``positional[:N]`` is the Nth bare token
    after the spec's own executable tokens.  The command must start with the
    spec's executable (the part after ``dsagt-run --code <name> --``); any
    other command names nothing.  The agent records the mapping once at registration and the record gets
    its files on every run, which is what the dependency graph in
    :func:`build_dependency_graph` reads.
    """
    import shlex

    executable = spec.get("executable", "")
    marker = " -- "
    inner = (
        executable.split(marker, 1)[1]
        if executable.startswith("dsagt-run")
        else executable
    )
    prefix = _without_uv_wrapper(shlex.split(inner))
    command = _without_uv_wrapper(list(command))
    if command[: len(prefix)] != prefix:
        return [], []
    args = command[len(prefix) :]
    params = spec.get("parameters") or {}

    by_flag: dict[str, tuple[str, dict]] = {}
    positional_params: list[tuple[int | None, str, dict]] = []
    for name, param in params.items():
        cli = param.get("cli") or f"--{name}"
        if cli.startswith("positional"):
            index = int(cli.split(":", 1)[1]) if ":" in cli else None
            positional_params.append((index, name, param))
        else:
            by_flag[cli.rstrip("=")] = (name, param)

    values: dict[str, str] = {}
    positionals: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("-"):
            flag, glued, value = token.partition("=")
            if flag in by_flag:
                name, param = by_flag[flag]
                if glued:
                    values[name] = value
                elif param.get("type") != "boolean" and i + 1 < len(args):
                    values[name] = args[i + 1]
                    i += 1
            i += 1
            continue
        positionals.append(token)
        i += 1
    unindexed = [p for p in positional_params if p[0] is None]
    for slot, (index, name, param) in enumerate(positional_params):
        position = index if index is not None else unindexed.index((index, name, param))
        if position < len(positionals):
            values[name] = positionals[position]

    inputs = [
        values[n] for n, p in params.items() if p.get("role") == "input" and n in values
    ]
    outputs = [
        values[n]
        for n, p in params.items()
        if p.get("role") == "output" and n in values
    ]
    return inputs, outputs


def _without_uv_wrapper(tokens: list[str]) -> list[str]:
    """*tokens* with a leading ``uv run ... --`` removed.

    A spec with dependencies stores ``uv run --with <deps> -- <command>``;
    an agent that runs the command without the wrapper still ran this
    code, and the roles apply to the arguments either way.
    """
    if tokens[:2] == ["uv", "run"] and "--" in tokens:
        return tokens[tokens.index("--") + 1 :]
    return tokens


def files_from_arguments(command: list[str]) -> list[str]:
    """The arguments of *command* that are existing files or directories.

    A spec with no parameter roles names no files; an argument that exists when the command starts is one the
    command reads or overwrites, which is what the dependency graph and the
    readiness reports need to know.  A directory counts (a simulation case,
    a dataset directory); it gets no hash.  The executable is left out, and
    so is the script an interpreter runs (``python x.py data.csv`` reads
    ``data.csv``; ``x.py`` is the program, and as an input it would read as
    the product of whichever step wrote it).
    """
    # A spec with dependencies stores `uv run --with <deps> -- python x.py ...`;
    # the interpreter and its script are found past that wrapper.
    inner = _without_uv_wrapper(command)
    args = inner[1:]
    program = Path(inner[0]).name if inner else ""
    if program.startswith("python") or program in _INTERPRETERS:
        script = next((a for a in args if not a.startswith("-")), None)
        if script is not None and _is_file(script):
            args = [a for a in args if a != script]
    return [arg for arg in args if _is_file(arg) or _is_dir(arg)]


_INTERPRETERS = ("python", "python3", "bash", "sh", "zsh", "Rscript", "perl", "node")


def new_files_from_arguments(command: list[str], before: list[str]) -> list[str]:
    """The arguments of *command* that are files now and were not in *before*.

    The script an interpreter runs is left out, as in :func:`files_from_arguments`.
    """
    return [
        arg
        for arg in files_from_arguments(command)
        if _is_file(arg) and arg not in before
    ]


def _is_file(arg: str) -> bool:
    """Whether *arg* names a regular file; an argument the OS cannot stat (a
    script passed inline, longer than a path may be) is not one."""
    try:
        return Path(arg).is_file()
    except OSError:
        return False


def _is_dir(arg: str) -> bool:
    try:
        return Path(arg).is_dir()
    except OSError:
        return False


def sha256_of(path: str) -> str | None:
    """The SHA-256 of a regular file, or ``None`` for a path that is not one.

    Streamed in 1 MiB chunks; the record identifies each input and output by
    content so a later pass can tell whether a file changed since the run,
    which a timestamp cannot (a copy has a new mtime, a move keeps an old one).
    """
    file = Path(path)
    if not _is_file(path):
        return None
    digest = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -------------------------------------------------------------------------
# Running the command
# -------------------------------------------------------------------------


_FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def _forward_signal(proc: "subprocess.Popen", signum: int) -> None:
    """Send *signum* to *proc*, ignoring a child that has already exited.

    A harness signalling the process group reaches the child too, and it may
    exit before this runs.  ``send_signal`` then raises ``ProcessLookupError``
    inside the signal handler, where it would escape into whatever the main
    thread was running: the caller catches ``OSError`` around the whole run
    and would report exit 1 with no output for a command that finished.
    ``proc.wait`` already holds the real status.
    """
    try:
        proc.send_signal(signum)
    except ProcessLookupError:
        pass


def _run_streaming(
    command: list[str], *, parent: str | None = None
) -> tuple[int, str, str]:
    """Run *command*, echoing its output as it arrives, and return the exit
    code with the full stdout and stderr.

    The child's two pipes are read on two threads, so a command that fills
    one while the other is being read cannot block.  Undecodable bytes are
    replaced, so a stray byte in a tool's log cannot lose the record of
    the run.  A SIGTERM, SIGINT, or SIGHUP to this process is
    forwarded to the child and the call returns the child's exit status
    (negative, the signal number, as ``subprocess`` reports it), so the
    caller writes the record for a run that was ended from outside; a
    headless harness ends a turn that way.  Raises ``FileNotFoundError``
    when the executable is absent.
    """
    env = _child_env()
    if parent:
        env["DSAGT_RUN_PARENT"] = parent
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=env,
    )
    forwarded: list[int] = []

    def forward(signum, _frame):
        forwarded.append(signum)
        _forward_signal(proc, signum)

    previous = {sig: signal.signal(sig, forward) for sig in _FORWARDED_SIGNALS}
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
    try:
        return_code = proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if forwarded:
        name = signal.Signals(forwarded[0]).name
        err_lines.append(f"dsagt-run: terminated by {name}\n")
    return return_code, "".join(out_lines), "".join(err_lines)


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


# -------------------------------------------------------------------------
# Writing the record
# -------------------------------------------------------------------------


def log_execution_trace_if_tracing(record_path: Path) -> None:
    """Log the record's ``code.execute`` trace when this process has tracing on."""
    from dsagt import observability

    if observability._initialized:
        observability.log_execution_trace(json.loads(Path(record_path).read_text()))


def run_and_record(
    code_name: str,
    command: list[str],
    records_dir: Path,
    session_id: str | None = None,
    record_id: str | None = None,
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
    log_trace=log_execution_trace_if_tracing,
) -> int:
    """Execute a command, write an execution record, return the exit code.

    The command's output is echoed as it arrives and kept in full for the
    record, so a slow code shows progress and the record still holds
    everything it printed.

    The run loads no tracing library: the record is the provenance, and the
    ``code.execute`` trace is built from the record afterwards by *log_trace*,
    called with the record's path.  The default logs in this process when
    tracing is initialized here; ``dsagt-run`` passes a function that hands
    the record to a detached process, because loading the store costs about a
    second, which every recorded command would otherwise pay before it starts.
    """
    record_id = record_id or uuid.uuid4().hex[:12]
    input_files = list(input_files or [])
    output_files = list(output_files or [])
    # Each side the roles leave empty is filled from the arguments: a spec
    # whose flags differ from the ones the agent used (fastp's -i against
    # --in1) matches nothing on one side and must not silence the other.
    derive_inputs = not input_files
    derive_outputs = not output_files
    if derive_inputs:
        input_files = files_from_arguments(command)
    file_hashes = {f: sha256_of(f) for f in input_files}
    parent_record_id = os.environ.get("DSAGT_RUN_PARENT")
    if session_id is None:
        # The MCP server mints the session at startup and records it in
        # ``.dsagt/state.yaml``; read the current tag from there so this
        # code span buckets with the rest of the session (cwd == project
        # dir by contract).  ``None`` if no session has been minted yet.
        session_id = _current_session_tag_from_cwd()

    timestamp_start = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()

    try:
        return_code, stdout, stderr = _run_streaming(command, parent=record_id)
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
    output_files = _outputs_after_the_run(
        command,
        input_files,
        output_files,
        file_hashes,
        derive=derive_outputs,
        derived_inputs=derive_inputs,
        return_code=return_code,
    )
    for f in output_files:
        file_hashes[f] = sha256_of(f)
    file_hashes = {f: h for f, h in file_hashes.items() if h is not None}

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
            "duration_ms": duration_ms,
            "input_files": input_files,
            "output_files": output_files,
            "file_hashes": file_hashes,
        },
    }
    if parent_record_id:
        # A run started by a recorded run (a loop script over samples): the
        # parent's command replays it, so the reconstruction leaves it out.
        record["parent_record_id"] = parent_record_id

    path = _write_record(record, records_dir)
    if log_trace is not None:
        log_trace(path)
    return return_code


def _outputs_after_the_run(
    command: list[str],
    input_files: list[str],
    output_files: list[str],
    hashes_before: dict[str, str | None],
    *,
    derive: bool,
    derived_inputs: bool,
    return_code: int,
) -> list[str]:
    """The run's output files, once the command has exited.

    With no roles on the output side, an argument that did not exist before
    the run is an output, and so is one the run changed: a converter that
    replaces its output file names it on every run, not only the first, and
    *hashes_before* is what tells the two apart.  A file the command moved
    away is gone, so it is not an output.  A failed run keeps only the
    outputs that exist, so the record names no producer for a file that is
    not there.
    """
    if derive:
        rewritten = [
            f
            for f in input_files
            if derived_inputs
            and sha256_of(f) is not None
            and hashes_before.get(f) not in (None, sha256_of(f))
        ]
        output_files = [
            f
            for f in new_files_from_arguments(command, input_files) + rewritten
            if f not in output_files
        ] + output_files
    if return_code != 0:
        return [f for f in output_files if _is_file(f) or _is_dir(f)]
    return output_files


def _resolve_records_dir() -> Path:
    """The project's ``trace_archive/``.

    ``$DSAGT_PROJECT_DIR`` (exported by ``dsagt start`` and the MCP env
    block) names the project, and the working directory is it otherwise.  The
    directory must hold ``.dsagt/config.yaml``, the project config ``dsagt
    init`` writes.  The project is a fixed place, the agent's working
    directory, so the directory is checked as given: a ``cd`` into a
    subdirectory before the command is the error, and the message names it.
    """
    env_dir = os.environ.get("DSAGT_PROJECT_DIR")
    if env_dir:
        project, source = Path(env_dir).resolve(), "DSAGT_PROJECT_DIR"
    else:
        project, source = Path.cwd().resolve(), "the working directory"
    if not (project / ".dsagt" / "config.yaml").exists():
        raise ValueError(
            f"{source} ({project}) is not a dsagt project: no .dsagt/config.yaml. "
            "Run dsagt-run from the project directory."
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


def _write_record(record: dict, records_dir: Path) -> Path:
    """Write a JSON execution record. Returns the file path."""
    records_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{record['code_name']}_{ts}_{record['record_id']}.json"
    path = records_dir / filename

    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


# -------------------------------------------------------------------------
# Records into the code_use collection
# -------------------------------------------------------------------------


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

    if texts:
        from contextlib import nullcontext

        from dsagt.observability import open_span

        # Open a categorization root only when there is work to index: a quiet
        # periodic pass produces no child spans, and a root around it would be
        # an empty, null-request trace.  Only the tagged background triggers
        # pass a ``source``; the reconstruct-pipeline caller passes none, so
        # its kb.* writes inherit the tool's own trace.
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
    ``dsagt-run`` writes one JSON record per call to ``trace_archive/``, and
    each :meth:`tick` embeds only the records absent from a persisted ack set
    keyed by ``record_id``, so re-ticks and cross-session re-reads never
    duplicate an entry.

    One primitive, three triggers, all safe to overlap: the MCP server's
    periodic pass (current-session freshness), startup catch-up (the previous
    session's tail), and the ``reconstruct_pipeline`` code (index, then
    reconstruct, so a pipeline review includes the calls just made).  An OS
    file lock around load, index, and save serializes those callers, distinct
    instances in one process, against the shared ack file.
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

        Passing ``source`` opens a ``dsagt.source=<source>`` categorization
        root around the indexing (see :func:`index_trace_archive`), only when
        records are indexed.  At the ``reconstruct_pipeline`` call site this
        runs inside the registry tool's trace with no source, so its ``kb.*``
        writes inherit ``dsagt.source=registry``.  The background callers use
        :meth:`tick_traced` so their writes are tagged.
        """
        with self._lock():
            acks = self._load_acks()
            before = len(acks)
            # index_trace_archive skips record_ids already in ``acks`` and adds
            # the newly-indexed ones to it (it mutates the set passed in).
            result = index_trace_archive(
                self._trace_dir, self._kb, indexed_ids=acks, source=source
            )
            if len(acks) != before:
                self._save_acks(acks)
            return result.get("indexed", 0)

    def tick_traced(self) -> int:
        """:meth:`tick` under a ``dsagt.source=code_use`` categorization root.

        For the background triggers (the periodic pass, startup catch-up), which
        run outside any tool-call trace.  Without the root, the indexer's
        ``kb.add_entries`` / ``kb.embed`` spans start their own untagged
        top-level traces in the ``unknown`` bucket, detached from the executions
        they index.  The root is opened only when a tick indexes records, so a
        pass with no new records emits no empty trace.  Runs on the caller's
        thread (callers dispatch *this* to the embedding worker), so the span
        opens there.
        """
        return self.tick(source="code_use")


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
    # rejects a null metadata value, which would fail the whole batch add and
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


# -------------------------------------------------------------------------
# Records into a script
# -------------------------------------------------------------------------


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
    return render_bash(records, deps, project_dir=Path(trace_dir).resolve().parent)


def load_pipeline_records(trace_dir: Path, session_id: str | None = None) -> list[dict]:
    """Load execution records that have the wrapper execution layer.

    Returns records sorted by execution start time.
    """
    if not trace_dir.is_dir():
        return []

    loaded = []
    for path in trace_dir.glob("*.json"):
        raw = json.loads(path.read_text())
        if raw.get("execution"):
            loaded.append(raw)
    succeeded = {
        r["record_id"] for r in loaded if r["execution"].get("return_code", 0) == 0
    }
    records = []
    for raw in loaded:
        if session_id and raw.get("session_id") != session_id:
            continue
        if raw.get("parent_record_id") in succeeded:
            # Started by another recorded run, whose command replays it.  The
            # child of a run that failed or was killed is kept: the parent is
            # rendered as a comment, and the child is the work that was done.
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


def render_bash(
    records: list[dict],
    deps: dict[int, list[int]],
    project_dir: Path | None = None,
) -> str:
    """Render the pipeline as a bash script, in the order the records ran.

    A run that exited non-zero is kept as a comment: the script opens with
    ``set -e``, so a live failed step would stop it at that point, and the
    failed attempts are part of the record the reader may want.  Paths under
    *project_dir* are written relative to it, so the script runs from the
    project directory or another checkout of the same layout.

    The script creates the output directories it needs and runs the recorded
    commands.  It removes nothing: a recorded output was written by the code
    ``dsagt-run`` wrapped, not by dsagt, and no record holds a removal, so
    deleting one would act on the reader's data on the strength of an
    inference.  A step whose command refuses to overwrite an existing output
    fails with its own error and ``set -e`` stops the script there.
    """
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Pipeline reconstructed from DSAgt execution records, in the order they ran",
        "#",
        "# Reconstruction is a work in progress, tracked as a separate effort.",
        "# Read this script against your own data before running it: it replays",
        "# what the records hold, and a step the session prepared by hand is not",
        "# in them.",
        "",
    ]

    # The recorded commands assume the directories the session had made by
    # hand; on a fresh copy the script makes them first.
    output_dirs: list[str] = []
    for record in records:
        for f in record["execution"].get("output_files", []):
            parent = str(Path(_project_file(f, project_dir)).parent)
            if parent not in (".", "") and parent not in output_dirs:
                output_dirs.append(parent)
    output_dirs = [
        d for d in output_dirs if not any(o.startswith(d + "/") for o in output_dirs)
    ]
    if output_dirs:
        lines.append("mkdir -p " + " ".join(_shell_quote(d) for d in output_dirs))
        lines.append("")

    for i, record in enumerate(records):
        code = record["code_name"]
        execution = record["execution"]
        cmd = [_relative_to_project(a, project_dir) for a in execution["exact_command"]]
        rc = execution.get("return_code", 0)
        inputs = [
            _project_file(f, project_dir) for f in execution.get("input_files", [])
        ]
        outputs = [
            _project_file(f, project_dir) for f in execution.get("output_files", [])
        ]

        lines.append(f"# Step {i + 1}: {code}")
        if inputs:
            lines.append(f"#   inputs:  {', '.join(inputs)}")
        if outputs:
            lines.append(f"#   outputs: {', '.join(outputs)}")
        if deps[i]:
            dep_names = [records[d]["code_name"] for d in deps[i]]
            lines.append(f"#   depends: {', '.join(dep_names)}")

        cmd_str = " ".join(_shell_quote(arg) for arg in cmd)
        if rc != 0:
            lines.append(f"#   failed with exit code {rc}; kept as a comment")
            lines.append(f"# {cmd_str}")
        else:
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


def _relative_to_project(arg: str, project_dir: Path | None) -> str:
    """*arg* with a leading *project_dir* removed, so the script runs from the project."""
    if project_dir is None:
        return arg
    root = str(project_dir)
    if arg == root:
        return "."
    if arg.startswith(root + "/"):
        return arg[len(root) + 1 :]
    return arg


def _project_file(arg: str, project_dir: Path | None) -> str:
    """A recorded file path as it sits under the project, in one spelling.

    A record holds the path the agent typed, so one file arrives as
    ``data/x.csv`` in one record and ``./data/x.csv`` in the next, and
    comparing them as strings misses the match.  File paths go through this;
    a command argument does not, since it may be a flag or an empty string.
    """
    return os.path.normpath(_relative_to_project(arg, project_dir)) if arg else arg


def _shell_quote(s: str) -> str:
    """Quote a string for shell if it contains special characters."""
    if not s:
        return "''"
    safe = all(c.isalnum() or c in "-_./:=@+" for c in s)
    if safe:
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def readiness_reports(project_dir: Path, path: str) -> list[dict]:
    """The AI-readiness reports on record for *path*, newest first.

    Reads the ``aidrin`` records under ``<project>/trace_archive/`` whose
    inputs name *path* (given relative to the project, or absolute under it)
    and gives, per run, the record's id, the report (what the run printed,
    which the record holds in full), the run's start time, and whether the
    file's content is what it was at the run (``unchanged``), from the
    record's hash against the file now.  The readiness paragraph asks the
    agent to call this before a check, so the report after one stage is the
    report before the next.  A record with no
    hash for the file, from a run before hashes were recorded, reports
    ``unchanged`` as ``None``.
    """
    project_dir = Path(project_dir)
    target = _project_file(path, project_dir)
    current = sha256_of(str(project_dir / target))
    reports = []
    for record in load_pipeline_records(project_dir / "trace_archive"):
        if record.get("code_name") != "aidrin":
            continue
        execution = record["execution"]
        inputs = [
            _project_file(f, project_dir) for f in execution.get("input_files", [])
        ]
        if target not in inputs:
            continue
        recorded = execution.get("file_hashes", {})
        digest = next(
            (h for f, h in recorded.items() if _project_file(f, project_dir) == target),
            None,
        )
        reports.append(
            {
                "record_id": record["record_id"],
                "report": execution.get("stdout", ""),
                "timestamp": execution.get("timestamp_start"),
                "command": " ".join(execution.get("exact_command", [])),
                "unchanged": None if digest is None else digest == current,
            }
        )
    reports.reverse()
    return reports
