"""
Tool execution wrapper logic for provenance capture.

Wraps a shell command, captures exact execution data, and writes
a JSON execution record. The agent sees no difference — stdout, stderr,
and exit code pass through unchanged.

Execution records are written to <records-dir>/<tool>_<timestamp>_<id>.json.
"""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _resolve_records_dir(explicit: str | None) -> Path:
    """Determine the records directory from arg, env var, or default.

    Priority: explicit flag → $DSAGT_RECORDS_DIR → $DSAGT_PROJECT_DIR/trace_archive → fallback.
    """
    if explicit:
        return Path(explicit)
    from_env = os.environ.get("DSAGT_RECORDS_DIR")
    if from_env:
        return Path(from_env)
    project_dir = os.environ.get("DSAGT_PROJECT_DIR")
    if project_dir:
        return Path(project_dir) / "trace_archive"
    return Path("runtime/trace_archive")


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
    record_id = record_id or uuid.uuid4().hex[:12]
    session_id = session_id or os.environ.get("DSAGT_SESSION_ID")

    timestamp_start = datetime.now(timezone.utc).isoformat()

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
    except Exception as e:
        return_code = 1
        stdout = ""
        stderr = f"dsagt-run: execution error: {e}"

    timestamp_end = datetime.now(timezone.utc).isoformat()

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

    # Pass through stdout/stderr so the caller sees the same output
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
