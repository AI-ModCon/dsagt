"""
dsagt-run: Tool execution wrapper for provenance capture.

Wraps a shell command, captures exact execution data, and writes
a JSON execution record. The agent sees no difference — stdout, stderr,
and exit code pass through unchanged.

Usage:
    dsagt-run --tool fastp -- fastp -q 20 -l 50 --in1 reads.fq.gz
    dsagt-run --tool megahit --session abc123 -- megahit -1 R1.fq -2 R2.fq -o out

Execution records are written to <records-dir>/<tool>_<timestamp>_<id>.json.
The records directory defaults to runtime/trace_archive/ or $DSAGT_RECORDS_DIR.
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse dsagt-run args and split off the wrapped command after '--'."""
    # Split on '--' ourselves since argparse.REMAINDER is fragile
    args_to_parse = argv if argv is not None else sys.argv[1:]

    try:
        sep = args_to_parse.index("--")
    except ValueError:
        # No '--' found — print help and exit
        _make_parser().parse_args(["--help"])
        sys.exit(1)  # unreachable, but explicit

    wrapper_args = args_to_parse[:sep]
    command_args = args_to_parse[sep + 1:]

    parsed = _make_parser().parse_args(wrapper_args)
    return parsed, command_args


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsagt-run",
        description="Wrap a tool command and capture execution provenance.",
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="Name of the tool being executed (must match registry).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID. Defaults to $DSAGT_SESSION_ID.",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Pre-assigned record ID (for proxy correlation). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--records-dir",
        default=None,
        help="Directory for execution records. Defaults to $DSAGT_RECORDS_DIR or runtime/trace_archive/.",
    )
    parser.add_argument(
        "--input-files",
        default=None,
        help="Comma-separated list of input file paths.",
    )
    parser.add_argument(
        "--output-files",
        default=None,
        help="Comma-separated list of output file paths.",
    )
    return parser


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


def main(argv: list[str] | None = None) -> int:
    args, command = _parse_args(argv)

    if not command:
        print("dsagt-run: no command specified after '--'", file=sys.stderr)
        return 1

    records_dir = _resolve_records_dir(args.records_dir)

    return run_and_record(
        tool_name=args.tool,
        command=command,
        records_dir=records_dir,
        session_id=args.session,
        record_id=args.record_id,
        input_files=_parse_file_list(args.input_files),
        output_files=_parse_file_list(args.output_files),
    )


if __name__ == "__main__":
    sys.exit(main())
