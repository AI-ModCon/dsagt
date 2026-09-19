"""
dsagt-run: execution wrapper for provenance capture.

Usage:
    dsagt-run --code fastp -- fastp -q 20 -l 50 --in1 reads.fq.gz
    dsagt-run -- python compare.py a.json b.json            # ad-hoc, no spec
    dsagt-run --code aidrin --stdout audit/pre.json -- aidrin data-quality f.csv
"""

import argparse
import fcntl
import sys
from pathlib import Path

from dsagt.provenance import (
    _current_session_tag_from_cwd,
    _parse_file_list,
    _resolve_records_dir,
    file_roles_from_command,
    run_and_record,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsagt-run",
        description="Wrap a code command and capture execution provenance.",
    )
    parser.add_argument(
        "--code",
        default=None,
        help="Name of the registered code being executed; omitted for an "
        "ad-hoc run, which is recorded without a spec.",
    )
    parser.add_argument(
        "--log-trace",
        default=None,
        metavar="RECORD",
        help=argparse.SUPPRESS,  # internal: log the trace of a written record
    )
    parser.add_argument(
        "--stdout",
        default=None,
        metavar="PATH",
        help="Write the command's stdout to this file and record it as an "
        "output; the terminal gets one line naming the file.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID. Defaults to the DSAGT_SESSION_ID env var.",
    )
    parser.add_argument("--record-id", default=None, help="Pre-assigned record ID.")
    parser.add_argument(
        "--records-dir", default=None, help="Directory for execution records."
    )
    parser.add_argument(
        "--input-files",
        default=None,
        help="Comma-separated input file paths; derived from the spec's "
        "parameter roles when omitted.",
    )
    parser.add_argument(
        "--output-files",
        default=None,
        help="Comma-separated output file paths; derived from the spec's "
        "parameter roles when omitted.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse dsagt-run args and split off the wrapped command after '--'."""
    args_to_parse = argv if argv is not None else sys.argv[1:]

    try:
        sep = args_to_parse.index("--")
    except ValueError:
        _make_parser().parse_args(["--help"])
        sys.exit(1)

    wrapper_args = args_to_parse[:sep]
    command_args = args_to_parse[sep + 1 :]

    parsed = _make_parser().parse_args(wrapper_args)
    return parsed, command_args


def log_trace(record_path: str, session_id: str | None) -> None:
    """Log the ``code.execute`` trace of the record at *record_path*.

    The second half of a ``dsagt-run``: it loads the user's service
    credentials and the trace store, which the run itself leaves unloaded,
    and writes the backdated span.  Run in a detached process by
    :func:`_log_trace_detached`.
    """
    import json

    from dsagt.observability import init_tracing, log_execution_trace
    from dsagt.session import load_user_env

    # The agent runs dsagt-run from its own shell, not as a child of
    # dsagt-server, so under codex and cline the credentials file is the only
    # way a shared-store key or URI reaches the trace.
    load_user_env()
    # One writer at a time: several runs in a second would otherwise race to
    # create a store that does not exist yet, and all but one would fail.
    with open(Path(".dsagt") / "run_trace.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        init_tracing("dsagt-run", session_id=session_id)
        with open(record_path) as fh:
            log_execution_trace(json.load(fh))
        import mlflow

        mlflow.flush_trace_async_logging()


def _log_trace_detached(session_id: str | None, project_dir: Path):
    """A ``log_trace`` for :func:`run_and_record` that starts a detached
    process for the trace and returns at once.

    Loading the store costs about a second; the command the agent asked for
    has already finished and its record is written, so the agent's shell gets
    its prompt back while the trace is logged.  The process has its own
    session, so a harness that ends the turn's process group leaves it alone.
    """
    import subprocess

    if not (project_dir / ".dsagt" / "config.yaml").exists():
        # Records written outside a project (an explicit --records-dir) have
        # no store to go to.
        return None

    def start(record_path) -> None:
        argv = [
            sys.executable,
            "-m",
            "dsagt.commands.run_code",
            "--log-trace",
            str(Path(record_path).resolve()),
        ]
        if session_id:
            argv += ["--session", session_id]
        # A failure to log the trace is written where a person can find it;
        # the record, which is the provenance, is already on disk.
        with open(project_dir / ".dsagt" / "run_trace.log", "a") as errors:
            subprocess.Popen(
                argv,
                cwd=project_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                start_new_session=True,
            )

    return start


def main(argv: list[str] | None = None) -> int:
    args_to_parse = argv if argv is not None else sys.argv[1:]
    if args_to_parse[:1] == ["--log-trace"]:
        parsed = _make_parser().parse_args(args_to_parse)
        log_trace(parsed.log_trace, parsed.session)
        return 0

    args, command = _parse_args(argv)

    if not command:
        print("dsagt-run: no command specified after '--'", file=sys.stderr)
        return 1
    if args.stdout is not None and args.stdout in command:
        # The shell would open the file for the redirect and the command
        # write it too; the replay of such a record truncates the output.
        print(
            f"dsagt-run: --stdout {args.stdout} is also an argument of the command; "
            "a code that writes its own output file needs no --stdout",
            file=sys.stderr,
        )
        return 2

    # The session is resolved here so the record and the trace both carry it.
    session_id = args.session or _current_session_tag_from_cwd()

    try:
        records_dir = _resolve_records_dir(args.records_dir)
    except ValueError as err:
        print(f"dsagt-run: {err}", file=sys.stderr)
        return 1

    input_files = _parse_file_list(args.input_files)
    output_files = _parse_file_list(args.output_files)
    if args.code and not input_files and not output_files:
        # The spec's parameter roles name the files; the flags are the
        # override for a command the roles cannot describe.
        from dsagt.registry import CodeRegistry

        spec = CodeRegistry(runtime_dir=records_dir.parent).get_code(args.code)
        if spec is not None:
            input_files, output_files = file_roles_from_command(spec, command)

    return run_and_record(
        code_name=args.code or "",
        command=command,
        records_dir=records_dir,
        session_id=session_id,
        record_id=args.record_id,
        input_files=input_files,
        output_files=output_files,
        stdout_path=args.stdout,
        log_trace=_log_trace_detached(session_id, records_dir.parent),
    )


if __name__ == "__main__":
    sys.exit(main())
