"""
dsagt-run: registered-code execution wrapper for provenance capture.

Usage:
    dsagt-run --code fastp -- fastp -q 20 -l 50 --in1 reads.fq.gz

``--code`` names the registered code, and the command after ``--`` is run
verbatim.  A code's stored ``executable`` is that whole line, so the agent
copies it from the registry rather than composing it.  Everything else the
record needs is derived: the project from the working directory or
``DSAGT_PROJECT_DIR``, the session from ``.dsagt/state.yaml``, the record id
from the run, and the files from the spec's parameter roles and the arguments.
"""

import argparse
import sys
from pathlib import Path

from dsagt.provenance import (
    _current_session_tag_from_cwd,
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
        "--code", required=True, help="Name of the registered code being executed."
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


def _log_trace_detached(project_dir: Path):
    """A ``log_trace`` for :func:`run_and_record` that starts a detached
    process for the trace and returns at once.

    Loading the store costs about a second; the command the agent asked for
    has already finished and its record is written, so the agent's shell gets
    its prompt back while the trace is logged.  The process has its own
    session, so a harness that ends the turn's process group leaves it alone.
    ``None`` when the records directory is outside a project, which has no
    store for the trace to go to.
    """
    import subprocess

    if not (project_dir / ".dsagt" / "config.yaml").exists():
        return None

    def start(record_path) -> None:
        argv = [
            sys.executable,
            "-m",
            "dsagt.commands.log_trace",
            str(Path(record_path).resolve()),
        ]
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
    args, command = _parse_args(argv)

    if not command:
        print("dsagt-run: no command specified after '--'", file=sys.stderr)
        return 1

    try:
        records_dir = _resolve_records_dir()
    except ValueError as err:
        print(f"dsagt-run: {err}", file=sys.stderr)
        return 1

    # The spec's parameter roles name the files the command reads and writes;
    # each side the roles leave empty is filled from the arguments in
    # run_and_record.
    from dsagt.registry import CodeRegistry

    input_files: list[str] = []
    output_files: list[str] = []
    spec = CodeRegistry(runtime_dir=records_dir.parent).get_code(args.code)
    if spec is not None:
        input_files, output_files = file_roles_from_command(spec, command)

    return run_and_record(
        code_name=args.code,
        command=command,
        records_dir=records_dir,
        # The MCP server mints the session into .dsagt/state.yaml; the record
        # and the trace both carry it.
        session_id=_current_session_tag_from_cwd(),
        input_files=input_files,
        output_files=output_files,
        log_trace=_log_trace_detached(records_dir.parent),
    )


if __name__ == "__main__":
    sys.exit(main())
