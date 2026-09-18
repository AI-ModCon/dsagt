"""
dsagt-run: registered-code execution wrapper for provenance capture.

Usage:
    dsagt-run --code fastp -- fastp -q 20 -l 50 --in1 reads.fq.gz
"""

import argparse
import sys

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
        "--code", required=True, help="Name of the code being executed."
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


def main(argv: list[str] | None = None) -> int:
    # The agent runs this from its own shell (the instructions hand it the
    # `dsagt-run --code …` prefix), not as a child of dsagt-server — so under
    # codex/cline the credentials file is the only way a shared-store key or
    # URI reaches the code.execute trace.
    from dsagt.session import load_user_env

    load_user_env()
    args, command = _parse_args(argv)

    if not command:
        print("dsagt-run: no command specified after '--'", file=sys.stderr)
        return 1

    from dsagt.observability import init_tracing

    # The session is resolved before tracing starts so the `code.execute`
    # trace root carries it — resolving it later inside run_and_record only
    # stamps the on-disk record, and the trace lands unbucketed.
    session_id = args.session or _current_session_tag_from_cwd()
    init_tracing("dsagt-run", session_id=session_id)

    try:
        records_dir = _resolve_records_dir(args.records_dir)
    except ValueError as err:
        print(f"dsagt-run: {err}", file=sys.stderr)
        return 1

    input_files = _parse_file_list(args.input_files)
    output_files = _parse_file_list(args.output_files)
    if not input_files and not output_files:
        # The spec's parameter roles name the files; the flags are the
        # override for a command the roles cannot describe.
        from dsagt.registry import CodeRegistry

        spec = CodeRegistry(runtime_dir=records_dir.parent).get_code(args.code)
        if spec is not None:
            input_files, output_files = file_roles_from_command(spec, command)

    return run_and_record(
        code_name=args.code,
        command=command,
        records_dir=records_dir,
        session_id=session_id,
        record_id=args.record_id,
        input_files=input_files,
        output_files=output_files,
    )


if __name__ == "__main__":
    sys.exit(main())
