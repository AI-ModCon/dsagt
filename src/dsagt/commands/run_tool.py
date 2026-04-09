"""
dsagt-run: Tool execution wrapper for provenance capture.

Usage:
    dsagt-run --tool fastp -- fastp -q 20 -l 50 --in1 reads.fq.gz
"""

import argparse
import sys

from dsagt.provenance import _resolve_records_dir, _parse_file_list, run_and_record


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsagt-run",
        description="Wrap a tool command and capture execution provenance.",
    )
    parser.add_argument("--tool", required=True, help="Name of the tool being executed.")
    parser.add_argument("--session", default=None, help="Session ID. Defaults to $DSAGT_SESSION_ID.")
    parser.add_argument("--record-id", default=None, help="Pre-assigned record ID.")
    parser.add_argument("--records-dir", default=None, help="Directory for execution records.")
    parser.add_argument("--input-files", default=None, help="Comma-separated input file paths.")
    parser.add_argument("--output-files", default=None, help="Comma-separated output file paths.")
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
    command_args = args_to_parse[sep + 1:]

    parsed = _make_parser().parse_args(wrapper_args)
    return parsed, command_args


def main(argv: list[str] | None = None) -> int:
    args, command = _parse_args(argv)

    if not command:
        print("dsagt-run: no command specified after '--'", file=sys.stderr)
        return 1

    from dsagt.observability import init_tracing
    init_tracing("dsagt-run", session_id=args.session)

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
