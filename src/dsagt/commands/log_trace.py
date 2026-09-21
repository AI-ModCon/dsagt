"""The second half of a ``dsagt-run``: one execution record's ``code.execute`` trace.

``dsagt-run`` writes the record and starts this module detached, so the run
itself loads no trace store and the agent's shell gets its prompt back about a
second sooner.  The span is backdated to the run's own start and end, and the
session it belongs to is the one the record carries.

    python -m dsagt.commands.log_trace <record path>

Run from the project directory, whose ``.dsagt/`` holds the lock this takes and
the log an error goes to.
"""

import fcntl
import json
import sys
from pathlib import Path


def log_trace(record_path: str) -> None:
    """Log the ``code.execute`` trace of the record at *record_path*.

    Loads the user's service credentials and the trace store, which the run
    leaves unloaded.  One writer at a time: several runs in a second would
    otherwise race to create a store that does not exist yet, and all but one
    would fail.
    """
    from dsagt.observability import init_tracing, log_execution_trace
    from dsagt.session import load_user_env

    # The agent runs dsagt-run from its own shell, not as a child of
    # dsagt-server, so under codex and cline the credentials file is the only
    # way a shared-store key or URI reaches the trace.
    load_user_env()
    record = json.loads(Path(record_path).read_text())
    with open(Path(".dsagt") / "run_trace.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        init_tracing("dsagt-run", session_id=record.get("session_id"))
        log_execution_trace(record)
        import mlflow

        mlflow.flush_trace_async_logging()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(
            "usage: python -m dsagt.commands.log_trace <record path>", file=sys.stderr
        )
        return 2
    log_trace(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
