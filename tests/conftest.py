"""Shared test setup for dsagt."""

import os
import sys
from pathlib import Path

import pytest

# MLflow writes spans to the store on a background thread by default; tests
# that read a trace right after emitting it would race that write.  Set before
# any test imports mlflow so the exporter is built synchronous.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

# Ensure tests/ is on sys.path so mcp_helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))


def _put_git_tools_on_path() -> None:
    """On Windows, make the POSIX tools the suite runs as stand-in codes
    (``echo``, ``true``, ``cp``, ``cat``, ``mv``, ``sleep``) findable.

    Git for Windows ships them in ``<Git>\\usr\\bin`` and puts only ``git``
    on PATH; Git Bash, which is where an agent runs a registered code, has
    that directory on its PATH, so adding it here gives the tests the PATH
    the commands have in use.  It goes at the end, so nothing already found
    changes.
    """
    from dsagt._compat import IS_WINDOWS, find_bash

    if not IS_WINDOWS:
        return
    bash = find_bash()
    if bash is None:
        return
    usr_bin = Path(bash).parent.parent / "usr" / "bin"
    if usr_bin.is_dir():
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(usr_bin)


_put_git_tools_on_path()

#: For a test of something only POSIX has: a signal delivered to a process
#: (Windows ends the process instead), or the kernel reading a ``#!`` line.
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX-only behavior")


@pytest.fixture(autouse=True)
def _no_shell_tracking_server(monkeypatch):
    """Keep the suite hermetic.  ``resolve_tracking_uri`` honors
    ``MLFLOW_TRACKING_URI``, so a developer with a shared server exported would
    otherwise have every in-project test log real spans there and fail the
    assertions that expect the sqlite default."""
    for var in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_API_KEY",
        "MLFLOW_TRACKING_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
