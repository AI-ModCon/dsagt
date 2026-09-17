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


@pytest.fixture(autouse=True)
def _no_shell_tracking_server(monkeypatch):
    """Keep the suite hermetic.  ``resolve_tracking_uri`` honors
    ``MLFLOW_TRACKING_URI``, so a developer with a shared server exported would
    otherwise have every in-project test log real spans there — and fail the
    assertions that expect the sqlite default."""
    for var in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_API_KEY",
        "MLFLOW_TRACKING_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
