"""Shared test setup for dsagt."""

import os
import sys
from pathlib import Path

# MLflow writes spans to the store on a background thread by default; tests
# that read a trace right after emitting it would race that write.  Set before
# any test imports mlflow so the exporter is built synchronous.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

# Ensure tests/ is on sys.path so mcp_helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))
