"""
dsagt traces <project> — open the MLflow trace viewer over the project's
serverless store, frictionlessly.

Three ergonomic wins over a raw
``mlflow ui --backend-store-uri sqlite:///<pdir>/mlflow.db``:

1. **Catch-up first.** Runs :func:`dsagt.session.catch_up_extraction`, which
   flushes the most-recent session's deferred final turn (the one an ungraceful
   agent exit leaves unlogged) into the store — so "I don't see my last query"
   is fixed before the viewer even opens.
2. **Deep link to the Traces tab.** DSAGT writes MLflow *traces*, not classic
   runs, so the default Experiments/Runs view looks empty.  We resolve the
   project's experiment id and print the URL that lands directly on its Traces
   tab.
3. **Quiet.** ``--workers 1`` and ``PYTHONWARNINGS=ignore`` drop the repeated
   Starlette deprecation spam; the viewer runs in the foreground (Ctrl-C to
   stop), so there is no background server to hunt down and kill later — the
   store stays serverless.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from dsagt.session import catch_up_extraction, load_config

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 5000


def _resolve_experiment_id(tracking_uri: str, project: str) -> str | None:
    """The MLflow experiment id for *project*, or None if not yet created."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(project)
        return exp.experiment_id if exp else None
    except Exception as e:  # noqa: BLE001 — a missing id only costs the deep link
        logger.debug("Could not resolve experiment id for %s: %s", project, e)
        return None


def run(project: str, port: int = _DEFAULT_PORT) -> int:
    config = load_config(project)
    pdir = Path(config["project_dir"])
    db = pdir / "mlflow.db"
    if not db.exists():
        print(
            f"No trace store yet for '{project}' ({db} not found). "
            "Run a session first: dsagt start "
            f"{project}"
        )
        return 1

    # 1. Catch-up: surface the most-recent session's deferred final turn before
    #    the viewer opens.  Best-effort — a viewer must open even if catch-up
    #    hiccups.
    try:
        result = catch_up_extraction(pdir, config)
        caught = result.get("traces_caught_up", 0)
        if caught:
            print(f"Caught up {caught} trailing trace(s) from the last session.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Trace catch-up before viewer failed: %s", e)

    # 2. Deep-link to the project's Traces tab (DSAGT emits traces, not runs, so
    #    the default Runs view looks empty).
    tracking_uri = f"sqlite:///{db}"
    exp_id = _resolve_experiment_id(tracking_uri, project)
    base = f"http://127.0.0.1:{port}"
    url = f"{base}/#/experiments/{exp_id}/traces" if exp_id else base

    print(f"\nMLflow trace view for '{project}':\n  {url}")
    print("(Ctrl-C to stop the viewer — the store itself is serverless.)\n")

    # 3. Foreground, quiet: one worker + warnings off drops the Starlette noise.
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        tracking_uri,
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    try:
        return subprocess.run(cmd, env=env).returncode
    except FileNotFoundError:
        print(
            "mlflow not found on PATH.  It ships with dsagt — activate the same "
            "environment dsagt runs in."
        )
        return 1
    except KeyboardInterrupt:
        return 0
