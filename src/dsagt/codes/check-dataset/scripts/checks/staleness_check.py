#!/usr/bin/env python
"""Recompute the pipeline fingerprint and compare it against the contract's.

Skipped for a ``standalone``-mode contract, which carries no
``pipeline_fingerprint`` to compare against (there is no upstream DSAgt
pipeline; the data root was characterized directly).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import worker_main  # noqa: E402

from dsagt.contract import load_contract  # noqa: E402
from dsagt.provenance import (
    compute_pipeline_fingerprint,
    reconstruct_pipeline,
)  # noqa: E402


def run_check(args: dict) -> dict:
    contract = load_contract(args["contract"])
    if contract["mode"] == "standalone":
        return {
            "status": "skipped",
            "reason": "contract mode is 'standalone'; no pipeline to fingerprint",
        }

    trace_dir = Path(args.get("project_dir") or ".") / "trace_archive"
    structured = json.loads(reconstruct_pipeline(trace_dir, fmt="json"))
    current_fingerprint = compute_pipeline_fingerprint(structured)
    contract_fingerprint = contract["pipeline_fingerprint"]

    stale = current_fingerprint != contract_fingerprint
    detail = {
        "contract_fingerprint": contract_fingerprint,
        "current_fingerprint": current_fingerprint,
        "current_steps": [r["code_name"] for r in structured.get("records", [])],
        "current_terminal_outputs": structured.get("terminal_outputs", []),
    }
    return {"status": "failed" if stale else "passed", "detail": detail}


if __name__ == "__main__":
    worker_main(run_check)
