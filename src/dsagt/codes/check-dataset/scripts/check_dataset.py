#!/usr/bin/env python
"""Validate a generated PyTorch Dataset against its sample contract.

Runs each check in ``checks/`` (contract, determinism, worker_equivalence,
split_leakage, throughput, model_forward) as its own subprocess:
importing the user's ``Dataset`` into this process would contaminate the
fork-sensitive state ``worker_equivalence`` specifically measures (open file
handles, a seeded global RNG, an initialized CUDA context), and a segfault in
a native reader (HDF5, ADIOS2) must not take down the whole run. Each
subprocess is bounded by ``--check-timeout-s``, so a wedged reader or a
check whose own worker processes fail to tear down cannot hang the whole
run either. Emits one JSON report to ``--output`` (default
``audit/check_dataset_<ts>.json``) and to stdout. Exits 0 iff every
requested, non-skipped check passed.

The code runs in the environment the user's ``Dataset`` imports in and
imports nothing from the ``dsagt`` package; whether the upstream pipeline
changed since the contract was written is the ``check_contract_staleness``
MCP tool's question, answered from the execution records dsagt holds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ALL_CHECKS = [
    "contract",
    "determinism",
    "worker_equivalence",
    "split_leakage",
    "throughput",
    "model_forward",
]

_CHECKS_DIR = Path(__file__).parent / "checks"
sys.path.insert(0, str(_CHECKS_DIR))
from _common import load_contract  # noqa: E402


def _parse_checks(raw: str | None) -> list[str]:
    if not raw or raw.strip().lower() == "all":
        return list(ALL_CHECKS)
    requested = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in requested if c not in ALL_CHECKS]
    if unknown:
        raise ValueError(f"Unknown check(s): {unknown}. Valid: {ALL_CHECKS}")
    return requested


def _run_one(check_name: str, worker_args: dict, timeout_s: float) -> dict:
    script = _CHECKS_DIR / f"{check_name}_check.py"
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script), json.dumps(worker_args)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        # The check's worker is killed by subprocess.run itself on timeout; a
        # native reader wedged in a blocking read, or a DataLoader whose
        # workers fail to tear down, must not hang the whole run.
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        return {
            "status": "error",
            "error": f"check timed out after {timeout_s}s",
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
            "duration_ms": duration_ms,
        }
    duration_ms = round((time.perf_counter() - start) * 1000, 3)

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        result = {
            "status": "error",
            "error": "worker produced no parseable JSON",
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
        }
    result["duration_ms"] = duration_ms
    return result


def _resolve_id_key(explicit: str | None, contract_path: str) -> str | None:
    if explicit:
        return explicit
    contract = load_contract(contract_path)
    for name, spec in contract["keys"].items():
        if spec.get("role") == "metadata":
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-args", "--dataset_args", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checks", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", "--sample_index", type=int, default=0)
    parser.add_argument("--id-key", "--id_key", default=None)
    parser.add_argument("--split-manifest", "--split_manifest", default=None)
    parser.add_argument(
        "--n-throughput-samples", "--n_throughput_samples", type=int, default=50
    )
    parser.add_argument(
        "--slow-threshold-s", "--slow_threshold_s", type=float, default=5.0
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-args", "--model_args", default=None)
    parser.add_argument("--collate-fn", "--collate_fn", default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=2)
    parser.add_argument(
        "--check-timeout-s", "--check_timeout_s", type=float, default=300.0
    )
    args = parser.parse_args()

    try:
        selected = _parse_checks(args.checks)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(2)

    worker_args = {
        "contract": args.contract,
        "dataset": args.dataset,
        "dataset_args": json.loads(args.dataset_args) if args.dataset_args else {},
        "sample_index": args.sample_index,
        "seed": args.seed,
        "id_key": _resolve_id_key(args.id_key, args.contract),
        "split_manifest": args.split_manifest,
        "n_samples": args.n_throughput_samples,
        "slow_threshold_s": args.slow_threshold_s,
        "model": args.model,
        "model_args": json.loads(args.model_args) if args.model_args else {},
        "collate_fn": args.collate_fn,
        "batch_size": args.batch_size,
    }

    results = {
        check_name: _run_one(check_name, worker_args, args.check_timeout_s)
        for check_name in selected
    }
    overall_passed = all(
        r.get("status") in ("passed", "skipped") for r in results.values()
    )

    report = {
        "dataset": args.dataset,
        "contract": args.contract,
        "checks_run": selected,
        "overall_passed": overall_passed,
        "checks": results,
    }

    output_path = args.output
    if not output_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = f"audit/check_dataset_{ts}.json"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
