#!/usr/bin/env python
"""Validate a split manifest against the contract's split policy.

Fails if any id (a row index for ``strategy: random``, a group id for
``group``/``time``) appears in more than one split, or if realized split
sizes deviate from ``split.ratios`` by more than a fixed tolerance. Does not
import the user's dataset — this is pure manifest validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import worker_main  # noqa: E402

from dsagt.contract import load_contract  # noqa: E402

_RATIO_TOLERANCE_PCT = 5.0


def run_check(args: dict) -> dict:
    contract = load_contract(args["contract"])

    manifest_path = args.get("split_manifest")
    if not manifest_path:
        return {"status": "skipped", "reason": "--split-manifest not provided"}

    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, dict) or not manifest:
        return {
            "status": "error",
            "error": "split manifest must be a non-empty mapping of split name to id list",
        }

    first_seen: dict[str, str] = {}
    leaked_splits: dict[str, set] = {}
    for split_name, ids in manifest.items():
        for identity in ids:
            key = str(identity)
            owner = first_seen.get(key)
            if owner is None:
                first_seen[key] = split_name
            elif owner != split_name:
                leaked_splits.setdefault(key, {owner}).add(split_name)

    total = sum(len(ids) for ids in manifest.values())
    ratios = contract["split"]["ratios"]
    size_problems = {}
    for split_name, ratio in ratios.items():
        actual = len(manifest.get(split_name, []))
        actual_pct = (actual / total * 100) if total else 0.0
        expected_pct = ratio * 100
        if abs(actual_pct - expected_pct) > _RATIO_TOLERANCE_PCT:
            size_problems[split_name] = {
                "expected_pct": expected_pct,
                "actual_pct": actual_pct,
            }

    passed = not leaked_splits and not size_problems
    return {
        "status": "passed" if passed else "failed",
        "detail": {
            "total_samples": total,
            "leaked_ids": {k: sorted(v) for k, v in leaked_splits.items()},
            "size_problems": size_problems,
        },
    }


if __name__ == "__main__":
    worker_main(run_check)
