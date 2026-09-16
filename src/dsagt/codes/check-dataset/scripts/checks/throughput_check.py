#!/usr/bin/env python
"""Time __getitem__ latency over random indices and flag pathologically slow samples."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_object, worker_main  # noqa: E402


def run_check(args: dict) -> dict:
    dataset = build_object(args["dataset"], args.get("dataset_args") or {})
    n_requested = args.get("n_samples", 50)
    slow_threshold_s = args.get("slow_threshold_s", 5.0)

    length = len(dataset)
    if length == 0:
        return {"status": "error", "error": "dataset has length 0"}

    n = min(n_requested, length) if n_requested else length
    rng = random.Random(0)
    indices = [rng.randrange(length) for _ in range(n)]

    timings = []
    for i in indices:
        start = time.perf_counter()
        dataset[i]
        timings.append((i, time.perf_counter() - start))

    durations = sorted(t for _, t in timings)
    mean_s = sum(durations) / len(durations)
    median_s = durations[len(durations) // 2]
    max_s = durations[-1]
    slow = [{"index": i, "latency_s": t} for i, t in timings if t > slow_threshold_s]

    return {
        "status": "failed" if slow else "passed",
        "detail": {
            "n_samples": n,
            "mean_latency_s": mean_s,
            "median_latency_s": median_s,
            "max_latency_s": max_s,
            "items_per_sec": (1.0 / mean_s) if mean_s > 0 else None,
            "slow_threshold_s": slow_threshold_s,
            "slow_samples": slow,
        },
    }


if __name__ == "__main__":
    worker_main(run_check)
