#!/usr/bin/env python
"""Verify that re-seeding and re-instantiating the dataset reproduces the same sample.

A dataset whose ``__getitem__`` draws from an unseeded or process-global
source of randomness (a module-level RNG never reset, ``time.time()``-derived
jitter) will fail this: the same seed and the same index should always
produce the same sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_object, values_equal, worker_main  # noqa: E402


def _seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_check(args: dict) -> dict:
    seed = args.get("seed", 0)
    index = args.get("sample_index", 0)
    dataset_args = args.get("dataset_args") or {}

    _seed_all(seed)
    sample_a = build_object(args["dataset"], dataset_args)[index]

    _seed_all(seed)
    sample_b = build_object(args["dataset"], dataset_args)[index]

    equal = values_equal(sample_a, sample_b)
    return {
        "status": "passed" if equal else "failed",
        "detail": {"seed": seed, "sample_index": index, "samples_equal": equal},
    }


if __name__ == "__main__":
    worker_main(run_check)
