#!/usr/bin/env python
"""Compare the multiset of samples yielded at num_workers=0 vs num_workers=2.

Only meaningful for an ``IterableDataset``: a map-style ``Dataset`` is
sharded by index regardless of worker count, so it carries no risk of the
duplication bug this check targets (an ``IterableDataset.__iter__`` that
ignores ``torch.utils.data.get_worker_info()`` yields every sample once per
worker instead of its shard).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_object, worker_main  # noqa: E402


def _collect_ids(dataset, id_key: str, num_workers: int) -> list:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    ids = []
    for sample in loader:
        value = sample[id_key]
        if hasattr(value, "item"):
            value = value.item()
        ids.append(value)
    return ids


def run_check(args: dict) -> dict:
    from torch.utils.data import IterableDataset

    dataset_args = args.get("dataset_args") or {}

    probe = build_object(args["dataset"], dataset_args)
    if not isinstance(probe, IterableDataset):
        return {
            "status": "skipped",
            "reason": "map-style dataset shards by index, not by worker iteration",
        }

    id_key = args.get("id_key")
    if not id_key:
        return {
            "status": "error",
            "error": "id_key is required for worker_equivalence on an IterableDataset",
        }

    ids_serial = _collect_ids(probe, id_key, 0)
    ids_parallel = _collect_ids(build_object(args["dataset"], dataset_args), id_key, 2)

    counts_serial = Counter(ids_serial)
    counts_parallel = Counter(ids_parallel)
    equal = counts_serial == counts_parallel

    detail = {
        "id_key": id_key,
        "n_serial": len(ids_serial),
        "n_parallel": len(ids_parallel),
    }
    if not equal:
        detail["missing_in_parallel"] = list((counts_serial - counts_parallel).keys())
        detail["duplicated_in_parallel"] = list(
            (counts_parallel - counts_serial).keys()
        )

    return {"status": "passed" if equal else "failed", "detail": detail}


if __name__ == "__main__":
    worker_main(run_check)
