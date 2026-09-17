"""Dataset/model fixtures for ``tests/test_check_dataset.py``.

Kept as a standalone, importable-by-path module (``tests.fixtures_check_dataset:Name``)
rather than inline classes in the test file, since ``check_dataset``'s checks
import the dataset under test via ``dsagt.codes.check-dataset``'s
``import_by_path``, exactly as they would a real project's generated
``Dataset``. Includes both well-behaved fixtures and deliberately-buggy ones
matching the issue's acceptance criteria (iterable-worker duplication,
non-determinism).
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class GoodDataset(Dataset):
    """Deterministic map-style dataset matching MINIMAL_CONTRACT."""

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {
            "features": torch.full((4,), float(index % 5) / 10.0, dtype=torch.float32),
            "label": torch.tensor(index % 3, dtype=torch.int64),
            "id": f"sample-{index}",
        }


class NonDeterministicDataset(Dataset):
    """Ignores any seeding: mixes a nanosecond-scale timer into the sample.

    Uses ``perf_counter_ns() % 1_000_000`` rather than raw ``perf_counter()``
    seconds: the latter is a large enough float that float32 rounding can
    make two back-to-back calls collide, masking the non-determinism this
    fixture exists to exercise.
    """

    def __len__(self):
        return 20

    def __getitem__(self, index):
        jitter = float(time.perf_counter_ns() % 1_000_000)
        return {
            "features": torch.full((4,), jitter, dtype=torch.float32),
            "label": torch.tensor(index % 3, dtype=torch.int64),
            "id": f"sample-{index}",
        }


class BadContractDataset(Dataset):
    """Wrong dtype, wrong shape, and an out-of-range value versus MINIMAL_CONTRACT."""

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {
            "features": torch.full((4,), 5.0, dtype=torch.float64),
            "label": torch.tensor([index % 3], dtype=torch.int64),
            "id": f"sample-{index}",
        }


class SlowDataset(Dataset):
    """Sleeps on every access, to exercise the throughput check's failure path."""

    def __len__(self):
        return 5

    def __getitem__(self, index):
        time.sleep(0.01)
        return {
            "features": torch.zeros(4, dtype=torch.float32),
            "label": torch.tensor(0, dtype=torch.int64),
            "id": f"sample-{index}",
        }


class GoodIterableDataset(IterableDataset):
    """Shards its range across workers via ``get_worker_info()``."""

    def __init__(self, n: int = 20):
        self.n = n

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            indices = range(self.n)
        else:
            indices = range(worker_info.id, self.n, worker_info.num_workers)
        for index in indices:
            yield {
                "features": torch.full((4,), float(index), dtype=torch.float32),
                "label": torch.tensor(index % 3, dtype=torch.int64),
                "id": index,
            }


class BuggyIterableDataset(IterableDataset):
    """Ignores ``get_worker_info()``: every worker yields the whole range."""

    def __init__(self, n: int = 20):
        self.n = n

    def __iter__(self):
        for index in range(self.n):
            yield {
                "features": torch.full((4,), float(index), dtype=torch.float32),
                "label": torch.tensor(index % 3, dtype=torch.int64),
                "id": index,
            }


class GoodModel(torch.nn.Module):
    """Accepts the contract's single input key by name."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, features):
        return self.linear(features)


class BadModel(torch.nn.Module):
    """Declares an input width incompatible with the contract's features shape."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(99, 2)

    def forward(self, features):
        return self.linear(features)
