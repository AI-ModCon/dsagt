#!/usr/bin/env python
"""Push one batch through model.forward() and report output shape/dtype or the failure.

Against the user's real model this verifies the consumer contract; against a
reference model generated from the same contract it only verifies that the
contract is self-consistent and tensors flow, since the reference model was
derived from that same contract. It still catches dtype errors, ragged
collation failures, and device mismatches.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_object, import_by_path, worker_main  # noqa: E402

from _common import load_contract  # noqa: E402


def _describe(value) -> dict:
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", None))}
    return {"type": type(value).__name__}


def run_check(args: dict) -> dict:
    model_path = args.get("model")
    if not model_path:
        return {"status": "skipped", "reason": "--model not provided"}

    import torch
    from torch.utils.data import DataLoader

    contract = load_contract(args["contract"])
    dataset = build_object(args["dataset"], args.get("dataset_args") or {})
    model = build_object(model_path, args.get("model_args") or {})

    collate_path = args.get("collate_fn")
    collate_fn = import_by_path(collate_path) if collate_path else None

    batch_size = args.get("batch_size", 2)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    batch = next(iter(loader))

    input_keys = {
        name for name, spec in contract["keys"].items() if spec.get("role") == "input"
    }

    model.eval()
    try:
        with torch.no_grad():
            if isinstance(batch, dict):
                inputs = {k: v for k, v in batch.items() if k in input_keys} or batch
                output = model(**inputs)
            else:
                output = model(batch)
    except Exception as e:  # noqa: BLE001 — the failure itself is the check result
        return {"status": "failed", "detail": {"error": f"{type(e).__name__}: {e}"}}

    if isinstance(output, dict):
        output_desc = {k: _describe(v) for k, v in output.items()}
    elif isinstance(output, (list, tuple)):
        output_desc = [_describe(v) for v in output]
    else:
        output_desc = _describe(output)

    return {
        "status": "passed",
        "detail": {"batch_size": batch_size, "output": output_desc},
    }


if __name__ == "__main__":
    worker_main(run_check)
