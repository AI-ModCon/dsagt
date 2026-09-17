#!/usr/bin/env python
"""Verify one dataset sample against its declared sample contract.

Constructs the dataset, reads ``ds[sample_index]``, and checks every
declared key's presence, dtype, shape (including symbolic dims, which bind
to their first observed value and must repeat consistently across keys),
and ``value_range``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_object, worker_main  # noqa: E402

from _common import load_contract  # noqa: E402


def _normalize_dtype(s: str) -> str:
    return s.rsplit(".", 1)[-1]


# Aliases for a value with no `.dtype` attribute (a plain Python object, per
# the tabular contract example's `patient_id: {dtype: string}`) — compared by
# Python type rather than by the tensor/array dtype string.
_PYTHON_DTYPE_ALIASES = {
    "string": str,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _dtype_matches(value, dtype_expected: str) -> bool:
    actual_dtype = getattr(value, "dtype", None)
    if actual_dtype is not None:
        return _normalize_dtype(str(actual_dtype)) == _normalize_dtype(dtype_expected)
    py_type = _PYTHON_DTYPE_ALIASES.get(dtype_expected.lower())
    if py_type is not None:
        return isinstance(value, py_type)
    return type(value).__name__ == dtype_expected


def _shape_list(value):
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _check_key(name: str, spec: dict, value, symbolic_dims: dict) -> list[str]:
    problems = []

    dtype_expected = spec["dtype"]
    if not _dtype_matches(value, dtype_expected):
        actual_dtype = getattr(value, "dtype", None)
        dtype_actual = (
            str(actual_dtype) if actual_dtype is not None else type(value).__name__
        )
        problems.append(
            f"dtype mismatch: expected {dtype_expected!r}, got {dtype_actual!r}"
        )

    shape_expected = spec.get("shape", [])
    shape_actual = _shape_list(value)
    if shape_actual is None:
        if shape_expected:
            problems.append("value has no shape but contract declares one")
    elif len(shape_actual) != len(shape_expected):
        problems.append(
            f"shape rank mismatch: expected {shape_expected}, got {shape_actual}"
        )
    else:
        for dim_expected, dim_actual in zip(shape_expected, shape_actual):
            if isinstance(dim_expected, int):
                if dim_expected != dim_actual:
                    problems.append(
                        f"shape mismatch: expected {shape_expected}, got {shape_actual}"
                    )
                    break
            else:
                bound = symbolic_dims.get(dim_expected)
                if bound is None:
                    symbolic_dims[dim_expected] = dim_actual
                elif bound != dim_actual:
                    problems.append(
                        f"symbolic dim {dim_expected!r} bound to {bound}, "
                        f"but key {name!r} has {dim_actual}"
                    )

    value_range = spec.get("value_range")
    if value_range is not None:
        try:
            actual_min = float(value.min())
            actual_max = float(value.max())
        except (AttributeError, TypeError, ValueError):
            actual_min = actual_max = None
        if actual_min is not None:
            lo, hi = value_range
            if actual_min < lo or actual_max > hi:
                problems.append(
                    f"value_range violated: declared [{lo}, {hi}], "
                    f"observed [{actual_min}, {actual_max}]"
                )

    return problems


def run_check(args: dict) -> dict:
    contract = load_contract(args["contract"])
    dataset = build_object(args["dataset"], args.get("dataset_args") or {})
    index = args.get("sample_index", 0)

    sample = dataset[index]
    if not isinstance(sample, dict):
        return {
            "status": "failed",
            "detail": {
                "error": f"sample must be a dict of key -> value, got {type(sample).__name__}"
            },
        }

    problems = {}
    symbolic_dims: dict = {}
    for name, spec in contract["keys"].items():
        if name not in sample:
            problems[name] = ["key missing from sample"]
            continue
        key_problems = _check_key(name, spec, sample[name], symbolic_dims)
        if key_problems:
            problems[name] = key_problems

    return {
        "status": "failed" if problems else "passed",
        "detail": {
            "sample_index": index,
            "keys_checked": list(contract["keys"]),
            "problems": problems,
            "symbolic_dims": symbolic_dims,
        },
    }


if __name__ == "__main__":
    worker_main(run_check)
