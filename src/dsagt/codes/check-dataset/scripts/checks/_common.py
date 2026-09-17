"""Shared helpers for check-dataset's per-check worker scripts.

Each worker script (``contract_check.py``, ``determinism_check.py``, etc.) is
invoked as its own subprocess by ``check_dataset.py`` so that importing the
user's ``Dataset`` never contaminates the orchestrator process, and a
segfault in a native reader takes down only one check. This module has no
dependency on the ``dsagt`` package itself (contract/staleness checks import
``dsagt.contract`` / ``dsagt.provenance`` directly, which is fine since they
run in the same environment ``dsagt-run`` was launched from) — it only
provides the import-by-path and JSON worker-loop plumbing every check shares.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback


def import_by_path(path: str):
    """Import ``module.sub:attr`` (or ``module.sub.attr``) and return ``attr``."""
    if ":" in path:
        module_name, _, attr_path = path.partition(":")
    else:
        module_name, _, attr_path = path.rpartition(".")
        if not module_name:
            raise ValueError(
                f"Cannot resolve import path {path!r}; expected 'module:attr' "
                "or 'module.attr'"
            )
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def build_object(path: str, kwargs: dict):
    """Import ``path`` and call it with ``**kwargs`` (a class or a factory)."""
    factory = import_by_path(path)
    return factory(**(kwargs or {}))


def _is_tensor_like(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def values_equal(a, b) -> bool:
    """Recursively compare possibly-nested dict/list/tuple of tensors/arrays/scalars."""
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(values_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(values_equal(x, y) for x, y in zip(a, b))
        )
    if _is_tensor_like(a) or _is_tensor_like(b):
        if not (_is_tensor_like(a) and _is_tensor_like(b)):
            return False
        if hasattr(a, "equal"):
            return bool(a.equal(b))
        import numpy as np

        return bool(np.array_equal(a, b))
    return a == b


def json_default(obj):
    """``json.dumps(default=...)`` hook for tensor/array/dtype-ish objects."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def worker_main(run_check) -> None:
    """Standard entrypoint: read JSON args from ``argv[1]``, run, print JSON, exit.

    Exit code 0 for ``passed``/``skipped``, 1 for ``failed``, 2 for a worker
    crash (``error``) — distinct from a check-observed failure so the
    orchestrator's subprocess return code alone can't be mistaken for one.
    """
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "error": f"invalid args JSON: {e}"}))
        sys.stdout.flush()
        sys.exit(2)

    try:
        result = run_check(args)
    except Exception as e:  # noqa: BLE001 — reported in the JSON result, not raised
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.stdout.flush()
        sys.exit(2)

    print(json.dumps(result, default=json_default, indent=2))
    sys.stdout.flush()
    sys.exit(0 if result.get("status") in ("passed", "skipped") else 1)
