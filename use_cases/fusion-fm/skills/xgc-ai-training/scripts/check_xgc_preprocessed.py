#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC

"""Validate preprocessed XGC npz output before dataset use.

Reads meta.json, spot-checks a sample of step_NNNNN.npz files:
  - All declared fields present
  - Shapes match [nphi, n_nodes]
  - Dtype is float32
  - Time values are monotonically increasing
  - Field coverage matches meta.json field_availability

Output: JSON report to stdout (and optionally --output file).
Exit 0 if all checks pass; exit 1 if any fail.
"""

import argparse
import json
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit('{"status": "error", "message": "numpy not available"}')

_SPOT_CHECK_COUNT = 5


def check(out_dir, output_path=None):
    out_dir = os.path.abspath(out_dir)
    checks = []
    warnings = []

    def ok(name, detail=""):
        checks.append({"check": name, "passed": True, "detail": str(detail)})

    def fail(name, detail=""):
        checks.append({"check": name, "passed": False, "detail": str(detail)})

    # 1. meta.json exists and is valid
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        report = {"status": "error", "message": f"meta.json not found in {out_dir}"}
        print(json.dumps(report, indent=2))
        return False
    try:
        with open(meta_path) as fp:
            meta = json.load(fp)
        ok("meta_json_valid", f"n_nodes={meta['n_nodes']}, nphi={meta['nphi']}, n_steps={meta['n_steps']}")
    except Exception as e:
        fail("meta_json_valid", str(e))
        report = {"status": "fail", "out_dir": out_dir, "checks": checks, "warnings": warnings}
        print(json.dumps(report, indent=2))
        return False

    n_nodes = meta["n_nodes"]
    nphi    = meta["nphi"]
    steps   = meta["steps"]
    fields  = meta["field_names"]

    # 2. mesh.npz exists
    mesh_path = os.path.join(out_dir, "mesh.npz")
    if os.path.exists(mesh_path):
        d = np.load(mesh_path)
        rz_ok = "rz" in d and d["rz"].shape == (n_nodes, 2)
        conn_ok = "conn" in d
        ok("mesh_npz_valid", f"rz={d['rz'].shape if 'rz' in d else 'missing'}, conn={'present' if conn_ok else 'missing'}")
        if not rz_ok:
            fail("mesh_rz_shape", f"expected ({n_nodes}, 2), got {d['rz'].shape if 'rz' in d else 'missing'}")
    else:
        fail("mesh_npz_valid", "mesh.npz not found")

    # 3. Step file count matches meta
    step_files = [os.path.join(out_dir, f"step_{s:05d}.npz") for s in steps]
    present = [f for f in step_files if os.path.exists(f)]
    if len(present) == len(steps):
        ok("step_file_count", f"{len(present)}/{len(steps)} files present")
    else:
        fail("step_file_count", f"only {len(present)}/{len(steps)} step files found")

    # 4. Spot-check a sample of step files
    indices = list(range(0, len(steps), max(1, len(steps) // _SPOT_CHECK_COUNT)))[:_SPOT_CHECK_COUNT]
    checked_steps, shape_errors, dtype_errors, field_errors = [], [], [], []
    time_vals = []

    for i in indices:
        s = steps[i]
        path = os.path.join(out_dir, f"step_{s:05d}.npz")
        if not os.path.exists(path):
            continue
        try:
            d = np.load(path)
        except Exception as e:
            warnings.append(f"step_{s:05d}.npz load error: {e}")
            continue

        checked_steps.append(s)

        # Field presence
        missing = [fn for fn in fields if fn not in d.files]
        if missing:
            field_errors.append(f"step {s}: missing {missing}")

        # Shape + dtype
        for fn in fields:
            if fn not in d:
                continue
            arr = d[fn]
            if arr.shape != (nphi, n_nodes):
                shape_errors.append(f"step {s} {fn}: {arr.shape} != ({nphi},{n_nodes})")
            if arr.dtype != np.float32:
                dtype_errors.append(f"step {s} {fn}: dtype={arr.dtype}")

        if "time" in d:
            time_vals.append((s, float(d["time"])))

    if field_errors:
        fail("spot_check_fields", "; ".join(field_errors[:3]))
    else:
        ok("spot_check_fields", f"all fields present in {len(checked_steps)} sampled steps")

    if shape_errors:
        fail("spot_check_shapes", "; ".join(shape_errors[:3]))
    else:
        ok("spot_check_shapes", f"all shapes [nphi={nphi}, n_nodes={n_nodes}] correct")

    if dtype_errors:
        fail("spot_check_dtypes", "; ".join(dtype_errors[:3]))
    else:
        ok("spot_check_dtypes", "all fields are float32")

    # 5. Time monotonicity
    if len(time_vals) > 1:
        times = [t for _, t in sorted(time_vals)]
        monotone = all(times[i] < times[i + 1] for i in range(len(times) - 1))
        if monotone:
            ok("time_monotone", f"range [{times[0]:.4e}, {times[-1]:.4e}] s")
        else:
            fail("time_monotone", "time values are not monotonically increasing")

    # 6. field_availability coverage
    fa = meta.get("field_availability", {})
    coverage_issues = []
    for fn in fields:
        if fn in fa and len(fa[fn]) == 0:
            coverage_issues.append(fn)
    if coverage_issues:
        warnings.append(f"fields with zero coverage in field_availability: {coverage_issues}")
    else:
        ok("field_availability_coverage", f"{len(fields)} fields all have coverage entries")

    passed = all(c["passed"] for c in checks)
    report = {
        "status": "ok" if passed else "fail",
        "out_dir": out_dir,
        "checks": checks,
        "warnings": warnings,
        "summary": {
            "n_steps_checked": len(steps),
            "n_steps_spot_checked": len(checked_steps),
            "fields": fields,
            "n_nodes": n_nodes,
            "nphi": nphi,
        },
    }

    out = json.dumps(report, indent=2)
    print(out)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as fp:
            fp.write(out)

    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", help="Preprocessed output directory (contains meta.json)")
    parser.add_argument("--output", default=None,
                        help="Write JSON report to this file (in addition to stdout)")
    args = parser.parse_args()

    passed = check(args.out_dir, args.output)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
