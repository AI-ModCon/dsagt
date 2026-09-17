#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC

"""Check XGC simulation directory structure before preprocessing.

Verifies mesh file presence, counts 3d/f3d timestep files, detects
nphi and axis order, and flags any structural inconsistencies.

Output: JSON report to stdout (and optionally --output file).
Exit 0 if all checks pass; exit 1 if any check fails.
"""

import argparse
import glob
import json
import os
import re
import sys

try:
    import adios2
except ImportError:
    sys.exit('{"status": "error", "message": "adios2 not available"}')


def _parse_step(filename):
    m = re.search(r'\.(\d+)\.bp$', filename)
    return int(m.group(1)) if m else -1


def _bp_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / 1024 ** 2


def _get_nphi_and_axis(bp_path):
    try:
        with adios2.FileReader(bp_path) as f:
            avail = f.available_variables()
            nphi = int(f.read("nphi")) if "nphi" in avail else None
            for name in ("dpot", "eden", "iden"):
                if name in avail:
                    shape_str = avail[name]["Shape"]
                    dims = [int(x.strip()) for x in shape_str.split(",")]
                    if nphi and len(dims) == 2:
                        axis = "first" if dims[0] == nphi else "last"
                        return nphi, axis
            return nphi, None
    except Exception as e:
        return None, str(e)


def check(case_dir, output_path=None):
    case_dir = os.path.abspath(case_dir)
    checks = []
    warnings = []

    def ok(name, detail=""):
        checks.append({"check": name, "passed": True, "detail": detail})

    def fail(name, detail=""):
        checks.append({"check": name, "passed": False, "detail": detail})

    # 1. Directory exists
    if not os.path.isdir(case_dir):
        report = {"status": "error", "message": f"Not a directory: {case_dir}"}
        print(json.dumps(report, indent=2))
        return False
    ok("directory_exists", case_dir)

    # 2. Mesh file
    mesh_path = os.path.join(case_dir, "xgc.mesh.bp")
    if os.path.exists(mesh_path):
        ok("mesh_file_present", f"{_bp_size_mb(mesh_path):.1f} MB")
    else:
        fail("mesh_file_present", "xgc.mesh.bp not found")

    # 3. Count 3d files
    files_3d = sorted(glob.glob(os.path.join(case_dir, "xgc.3d.*.bp")))
    steps_3d = [_parse_step(f) for f in files_3d]
    if files_3d:
        stride = (steps_3d[1] - steps_3d[0]) if len(steps_3d) > 1 else None
        ok("3d_files_found", f"{len(files_3d)} files, steps {steps_3d[0]}–{steps_3d[-1]}, stride {stride}")
    else:
        fail("3d_files_found", "no xgc.3d.*.bp files found")

    # 4. Count f3d files (optional)
    files_f3d = sorted(glob.glob(os.path.join(case_dir, "xgc.f3d.*.bp")))
    steps_f3d = [_parse_step(f) for f in files_f3d]
    if files_f3d:
        ok("f3d_files_found", f"{len(files_f3d)} files, steps {steps_f3d[0]}–{steps_f3d[-1]}")
        # Check f3d steps are a subset of 3d steps
        orphans = [s for s in steps_f3d if s not in set(steps_3d)]
        if orphans:
            warnings.append(f"f3d steps not in 3d step list: {orphans[:5]}")
    else:
        ok("f3d_files_found", "none (3d-only dataset)")

    # 5. Probe first 3d file for nphi and axis order
    if files_3d:
        nphi, axis = _get_nphi_and_axis(files_3d[0])
        if nphi:
            detail = f"nphi={nphi}, axis_order={axis} ({'[nphi,N]' if axis == 'first' else '[N,nphi]' if axis == 'last' else 'unknown'})"
            ok("nphi_detected", detail)
        else:
            fail("nphi_detected", f"could not read nphi: {axis}")

    # 6. Mesh node count consistency
    if os.path.exists(mesh_path) and files_3d:
        try:
            with adios2.FileReader(mesh_path) as f:
                n_n = int(f.read("n_n"))
            with adios2.FileReader(files_3d[0]) as f:
                avail = f.available_variables()
                nnode = int(f.read("nnode")) if "nnode" in avail else None
            if nnode is not None and nnode != n_n:
                fail("node_count_consistent", f"mesh n_n={n_n} != 3d nnode={nnode}")
            else:
                ok("node_count_consistent", f"n_nodes={n_n}")
        except Exception as e:
            warnings.append(f"node count check error: {e}")

    passed = all(c["passed"] for c in checks)
    report = {
        "status": "ok" if passed else "fail",
        "case_dir": case_dir,
        "checks": checks,
        "warnings": warnings,
        "summary": {
            "n_3d_steps": len(files_3d),
            "n_f3d_steps": len(files_f3d),
            "nphi": nphi if files_3d else None,
            "axis_order": axis if files_3d else None,
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
    parser.add_argument("case_dir", help="XGC simulation directory to check")
    parser.add_argument("--output", default=None,
                        help="Write JSON report to this file (in addition to stdout)")
    args = parser.parse_args()

    passed = check(args.case_dir, args.output)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
