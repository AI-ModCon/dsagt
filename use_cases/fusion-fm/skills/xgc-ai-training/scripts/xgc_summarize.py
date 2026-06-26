#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC

"""Summarize an XGC simulation directory and emit a structured JSON report.

Reads xgc.mesh.bp, xgc.3d.*.bp, xgc.f3d.*.bp, and units.m to produce
a machine-readable summary suitable for audit trails and data cards.

Output: JSON to stdout (and optionally --output file).
"""

import argparse
import glob
import json
import os
import re
import sys

try:
    import adios2
    import numpy as np
except ImportError as e:
    sys.exit(f'{{"status": "error", "message": "{e}"}}')


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


def _read_units(path):
    params = {}
    for fname in ("units.m", "units.txt"):
        fp = os.path.join(path, fname)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                line = line.strip().rstrip(";")
                if "=" in line and not line.startswith(("!", "#")):
                    k, _, v = line.partition("=")
                    try:
                        params[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        break
    return params


def _summarize_mesh(path):
    mesh_path = os.path.join(path, "xgc.mesh.bp")
    if not os.path.exists(mesh_path):
        return None
    info = {"size_mb": _bp_size_mb(mesh_path)}
    with adios2.FileReader(mesh_path) as f:
        avail = f.available_variables()
        info["n_nodes"] = int(f.read("n_n"))
        info["n_cells"] = int(f.read("n_t"))
        if "rz" in avail:
            rz = f.read("rz")
            info["R_range_m"] = [float(rz[:, 0].min()), float(rz[:, 0].max())]
            info["Z_range_m"] = [float(rz[:, 1].min()), float(rz[:, 1].max())]
        if "psi" in avail:
            psi = f.read("psi")
            info["psi_range"] = [float(psi.min()), float(psi.max())]
        if "region" in avail:
            region = f.read("region")
            info["region_codes"] = sorted(int(x) for x in np.unique(region).tolist())
        if "nd_connect_list" in avail:
            conn = f.read("nd_connect_list")
            pairs = np.concatenate([conn[:, [0, 1]], conn[:, [1, 2]], conn[:, [0, 2]]], axis=0)
            pairs = np.sort(pairs, axis=1)
            info["n_edges"] = int(len(np.unique(pairs, axis=0)))
        info["has_psi"]     = "psi" in avail
        info["has_region"]  = "region" in avail
        info["has_nextnode"] = "nextnode" in avail
        info["n_flux_surfaces"] = int(f.read("nsurf")) if "nsurf" in avail else None
    return info


def _summarize_steps(path, pattern):
    files = sorted(glob.glob(os.path.join(path, pattern)))
    if not files:
        return None
    steps = [_parse_step(f) for f in files]

    # Read nphi and variable list from first file
    nphi, axis_order, array_vars = None, None, {}
    time_vals = []
    with adios2.FileReader(files[0]) as f:
        avail = f.available_variables()
        if "nphi" in avail:
            nphi = int(f.read("nphi"))
        if "time" in avail:
            time_vals.append(float(f.read("time")))
        for k, v in avail.items():
            if v["SingleValue"] == "false" and "," in v.get("Shape", ""):
                dims = [int(x.strip()) for x in v["Shape"].split(",")]
                if nphi and len(dims) == 2 and nphi in dims:
                    array_vars[k] = v["Shape"]
                    if axis_order is None:
                        axis_order = "first" if dims[0] == nphi else "last"

    # Collect timestamps from remaining files
    for fp in files[1:]:
        try:
            with adios2.FileReader(fp) as f:
                avail = f.available_variables()
                if "time" in avail:
                    time_vals.append(float(f.read("time")))
        except Exception:
            pass

    stride = (steps[1] - steps[0]) if len(steps) > 1 else None
    dt_phys = ((max(time_vals) - min(time_vals)) / (len(time_vals) - 1)
               if len(time_vals) > 1 else None)

    return {
        "n_files": len(files),
        "step_range": [steps[0], steps[-1]],
        "step_stride": stride,
        "steps": steps,
        "nphi": nphi,
        "axis_order": axis_order,
        "time_range_s": [min(time_vals), max(time_vals)] if time_vals else None,
        "dt_phys_s": dt_phys,
        "total_size_mb": sum(_bp_size_mb(f) for f in files),
        "phi_field_names": sorted(array_vars.keys()),
    }


def summarize(case_dir):
    case_dir = os.path.abspath(case_dir)
    units = _read_units(case_dir)

    keys_of_interest = [
        "sml_dt", "sml_tran", "vth", "eq_axis_r", "eq_axis_z",
        "eq_axis_b", "eq_tempi_v1", "ptl_ion_mass_au", "ptl_ion_charge_eu",
    ]
    physical_params = {k: units[k] for k in keys_of_interest if k in units}

    return {
        "status": "ok",
        "case_dir": case_dir,
        "case_name": os.path.basename(case_dir),
        "physical_params": physical_params,
        "mesh": _summarize_mesh(case_dir),
        "fields_3d": _summarize_steps(case_dir, "xgc.3d.*.bp"),
        "fields_f3d": _summarize_steps(case_dir, "xgc.f3d.*.bp"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", help="XGC simulation directory")
    parser.add_argument("--output", default=None,
                        help="Write JSON report to this file (in addition to stdout)")
    args = parser.parse_args()

    if not os.path.isdir(args.case_dir):
        sys.exit(json.dumps({"status": "error", "message": f"Not a directory: {args.case_dir}"}))

    report = summarize(args.case_dir)
    out = json.dumps(report, indent=2)
    print(out)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fp:
            fp.write(out)


if __name__ == "__main__":
    main()
