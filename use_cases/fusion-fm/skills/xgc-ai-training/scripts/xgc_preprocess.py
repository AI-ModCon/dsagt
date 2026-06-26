#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC

"""Preprocess an XGC simulation directory into npz files for AI training.

Reads ADIOS2 BP files and writes:
  out_dir/mesh.npz          - static mesh (rz, conn, psi, region, nextnode)
  out_dir/step_NNNNN.npz    - per-timestep fields, shape [nphi, n_nodes] float32
  out_dir/meta.json         - case metadata, step lists, field availability

Output: JSON status to stdout (and optionally --output file).
Exit 0 on success; exit 1 on error.
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

_FIELDS_3D  = ["dpot", "eden", "iden"]
_FIELDS_F3D = ["e_den", "e_T_para", "e_T_perp", "e_u_para",
               "i_den", "i_T_para", "i_T_perp", "i_u_para"]


def _parse_step(filename):
    m = re.search(r'\.(\d+)\.bp$', filename)
    return int(m.group(1)) if m else -1


def _find_steps(case_dir, pattern):
    files = sorted(glob.glob(os.path.join(case_dir, pattern)))
    return {_parse_step(f): f for f in files if _parse_step(f) >= 0}


def _normalize_phi_first(arr, nphi, n_nodes):
    if arr.shape == (nphi, n_nodes):
        return arr.astype(np.float32)
    if arr.shape == (n_nodes, nphi):
        return arr.T.astype(np.float32)
    raise ValueError(f"Shape {arr.shape} does not match ({nphi},{n_nodes}) or ({n_nodes},{nphi})")


def _get_nphi(bp_path):
    with adios2.FileReader(bp_path) as f:
        avail = f.available_variables()
        if "nphi" in avail:
            return int(f.read("nphi"))
        for name in ("dpot", "eden", "iden"):
            if name in avail:
                dims = [int(x.strip()) for x in avail[name]["Shape"].split(",")]
                return min(dims)
    return None


def _read_mesh(case_dir):
    mesh_path = os.path.join(case_dir, "xgc.mesh.bp")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"xgc.mesh.bp not found in {case_dir}")
    arrays = {}
    with adios2.FileReader(mesh_path) as f:
        avail = f.available_variables()
        arrays["n_nodes"] = np.array(int(f.read("n_n")), dtype=np.int32)
        arrays["n_cells"] = np.array(int(f.read("n_t")), dtype=np.int32)
        arrays["rz"]   = f.read("rz").astype(np.float32)
        arrays["conn"] = f.read("nd_connect_list").astype(np.int32)
        for name in ("psi", "theta"):
            if name in avail:
                arrays[name] = f.read(name).astype(np.float32)
        for name in ("region", "nextnode"):
            if name in avail:
                arrays[name] = f.read(name).astype(np.int32)
    return arrays


def _read_fields(bp_path, field_names, nphi, n_nodes):
    out, time_val = {}, None
    with adios2.FileReader(bp_path) as f:
        avail = f.available_variables()
        if "time" in avail:
            time_val = float(f.read("time"))
        for name in field_names:
            if name not in avail:
                continue
            try:
                out[name] = _normalize_phi_first(f.read(name), nphi, n_nodes)
            except ValueError as e:
                print(f"  WARNING: skipping {name}: {e}", file=sys.stderr)
    return out, time_val


def preprocess(case_dir, out_dir, fields_3d=None, fields_f3d=None,
               steps=None, no_f3d=False, overwrite=False):
    case_dir = os.path.abspath(case_dir)
    out_dir  = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fields_3d  = fields_3d  or _FIELDS_3D
    fields_f3d = fields_f3d or _FIELDS_F3D

    mesh = _read_mesh(case_dir)
    n_nodes = int(mesh["n_nodes"])

    mesh_out = os.path.join(out_dir, "mesh.npz")
    if overwrite or not os.path.exists(mesh_out):
        np.savez_compressed(mesh_out, **mesh)

    step_3d  = _find_steps(case_dir, "xgc.3d.*.bp")
    step_f3d = _find_steps(case_dir, "xgc.f3d.*.bp") if not no_f3d else {}

    if not step_3d:
        raise RuntimeError(f"No xgc.3d.*.bp files in {case_dir}")

    nphi = _get_nphi(next(iter(step_3d.values())))

    if steps is not None:
        steps_to_do = sorted(s for s in step_3d if s in set(steps))
    else:
        steps_to_do = sorted(step_3d)

    field_availability = {}
    saved_steps = []

    for step in steps_to_do:
        out_path = os.path.join(out_dir, f"step_{step:05d}.npz")
        if not overwrite and os.path.exists(out_path):
            d = np.load(out_path)
            for k in d.files:
                if k not in ("time", "step", "has_f3d"):
                    field_availability.setdefault(k, []).append(step)
            saved_steps.append(step)
            continue

        save_dict = {}
        data_3d, time_val = _read_fields(step_3d[step], fields_3d, nphi, n_nodes)
        save_dict.update(data_3d)

        has_f3d = step in step_f3d
        if has_f3d:
            data_f3d, _ = _read_fields(step_f3d[step], fields_f3d, nphi, n_nodes)
            save_dict.update(data_f3d)

        if not save_dict:
            continue

        save_dict["step"]    = np.array(step, dtype=np.int32)
        save_dict["has_f3d"] = np.array(has_f3d, dtype=bool)
        if time_val is not None:
            save_dict["time"] = np.array(time_val, dtype=np.float64)

        np.savez_compressed(out_path, **save_dict)
        saved_steps.append(step)
        for k in save_dict:
            if k not in ("time", "step", "has_f3d"):
                field_availability.setdefault(k, []).append(step)

    all_fields = list(field_availability.keys())
    meta = {
        "case_dir":           case_dir,
        "case_name":          os.path.basename(case_dir),
        "n_nodes":            n_nodes,
        "nphi":               nphi,
        "n_cells":            int(mesh["n_cells"]),
        "field_names":        all_fields,
        "steps":              saved_steps,
        "n_steps":            len(saved_steps),
        "f3d_steps":          sorted(step_f3d.keys()),
        "field_availability": field_availability,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return {
        "status":      "ok",
        "case_dir":    case_dir,
        "out_dir":     out_dir,
        "n_nodes":     n_nodes,
        "nphi":        nphi,
        "n_steps":     len(saved_steps),
        "field_names": all_fields,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", help="XGC simulation directory")
    parser.add_argument("out_dir",  help="Output directory for npz files")
    parser.add_argument("--fields_3d",  default=",".join(_FIELDS_3D),
                        help=f"Comma-separated 3d fields (default: {','.join(_FIELDS_3D)})")
    parser.add_argument("--fields_f3d", default=",".join(_FIELDS_F3D),
                        help=f"Comma-separated f3d fields (default: {','.join(_FIELDS_F3D)})")
    parser.add_argument("--no_f3d",   action="store_true", help="Skip f3d files")
    parser.add_argument("--steps",    default=None,
                        help="Comma-separated step indices (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--output",    default=None,
                        help="Write JSON status to this file (in addition to stdout)")
    args = parser.parse_args()

    fields_3d  = [f.strip() for f in args.fields_3d.split(",")  if f.strip()]
    fields_f3d = [f.strip() for f in args.fields_f3d.split(",") if f.strip()]
    steps = [int(s) for s in args.steps.split(",")] if args.steps else None

    try:
        report = preprocess(
            args.case_dir, args.out_dir,
            fields_3d=fields_3d, fields_f3d=fields_f3d,
            steps=steps, no_f3d=args.no_f3d, overwrite=args.overwrite,
        )
    except Exception as e:
        report = {"status": "error", "message": str(e)}

    out = json.dumps(report, indent=2)
    print(out)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fp:
            fp.write(out)

    sys.exit(0 if report.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
