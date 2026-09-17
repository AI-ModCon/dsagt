#!/usr/bin/env python3
"""Cut a BlastNet trajectory and its WELL reference file down to the first K snapshots.

Full trajectories are tens of GB, too large to host for a demo. This writes,
under ``out_dir``, the layout the use-case README expects::

    data/blastnet_data/<dataset>/<trajectory>/   info.json (K snapshots), grid/,
                                                 chem_thermo_tran/, data/*_id####.dat
    data/holdout/well_output/<reference>.hdf5    time-varying datasets sliced to K

The subset converts with ``convert_to_well_format_v4.py`` and matches the
sliced reference under ``check_well_output.py``, because the converter names
data files ``data/<variable>_id<id:04d>.dat`` from ``info.json`` and every
time-varying WELL dataset carries time on axis 1 (``dimensions/time`` on axis 0).

Usage::

    python3 make_demo_subset.py <traj_dir> <reference.hdf5> <out_dir> --steps 5
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py


def subset_trajectory(traj_dir: Path, out_traj: Path, steps: int) -> None:
    info = json.loads((traj_dir / "info.json").read_text())
    local = sorted(info["local"], key=lambda entry: entry["id"])
    if steps > len(local):
        raise SystemExit(
            f"--steps {steps} exceeds the {len(local)} snapshots in {traj_dir}"
        )
    keep = local[:steps]
    info["local"] = keep
    info["global"]["snapshots"] = steps

    out_traj.mkdir(parents=True)
    (out_traj / "info.json").write_text(json.dumps(info, indent=2))
    for sub in ("grid", "chem_thermo_tran"):
        if (traj_dir / sub).is_dir():
            shutil.copytree(traj_dir / sub, out_traj / sub)
    # 2D grids may sit in the trajectory root rather than grid/.
    for grid_path in info["global"]["grid"].values():
        src = traj_dir / grid_path.lstrip("./")
        dst = out_traj / grid_path.lstrip("./")
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    (out_traj / "data").mkdir()
    for entry in keep:
        for var in info["global"]["variables"]:
            name = f"{var}_id{entry['id']:04d}.dat"
            shutil.copy2(traj_dir / "data" / name, out_traj / "data" / name)


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def slice_reference(ref_path: Path, out_path: Path, steps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(ref_path, "r") as src, h5py.File(out_path, "w") as dst:
        _copy_attrs(src, dst)

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                _copy_attrs(obj, dst.require_group(name))
                return
            if name == "dimensions/time":
                data = obj[:steps]
            elif obj.attrs.get("time_varying", False):
                data = obj[:, :steps]
            else:
                data = obj[()]
            new = dst.create_dataset(name, data=data, dtype=obj.dtype)
            _copy_attrs(obj, new)

        src.visititems(visit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("traj_dir", type=Path, help="BlastNet trajectory directory")
    parser.add_argument(
        "reference", type=Path, help="Full WELL HDF5 file for that trajectory"
    )
    parser.add_argument("out_dir", type=Path, help="Bundle root; must not exist")
    parser.add_argument(
        "--steps", type=int, default=5, help="Snapshots to keep (default 5)"
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists")

    traj_dir = args.traj_dir.resolve()
    dataset = traj_dir.parent.name
    out_traj = args.out_dir / "data" / "blastnet_data" / dataset / traj_dir.name
    subset_trajectory(traj_dir, out_traj, args.steps)
    out_ref = args.out_dir / "data" / "holdout" / "well_output" / args.reference.name
    slice_reference(args.reference, out_ref, args.steps)
    print(f"trajectory : {out_traj}")
    print(f"reference  : {out_ref}")
    print(f"tar czf comb_flow_uni_data.tar.gz -C {args.out_dir} data")


if __name__ == "__main__":
    main()
