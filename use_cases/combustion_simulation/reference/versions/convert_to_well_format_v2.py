"""Convert BlastNet lifted_hydrogen_jet data to WELL HDF5 format (v2).

Changes from v1:
- Fixed data reshape: .dat files are (Nx, Ny) C-order; removed incorrect .T transpose
- Species names changed to mass_fraction_<species> convention
- Root attributes: simulation_parameters=[], Re_jet removed
- BC mask dtype changed from bool to int8
"""

import json
import argparse
from pathlib import Path

import numpy as np
import h5py

# BlastNet variable name → WELL field name
SCALAR_MAP = {
    "RHO_kgm-3": "density",
    "P_Pa":       "pressure",
    "T_K":        "temperature",
    "YH":         "mass_fraction_h",
    "YH2":        "mass_fraction_h2",
    "YO":         "mass_fraction_o",
    "YO2":        "mass_fraction_o2",
    "YOH":        "mass_fraction_oh",
    "YH2O":       "mass_fraction_h2o",
    "YHO2":       "mass_fraction_ho2",
    "YH2O2":      "mass_fraction_h2o2",
}

VELOCITY_VARS = ["UX_ms-1", "UY_ms-1", "UZ_ms-1"]

_STR_DT = h5py.string_dtype()


def _str_attr(grp_or_file, name, lst):
    """Write a variable-length string array attribute; empty uses float64."""
    if not lst:
        grp_or_file.attrs[name] = np.array([], dtype=np.float64)
    else:
        grp_or_file.attrs.create(name, data=np.array(lst, dtype=object), dtype=_STR_DT)


def load_dat(path):
    try:
        return np.fromfile(path, dtype=np.float32)
    except Exception:
        return np.fromfile(path, dtype=np.float64).astype(np.float32)


def convert(traj_dir: Path, output_file: Path, dry_run: bool = False):
    traj_dir = Path(traj_dir)

    with open(traj_dir / "info.json") as f:
        info = json.load(f)

    g = info["global"]
    Nx, Ny    = g["Nxyz"]          # [1600, 2000]
    n_steps   = g["snapshots"]     # 201
    dt        = g["time-step snapshot [s]"]
    variables = g["variables"]

    # Grid files are stored flat in (Ny, Nx) Fortran-style order; data files
    # are stored in (Nx, Ny) C order — different conventions, same flat size.
    x_flat = load_dat(traj_dir / g["grid"]["x"].lstrip("./"))
    y_flat = load_dat(traj_dir / g["grid"]["y"].lstrip("./"))
    x_1d = x_flat.reshape(Ny, Nx)[0, :]   # shape (Nx,)
    y_1d = y_flat.reshape(Ny, Nx)[:, 0]   # shape (Ny,)

    times = (np.arange(n_steps) * dt).astype(np.float32)

    scalar_blastnet = [v for v in variables if v not in VELOCITY_VARS]
    vel_blastnet    = [v for v in variables if v in VELOCITY_VARS]
    scalar_well     = [SCALAR_MAP.get(v, v) for v in scalar_blastnet]
    n_vel_dims      = len(vel_blastnet)

    print(f"Dataset : lifted_hydrogen_jet")
    print(f"Traj dir: {traj_dir}")
    print(f"Grid    : Nx={Nx}, Ny={Ny}")
    print(f"Steps   : {n_steps}  dt={dt} s")
    print(f"Scalars : {scalar_well}")
    print(f"Velocity: {vel_blastnet}  ({n_vel_dims}D)")
    print(f"Output  : {output_file}")

    if dry_run:
        print("[dry-run] No file written.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup: step_id → {var: path}
    local_lookup = {}
    for entry in info["local"]:
        sid = entry["id"]
        local_lookup[sid] = {v: traj_dir / entry[f"{v} filename"].lstrip("./")
                             for v in variables}

    with h5py.File(output_file, "w") as f:
        # Root attributes
        f.attrs["dataset_name"]   = "lifted_hydrogen_jet"
        f.attrs["grid_type"]      = "cartesian"
        f.attrs["n_spatial_dims"] = 2
        f.attrs["n_trajectories"] = 1
        _str_attr(f, "simulation_parameters", [])

        # dimensions
        dim_grp = f.create_group("dimensions")
        _str_attr(dim_grp, "spatial_dims", ["x", "y"])

        ds_t = dim_grp.create_dataset("time", data=times)
        ds_t.attrs["sample_varying"] = False
        ds_t.attrs["time_varying"]   = True

        ds_x = dim_grp.create_dataset("x", data=x_1d.astype(np.float32))
        ds_x.attrs["sample_varying"] = False
        ds_x.attrs["time_varying"]   = False

        ds_y = dim_grp.create_dataset("y", data=y_1d.astype(np.float32))
        ds_y.attrs["sample_varying"] = False
        ds_y.attrs["time_varying"]   = False

        # boundary_conditions (x: open inflow/outflow, y: open pressure outlet)
        bc_grp = f.create_group("boundary_conditions")
        for dim_name, size, label in [("x", Nx, "x_open"), ("y", Ny, "y_open")]:
            mask = np.zeros(size, dtype=np.int8)
            mask[0] = mask[-1] = 1
            grp = bc_grp.create_group(label)
            _str_attr(grp, "associated_dims",   [dim_name])
            _str_attr(grp, "associated_fields", [])
            grp.attrs["bc_type"]        = "OPEN"
            grp.attrs["sample_varying"] = False
            grp.attrs["time_varying"]   = False
            grp.create_dataset("mask", data=mask)

        # scalars (empty placeholder)
        sc_grp = f.create_group("scalars")
        _str_attr(sc_grp, "field_names", [])

        # t0_fields — pre-allocate, write one step at a time to limit memory use
        t0_grp = f.create_group("t0_fields")
        _str_attr(t0_grp, "field_names", scalar_well)
        t0_ds = {}
        for wname in scalar_well:
            ds = t0_grp.create_dataset(
                wname, shape=(1, n_steps, Nx, Ny), dtype=np.float32,
                chunks=(1, 1, Nx, Ny))
            ds.attrs["dim_varying"]    = np.array([True, True])
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"]   = True
            t0_ds[wname] = ds

        # t1_fields — velocity vector
        t1_grp = f.create_group("t1_fields")
        _str_attr(t1_grp, "field_names", ["velocity"])
        vel_ds = None
        if n_vel_dims > 0:
            vel_ds = t1_grp.create_dataset(
                "velocity", shape=(1, n_steps, Nx, Ny, n_vel_dims),
                dtype=np.float32, chunks=(1, 1, Nx, Ny, n_vel_dims))
            vel_ds.attrs["dim_varying"]    = np.array([True, True])
            vel_ds.attrs["sample_varying"] = True
            vel_ds.attrs["time_varying"]   = True

        # t2_fields (empty)
        t2_grp = f.create_group("t2_fields")
        _str_attr(t2_grp, "field_names", [])

        # Write data one timestep at a time
        for sid in range(n_steps):
            print(f"  step {sid+1}/{n_steps}", end="\r", flush=True)
            paths = local_lookup[sid]

            for bname, wname in zip(scalar_blastnet, scalar_well):
                arr = load_dat(paths[bname]).reshape(Nx, Ny)  # (Nx, Ny) C-order
                t0_ds[wname][0, sid, :, :] = arr

            if vel_ds is not None:
                vel_stack = np.stack(
                    [load_dat(paths[v]).reshape(Nx, Ny) for v in vel_blastnet],
                    axis=-1)  # (Nx, Ny, n_dims)
                vel_ds[0, sid, :, :, :] = vel_stack

        print(f"\nWrote {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a BlastNet trajectory directory to WELL HDF5 format.")
    parser.add_argument("traj_dir",
                        help="Path to trajectory directory containing info.json")
    parser.add_argument("--output-path", default="./well_output_v2",
                        help="Output directory (default: ./well_output_v2)")
    parser.add_argument("--output-file",
                        help="Explicit output file path (overrides --output-path)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Derive filename from parent dataset dir and trajectory subdir names
        dataset_name = traj_dir.parent.name
        traj_name    = traj_dir.name
        output_file  = Path(args.output_path) / f"{dataset_name}_{traj_name}.hdf5"

    convert(traj_dir, output_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
