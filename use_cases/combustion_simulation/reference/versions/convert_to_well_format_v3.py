"""Convert a BlastNet trajectory directory to WELL HDF5 format (v3).

Supports 2D and 3D datasets. Grid, time, and BC handling adapt automatically
based on the dataset name and info.json contents.
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

# BC config per dataset: list of (dim_name, group_name, bc_type, mask_style)
#   mask_style "all"      → True for every grid point (periodic)
#   mask_style "endpoint" → True only at first and last index (wall / open)
BC_CONFIGS = {
    "lifted_hydrogen_jet": [
        ("x", "x_open",     "OPEN",     "endpoint"),
        ("y", "y_open",     "OPEN",     "endpoint"),
    ],
    "nonreacting_channel_flow": [
        ("x", "x_periodic", "PERIODIC", "all"),
        ("y", "y_wall",     "WALL",     "endpoint"),
        ("z", "z_periodic", "PERIODIC", "all"),
    ],
}

_STR_DT = h5py.string_dtype()


def _str_attr(grp_or_file, name, lst):
    if not lst:
        grp_or_file.attrs[name] = np.array([], dtype=np.float64)
    else:
        grp_or_file.attrs.create(name, data=np.array(lst, dtype=object), dtype=_STR_DT)


def load_dat(path):
    try:
        return np.fromfile(path, dtype=np.float32)
    except Exception:
        return np.fromfile(path, dtype=np.float64).astype(np.float32)


def load_grid_1d(traj_dir, grid_path_str, expected_size, ny_nx_shape=None):
    """Load a grid file and return a 1D coordinate array.

    If the file contains exactly expected_size elements it is already 1D.
    Otherwise assumes a 2D (Ny, Nx) Fortran-style layout and extracts coords:
      - first call (x): first row of (Ny, Nx) array
      - second call (y): first column of (Ny, Nx) array
    ny_nx_shape is required for the 2D case.
    """
    flat = load_dat(traj_dir / grid_path_str.lstrip("./"))
    if flat.size == expected_size:
        return flat.astype(np.float32)
    # 2D grid layout
    Ny, Nx = ny_nx_shape
    arr2d = flat.reshape(Ny, Nx)
    if expected_size == Nx:
        return arr2d[0, :].astype(np.float32)   # row 0 → x-coordinates
    else:
        return arr2d[:, 0].astype(np.float32)   # col 0 → y-coordinates


def convert(traj_dir: Path, output_file: Path, dry_run: bool = False):
    traj_dir     = Path(traj_dir)
    dataset_name = traj_dir.parent.name

    with open(traj_dir / "info.json") as f:
        info = json.load(f)

    g         = info["global"]
    Nxyz      = g["Nxyz"]                   # [Nx, Ny] or [Nx, Ny, Nz]
    n_dims    = len(Nxyz)
    n_steps   = g["snapshots"]
    variables = g["variables"]
    dim_names = ["x", "y", "z"][:n_dims]

    # Time array
    local_by_id = {entry["id"]: entry for entry in info["local"]}
    sorted_ids  = sorted(local_by_id)
    if "time-step snapshot [s]" in g:
        dt    = g["time-step snapshot [s]"]
        times = (np.arange(n_steps) * dt).astype(np.float32)
    else:
        times = np.array([local_by_id[sid]["time"] for sid in sorted_ids],
                         dtype=np.float32)

    # Grid coordinates (1D per dimension)
    ny_nx = (Nxyz[1], Nxyz[0]) if n_dims == 2 else None   # only needed for 2D layout
    grid_coords = {}
    for dname in dim_names:
        expected = Nxyz[dim_names.index(dname)]
        grid_coords[dname] = load_grid_1d(
            traj_dir, g["grid"][dname], expected, ny_nx)

    # Separate scalar and velocity variables
    scalar_blastnet = [v for v in variables if v not in VELOCITY_VARS]
    vel_blastnet    = [v for v in variables if v in VELOCITY_VARS]
    scalar_well     = [SCALAR_MAP.get(v, v) for v in scalar_blastnet]
    n_vel_dims      = len(vel_blastnet)

    print(f"Dataset       : {dataset_name}")
    print(f"Traj dir      : {traj_dir}")
    print(f"n_spatial_dims: {n_dims}  Nxyz={Nxyz}")
    print(f"Steps         : {n_steps}  time=[{times[0]:.4g} .. {times[-1]:.4g}]")
    print(f"Scalars       : {scalar_well}")
    print(f"Velocity      : {vel_blastnet}  ({n_vel_dims}D)")
    print(f"Output        : {output_file}")

    if dry_run:
        print("[dry-run] No file written.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    bc_config   = BC_CONFIGS.get(dataset_name, [])
    field_shape = tuple([1, n_steps] + list(Nxyz))
    vel_shape   = tuple([1, n_steps] + list(Nxyz) + [n_vel_dims])
    dim_varying = np.array([True] * n_dims)

    if not bc_config:
        print(f"WARNING: no BC config for '{dataset_name}', "
              "boundary_conditions group will be empty.")

    with h5py.File(output_file, "w") as f:
        # Root attributes
        f.attrs["dataset_name"]   = dataset_name
        f.attrs["grid_type"]      = "cartesian"
        f.attrs["n_spatial_dims"] = n_dims
        f.attrs["n_trajectories"] = 1
        _str_attr(f, "simulation_parameters", [])

        # dimensions
        dim_grp = f.create_group("dimensions")
        _str_attr(dim_grp, "spatial_dims", dim_names)

        ds_t = dim_grp.create_dataset("time", data=times)
        ds_t.attrs["sample_varying"] = False
        ds_t.attrs["time_varying"]   = True

        for dname in dim_names:
            ds = dim_grp.create_dataset(dname, data=grid_coords[dname])
            ds.attrs["sample_varying"] = False
            ds.attrs["time_varying"]   = False

        # boundary_conditions
        bc_grp = f.create_group("boundary_conditions")
        for dim_name, grp_label, bc_type, mask_style in bc_config:
            size = Nxyz[dim_names.index(dim_name)]
            mask = (np.ones(size, dtype=np.int8) if mask_style == "all"
                    else np.array([1] + [0] * (size - 2) + [1], dtype=np.int8))
            grp = bc_grp.create_group(grp_label)
            _str_attr(grp, "associated_dims",   [dim_name])
            _str_attr(grp, "associated_fields", [])
            grp.attrs["bc_type"]        = bc_type
            grp.attrs["sample_varying"] = False
            grp.attrs["time_varying"]   = False
            grp.create_dataset("mask", data=mask)

        # scalars (empty placeholder)
        sc_grp = f.create_group("scalars")
        _str_attr(sc_grp, "field_names", [])

        # t0_fields
        t0_grp = f.create_group("t0_fields")
        _str_attr(t0_grp, "field_names", scalar_well)
        t0_ds = {}
        for wname in scalar_well:
            ds = t0_grp.create_dataset(
                wname, shape=field_shape, dtype=np.float32,
                chunks=tuple([1, 1] + list(Nxyz)))
            ds.attrs["dim_varying"]    = dim_varying
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"]   = True
            t0_ds[wname] = ds

        # t1_fields — velocity
        t1_grp = f.create_group("t1_fields")
        _str_attr(t1_grp, "field_names", ["velocity"] if n_vel_dims > 0 else [])
        vel_ds = None
        if n_vel_dims > 0:
            vel_ds = t1_grp.create_dataset(
                "velocity", shape=vel_shape, dtype=np.float32,
                chunks=tuple([1, 1] + list(Nxyz) + [n_vel_dims]))
            vel_ds.attrs["dim_varying"]    = dim_varying
            vel_ds.attrs["sample_varying"] = True
            vel_ds.attrs["time_varying"]   = True

        # t2_fields (empty)
        t2_grp = f.create_group("t2_fields")
        _str_attr(t2_grp, "field_names", [])

        # Write data one timestep at a time
        for step_idx, sid in enumerate(sorted_ids):
            print(f"  step {step_idx+1}/{n_steps}", end="\r", flush=True)
            entry = local_by_id[sid]

            for bname, wname in zip(scalar_blastnet, scalar_well):
                fpath = traj_dir / entry[f"{bname} filename"].lstrip("./")
                t0_ds[wname][0, step_idx] = load_dat(fpath).reshape(*Nxyz)

            if vel_ds is not None:
                vel_stack = np.stack(
                    [load_dat(traj_dir / entry[f"{v} filename"].lstrip("./")).reshape(*Nxyz)
                     for v in vel_blastnet],
                    axis=-1)
                vel_ds[0, step_idx] = vel_stack

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
        dataset_name = traj_dir.parent.name
        traj_name    = traj_dir.name
        output_file  = Path(args.output_path) / f"{dataset_name}_{traj_name}.hdf5"

    convert(traj_dir, output_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
