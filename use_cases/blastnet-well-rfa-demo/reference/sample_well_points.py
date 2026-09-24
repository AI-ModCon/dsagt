"""Sample a WELL HDF5 file into a tabular point table for readiness checks.

AIDRIN reads an HDF5 file as one flat table and refuses a file whose datasets
have different shapes, which every WELL file has: scalar fields are
(n_traj, n_steps, W, H), velocity carries a trailing component axis, and the
coordinate arrays are one-dimensional.  This writes the tabular view the
readiness metrics need: one row per sampled grid point per snapshot, one column
per field, with the coordinates and the snapshot time beside them.  Sampling
rather than writing every point keeps the table at a size the metrics run on in
seconds; the sample is drawn once per snapshot from a seeded generator, so two
runs on the same file give the same rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def spatial_dims(handle: h5py.File) -> list[str]:
    """Return the coordinate names in axis order."""
    dims = handle["dimensions"]
    if "spatial_dims" in dims.attrs:
        return [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in dims.attrs["spatial_dims"]
        ]
    return [name for name in ("x", "y", "z") if name in dims]


def sample_indices(
    shape: tuple[int, ...], n_points: int, rng
) -> tuple[np.ndarray, ...]:
    """Draw n_points grid indices, one index array per spatial axis."""
    return tuple(rng.integers(0, size, n_points) for size in shape)


def point_rows(path: Path, n_points: int, seed: int, trajectory: int) -> list[dict]:
    """Read the file and return one row per sampled point per snapshot."""
    rows: list[dict] = []
    with h5py.File(path, "r") as handle:
        coords = spatial_dims(handle)
        scalars = sorted(handle["t0_fields"].keys())
        if not scalars:
            raise ValueError(f"{path} has no t0_fields datasets to sample")
        grid_shape = handle["t0_fields"][scalars[0]].shape[2:]
        times = np.asarray(handle["dimensions/time"]).reshape(-1)
        axes = [np.asarray(handle[f"dimensions/{name}"]) for name in coords]
        rng = np.random.default_rng(seed)

        for step, time in enumerate(times):
            index = sample_indices(grid_shape, n_points, rng)
            columns: dict[str, np.ndarray] = {}
            for name in scalars:
                plane = handle["t0_fields"][name][trajectory, step]
                columns[name] = np.asarray(plane)[index]
            if "velocity" in handle.get("t1_fields", {}):
                velocity = np.asarray(handle["t1_fields/velocity"][trajectory, step])
                for component, name in enumerate(coords):
                    columns[f"velocity_{name}"] = velocity[index][:, component]
            for axis, name in zip(axes, coords):
                columns[name] = axis[index[coords.index(name)]]

            for point in range(n_points):
                row = {"trajectory": trajectory, "snapshot": step, "time": float(time)}
                row.update(
                    {key: float(values[point]) for key, values in columns.items()}
                )
                rows.append(row)
    return rows


def write_table(rows: list[dict], output_file: Path) -> None:
    """Write the rows as CSV, creating the parent directory."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a WELL HDF5 file into a point table for readiness checks."
    )
    parser.add_argument("well_file", help="WELL HDF5 file to sample")
    parser.add_argument(
        "--output-file", required=True, help="CSV file to write the point table to"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=2000,
        help="grid points sampled per snapshot (default 2000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument(
        "--trajectory",
        type=int,
        default=0,
        help="trajectory index to sample (default 0)",
    )
    args = parser.parse_args()

    rows = point_rows(Path(args.well_file), args.n_points, args.seed, args.trajectory)
    write_table(rows, Path(args.output_file))
    print(f"{args.output_file}: {len(rows)} rows x {len(rows[0])} columns")
    print("columns: " + ", ".join(rows[0]))


if __name__ == "__main__":
    main()
