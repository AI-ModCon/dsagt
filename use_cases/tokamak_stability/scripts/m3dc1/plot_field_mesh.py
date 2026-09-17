"""Submodule: `import m3dc1.plot_field_mesh` — exposes mutable shortlbl/label globals.

plot_field_mesh evaluates the field at mesh vertices and renders with
matplotlib.tri.Triangulation (avoids interpolation artefacts near element edges).
"""
from pathlib import Path
import warnings
import numpy as np
from m3dc1 import eval_field  # noqa: F401

# Mutable globals set by calling code before invoking plot functions.
shortlbl: dict = {}
label: dict = {}

__all__ = ["plot_field_mesh", "eval_field", "shortlbl", "label"]


def plot_field_mesh(field, filename, time=0, coord="scalar", phi=0.0,
                    save=False, savedir="./", quiet=True):
    """Plot a field evaluated at mesh vertices using Triangulation rendering."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.tri as tri  # noqa: PLC0415
    import sys
    import os

    fio_lib = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    if fio_lib not in sys.path:
        sys.path.insert(0, fio_lib)

    try:
        from fpy import sim_data  # noqa: PLC0415
    except ImportError:
        raise ImportError("fpy.sim_data not importable; check your fusion-io installation.")

    sim = sim_data(filename=str(filename), time=time)
    mesh_obj = sim.get_mesh(quiet=True)
    el = mesh_obj.elements

    # Columns 0-2 of mesh/elements are floating-point values, not integer node
    # indices, so they cannot be used as triangle connectivity.  Evaluate the
    # field at all per-element (R, Z) positions, then deduplicate to obtain
    # unique vertices and a valid Delaunay triangulation.
    R_all = el[:, 4].astype(float)
    Z_all = el[:, 5].astype(float)
    phi_all = np.full_like(R_all, phi)

    f_all = eval_field(field, R_all, phi_all, Z_all,
                       coord=coord, sim=sim, time=time, quiet=quiet)

    _, idx = np.unique(np.column_stack([R_all, Z_all]), axis=0, return_index=True)
    R_unique = R_all[idx]
    Z_unique = Z_all[idx]
    f_unique = f_all[idx]

    triang = tri.Triangulation(R_unique, Z_unique)
    fig, ax = plt.subplots(figsize=(5, 7))
    tcf = ax.tripcolor(triang, f_unique, shading="gouraud")
    plt.colorbar(tcf, ax=ax, label=label.get(field, field))
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{shortlbl.get(field, field)}  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"mesh_{field}_{time:03d}.png", dpi=150)
    if plt.isinteractive():
        plt.show()
