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
    R_verts = el[:, 4]
    Z_verts = el[:, 5]
    phi_verts = np.full_like(R_verts, phi)

    f_verts = eval_field(field, R_verts, phi_verts, Z_verts,
                         coord=coord, sim=sim, time=time, quiet=quiet)

    triang = tri.Triangulation(R_verts, Z_verts, el[:, :3].astype(int))
    fig, ax = plt.subplots()
    tcf = ax.tripcolor(triang, f_verts, shading="flat")
    plt.colorbar(tcf, ax=ax, label=label.get(field, field))
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{shortlbl.get(field, field)}  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"mesh_{field}_{time:03d}.png", dpi=150)
    plt.show()
