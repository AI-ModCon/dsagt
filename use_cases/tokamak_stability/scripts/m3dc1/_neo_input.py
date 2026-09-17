"""Private module: run write_neo_input subprocess and read neo_input.nc."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _find_write_neo_input():
    fio_dir = os.environ.get("FIO_INSTALL_DIR")
    if fio_dir:
        candidate = Path(fio_dir) / "bin" / "write_neo_input"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("write_neo_input")
    if found:
        return found
    raise RuntimeError(
        "write_neo_input executable not found. "
        "Set $FIO_INSTALL_DIR to the fusion-io install prefix "
        "or add write_neo_input to $PATH."
    )


def run_write_neo_input(
    c1h5_path,
    timeslice,
    psi_start=0.01,
    psi_end=0.99,
    nr=50,
    ntheta=400,
    nphi=1,
    dl=0.01,
    workdir=None,
    verbose=False,
):
    """Run write_neo_input and return (nc_path, tmpdir_obj).

    tmpdir_obj is a TemporaryDirectory instance if workdir was None (keep a
    reference to prevent early cleanup), or None if workdir was provided.
    nc_path is the absolute Path to neo_input.nc inside workdir.
    """
    exe = _find_write_neo_input()
    c1h5_path = Path(c1h5_path).resolve()

    tmpdir_obj = None
    if workdir is None:
        tmpdir_obj = tempfile.TemporaryDirectory()
        workdir = Path(tmpdir_obj.name)
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "-m3dc1", str(c1h5_path), str(timeslice),
        "-psi_start", str(psi_start),
        "-psi_end", str(psi_end),
        "-nr", str(nr),
        "-ntheta", str(ntheta),
        "-nphi", str(nphi),
        "-dl", str(dl),
        "-bootstrap", "0",
    ]

    subprocess.run(
        cmd,
        cwd=str(workdir),
        check=True,
        capture_output=not verbose,
    )

    nc_path = workdir / "neo_input.nc"
    if not nc_path.exists():
        raise RuntimeError(
            f"write_neo_input completed but {nc_path} was not created."
        )
    return nc_path, tmpdir_obj


def read_neo_input(nc_path):
    """Read neo_input.nc; return a plain dict of NumPy arrays.

    Tries netCDF4 first, falls back to scipy.io.netcdf_file.
    Adds derived 'psi_norm' = (psi - psi_0) / (psi_1 - psi_0).
    """
    nc_path = str(nc_path)

    _3d_vars = ["R", "Z", "Jac", "B_R", "B_Phi", "B_Z", "p_3d", "ne_3d", "ni_3d"]
    _1d_vars = ["q", "psi", "ne0", "Te0", "ni0", "Ti0", "b2_fa", "bmag_fa", "Phi"]
    _opt_1d = ["bx_fa"]

    try:
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(nc_path, "r") as ds:
            data = {v: np.array(ds.variables[v][:]) for v in _3d_vars + _1d_vars}
            for v in _opt_1d:
                if v in ds.variables:
                    data[v] = np.array(ds.variables[v][:])
            data["ion_mass"] = float(ds.ion_mass)
            data["psi_0"] = float(ds.psi_0)
            data["psi_1"] = float(ds.psi_1)

    except ImportError:
        from scipy.io import netcdf_file  # noqa: PLC0415

        with netcdf_file(nc_path, "r", mmap=False) as ds:
            data = {v: np.array(ds.variables[v].data) for v in _3d_vars + _1d_vars}
            for v in _opt_1d:
                if v in ds.variables:
                    data[v] = np.array(ds.variables[v].data)
            data["ion_mass"] = float(ds.ion_mass)
            data["psi_0"] = float(ds.psi_0)
            data["psi_1"] = float(ds.psi_1)

    psi_0 = data["psi_0"]
    psi_1 = data["psi_1"]
    data["psi_norm"] = (data["psi"] - psi_0) / (psi_1 - psi_0)
    return data


def flux_surface_average(quantity, jacobian, theta, phi):
    """Flux-surface average: <f> = integral(f*J dtheta dphi) / integral(J dtheta dphi).

    quantity, jacobian: shape (ntheta, nphi) for a single flux surface.
    theta: 1-D array of length ntheta.
    phi:   1-D array of length nphi.
    """
    try:
        _trapz = np.trapezoid   # numpy >= 2.0
    except AttributeError:
        _trapz = np.trapz       # numpy < 2.0
    denom = _trapz(_trapz(jacobian, x=theta, axis=0), x=phi, axis=0)
    numer = _trapz(_trapz(quantity * jacobian, x=theta, axis=0), x=phi, axis=0)
    return numer / denom
