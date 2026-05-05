"""Standalone post-processing tools for M3D-C1 simulation output.

Each function does one computation and returns plain Python / NumPy data.
None of the public functions retain file handles or simulation objects between
calls, making them safe to use from an agentic pipeline.

Functions that require the ``m3dc1`` / ``fpy`` libraries are marked in the
docstring. All other functions depend only on ``h5py`` and ``numpy``.

Functions
---------
Inspection (h5py only):
    read_c1input           Parse C1input namelist
    list_time_snapshots    Discover time_NNN.h5 snapshot files
    read_snapshot_time     Physical time of a snapshot
    read_scalar_traces     Global time-trace scalars from C1.h5
    read_case_metadata     Combined summary of parameters and geometry

Mesh / evaluation grid (h5py / numpy only):
    read_mesh_vertices     Unique (R, Z) vertex positions
    make_evaluation_grid   Build (R, Z, phi) arrays for eval_field()

Growth rate (h5py / numpy only):
    compute_ke_growth_trace  Kinetic energy vs time from scalars
    compute_growth_rate      Mean linear growth rate (1/tau_A)

Equilibrium profiles (requires m3dc1 + fpy):
    compute_flux_average_profiles  Radial flux-surface-averaged profiles
    compute_q95                    Safety factor at psi_norm = 0.95
    compute_miller_geometry        R0, a, kappa, delta of the LCFS

Perturbed fields (requires m3dc1 + fpy):
    compute_perturbed_fields  Perturbed fields at arbitrary (R, Z, phi) points

Spectral analysis (requires m3dc1 + fpy):
    compute_poloidal_spectrum   Poloidal m-spectrum of a single field
    compute_standard_spectra    Spectra for p, B_R, B_Z, B_phi
"""
from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _in_case_dir(case_dir: Path):
    """Temporarily change the working directory to ``case_dir``.

    The m3dc1 library reads auxiliary files (geqdsk, equilibrium data) using
    relative paths from the working directory. This context manager provides
    that guarantee for m3dc1-dependent functions while leaving the global cwd
    restored on exit, even if an exception is raised.
    """
    old = os.getcwd()
    os.chdir(case_dir)
    try:
        yield
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# Inspection tools (no m3dc1 dependency)
# ---------------------------------------------------------------------------

def read_c1input(case_dir: str | Path) -> dict:
    """Parse the C1input namelist and return simulation parameters as a dict.

    Reads the Fortran namelist file ``C1input`` in the case directory and
    extracts the parameters most relevant to post-processing. Any parameter
    absent from the file is returned with value ``None``.

    Args:
        case_dir: Path to the M3D-C1 case directory containing ``C1input``.

    Returns:
        Dict with keys: ``"ntor"`` (int), ``"pscale"`` (float),
        ``"batemanscale"`` (float), ``"dt"`` (float), ``"ntimemax"`` (int),
        ``"ntimepr"`` (int), ``"linear"`` (int), ``"numvar"`` (int),
        ``"ion_mass"`` (float), ``"zeff"`` (float).
    """
    int_keys = {"ntor", "ntimemax", "ntimepr", "linear", "numvar"}
    float_keys = {"pscale", "batemanscale", "dt", "ion_mass", "zeff"}
    all_keys = int_keys | float_keys
    params: dict = {k: None for k in all_keys}

    c1input = Path(case_dir) / "C1input"
    if not c1input.exists():
        return params

    with c1input.open("r") as fh:
        for line in fh:
            line = line.split("!")[0].strip()
            if not line or "=" not in line:
                continue
            key_part, _, val_part = line.partition("=")
            key = key_part.strip().lower()
            val_str = val_part.strip().split()[0].rstrip(",") if val_part.strip() else None
            if key not in all_keys or val_str is None:
                continue
            try:
                if key in int_keys:
                    params[key] = int(float(val_str))
                else:
                    params[key] = float(val_str)
            except ValueError:
                pass

    return params


def list_time_snapshots(case_dir: str | Path) -> list[int]:
    """Return sorted list of time snapshot indices present in the case directory.

    Scans for files matching the pattern ``time_NNN.h5`` where NNN is an
    integer. Does not open the files.

    Args:
        case_dir: Path to the M3D-C1 case directory.

    Returns:
        Sorted list of integer indices, e.g. ``[0, 1]`` for ``time_000.h5``
        and ``time_001.h5``. Empty list if no matching files are found.
    """
    indices = []
    for path in sorted(Path(case_dir).glob("time_*.h5")):
        stem = path.stem          # e.g. "time_001"
        parts = stem.split("_", maxsplit=1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return sorted(set(indices))


def read_snapshot_time(
    case_dir: str | Path,
    time_idx: int,
    units: str = "alfven",
) -> float:
    """Return the simulation time corresponding to a snapshot index.

    Reads the ``time`` attribute stored directly on the snapshot group in
    ``time_NNN.h5`` (or the equivalent group inside ``C1.h5``). No m3dc1
    library needed for Alfvén-time results.

    Args:
        case_dir: Path to the M3D-C1 case directory.
        time_idx: Snapshot index (the NNN in ``time_NNN.h5``).
        units:    ``"alfven"`` (default) — return simulation time in Alfvén
                  times directly from the file attribute.  ``"s"`` or
                  ``"mks"`` — return physical seconds via
                  ``m3dc1.get_time_of_slice``; falls back to Alfvén times
                  with a warning if m3dc1 is not installed.

    Returns:
        Simulation time as a float. Returns ``nan`` if the snapshot is not
        found.
    """
    case_dir = Path(case_dir)
    snap_file = case_dir / f"time_{time_idx:03d}.h5"

    alfven_time = float("nan")
    if snap_file.exists():
        with h5py.File(snap_file, "r") as f:
            alfven_time = float(f.attrs.get("time", float("nan")))
    else:
        c1h5 = case_dir / "C1.h5"
        group_name = f"time_{time_idx:03d}"
        if c1h5.exists():
            with h5py.File(c1h5, "r") as f:
                if group_name in f:
                    alfven_time = float(f[group_name].attrs.get("time", float("nan")))

    if units in ("s", "mks"):
        try:
            from m3dc1.get_time_of_slice import get_time_of_slice
            return float(get_time_of_slice(
                time_idx,
                filename=str(case_dir / "C1.h5"),
                units="mks",
                quiet=True,
            ))
        except ImportError:
            warnings.warn(
                f"m3dc1 not installed; returning Alfvén time {alfven_time:.4g} "
                f"for time_idx={time_idx}",
                RuntimeWarning,
                stacklevel=2,
            )

    return alfven_time


def read_scalar_traces(
    case_dir: str | Path,
    names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read global time-trace scalars from ``C1.h5/scalars/``.

    Each scalar is a 1-D array of length ``ntimestep + 1`` (one entry per
    recorded timestep, including the initial state at t = 0). The ``"time"``
    scalar gives the simulation time in Alfvén times.

    Args:
        case_dir: Path to the M3D-C1 case directory.
        names:    List of scalar names to read, e.g. ``["E_K3", "time"]``.
                  If ``None``, all available scalars are returned.

    Returns:
        Dict mapping scalar name to a 1-D NumPy array.
    """
    c1h5 = Path(case_dir) / "C1.h5"
    result: dict[str, np.ndarray] = {}
    with h5py.File(c1h5, "r") as f:
        scalars = f["scalars"]
        keys = list(names) if names is not None else list(scalars.keys())
        for k in keys:
            if k in scalars:
                result[k] = scalars[k][:]
    return result


def read_case_metadata(case_dir: str | Path) -> dict:
    """Return a structured summary of a case's parameters and equilibrium state.

    Combines ``read_c1input``, ``list_time_snapshots``, and key equilibrium
    geometry scalars from ``C1.h5`` into a single dict. Useful as the first
    call when an agent encounters an unfamiliar case.

    Args:
        case_dir: Path to the M3D-C1 case directory.

    Returns:
        Dict with keys:

        ``"params"``
            Output of :func:`read_c1input`.
        ``"snapshots"``
            Output of :func:`list_time_snapshots`.
        ``"final_time"``
            Simulation time of the last snapshot in Alfvén times (``nan`` if
            no snapshots found).
        ``"R_mag"``
            R coordinate of the magnetic axis (metres).
        ``"Z_mag"``
            Z coordinate of the magnetic axis (metres).
        ``"R_xpoint"``
            R coordinate of the primary X-point (metres).
        ``"Z_xpoint"``
            Z coordinate of the primary X-point (metres).
        ``"psi_min"``
            ψ at the magnetic axis (internal units).
        ``"psi_lcfs"``
            ψ at the last closed flux surface (internal units).
    """
    case_dir = Path(case_dir)
    result: dict = {}
    result["params"] = read_c1input(case_dir)
    result["snapshots"] = list_time_snapshots(case_dir)

    if result["snapshots"]:
        result["final_time"] = read_snapshot_time(case_dir, result["snapshots"][-1])
    else:
        result["final_time"] = float("nan")

    c1h5 = case_dir / "C1.h5"
    if c1h5.exists():
        with h5py.File(c1h5, "r") as f:
            sc = f["scalars"]
            result["R_mag"] = float(sc["xmag"][0])
            result["Z_mag"] = float(sc["zmag"][0])
            result["R_xpoint"] = float(sc["xnull"][0])
            result["Z_xpoint"] = float(sc["znull"][0])
            result["psi_min"] = float(sc["psimin"][0])
            result["psi_lcfs"] = float(sc["psi_lcfs"][0])
    else:
        for key in ("R_mag", "Z_mag", "R_xpoint", "Z_xpoint", "psi_min", "psi_lcfs"):
            result[key] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Mesh and evaluation grid (no m3dc1 dependency)
# ---------------------------------------------------------------------------

def read_mesh_vertices(c1h5_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique mesh vertex (R, Z) coordinates from an M3D-C1 HDF5 file.

    M3D-C1 stores the reference vertex (R, Z) of each triangular finite element
    in columns 4 and 5 of ``mesh/elements``. Deduplicating these gives the set
    of unique vertex positions in the poloidal cross-section.

    Args:
        c1h5_path: Path to a ``C1.h5``, ``equilibrium.h5``, or
                   ``time_NNN.h5`` file.

    Returns:
        ``(R, Z)`` — a pair of 1-D float32 arrays, each of length
        ``n_unique_vertices``, sorted lexicographically by (R, Z).
    """
    path = Path(c1h5_path)
    with h5py.File(path, "r") as f:
        if "equilibrium/mesh/elements" in f:
            elements = f["equilibrium/mesh/elements"][:]
        elif "mesh/elements" in f:
            elements = f["mesh/elements"][:]
        else:
            raise KeyError(f"No mesh/elements group found in {c1h5_path}")

    rz = np.unique(np.c_[elements[:, 4], elements[:, 5]], axis=0)
    return rz[:, 0].astype(np.float32), rz[:, 1].astype(np.float32)


def make_evaluation_grid(
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    mode: str = "mesh",
    grid_res: int = 200,
    phi: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (R, Z, phi) evaluation point arrays for field interpolation.

    The returned arrays are suitable for passing directly to ``eval_field()``
    from the ``m3dc1`` library. Note that ``eval_field`` uses the argument
    order ``(field_name, R, phi, Z, ...)``, so pass ``phi_arr`` before ``Z``.

    Args:
        R_mesh:   1-D array of mesh vertex R coordinates (from
                  :func:`read_mesh_vertices`).
        Z_mesh:   1-D array of mesh vertex Z coordinates.
        mode:     ``"mesh"`` — use mesh vertices directly; R and Z are 1-D.
                  ``"grid"`` — build a regular ``grid_res × grid_res``
                  Cartesian grid spanning the mesh bounding box; R and Z are
                  2-D.
        grid_res: Number of points per axis for ``mode="grid"``.
        phi:      Toroidal angle in radians applied to all evaluation points.

    Returns:
        ``(R, Z, phi_arr)`` where all three arrays have the same shape:

        - ``mode="mesh"``: 1-D, length ``n_vertices``.
        - ``mode="grid"``: 2-D, shape ``(grid_res, grid_res)``.
    """
    if mode == "grid":
        r_lin = np.linspace(float(np.nanmin(R_mesh)), float(np.nanmax(R_mesh)), grid_res)
        z_lin = np.linspace(float(np.nanmin(Z_mesh)), float(np.nanmax(Z_mesh)), grid_res)
        R, Z = np.meshgrid(r_lin, z_lin)
    elif mode == "mesh":
        R = np.asarray(R_mesh)
        Z = np.asarray(Z_mesh)
    else:
        raise ValueError(f"mode must be 'mesh' or 'grid', got {mode!r}")

    phi_arr = np.full_like(R, phi, dtype=float)
    return R, Z, phi_arr


# ---------------------------------------------------------------------------
# Growth rate (h5py / numpy only — reads scalars directly from C1.h5)
# ---------------------------------------------------------------------------

def compute_ke_growth_trace(
    case_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total kinetic energy as a function of time from ``C1.h5/scalars/``.

    Uses the ``E_K3`` scalar (total 3-D kinetic energy). No m3dc1 dependency.

    Args:
        case_dir: Path to the M3D-C1 case directory.

    Returns:
        ``(time, ke)`` — two 1-D float arrays of length ``ntimestep + 1``.
        ``time`` is in Alfvén times; ``ke`` is in internal (normalised) units.
    """
    traces = read_scalar_traces(case_dir, names=["time", "E_K3"])
    return traces["time"], traces["E_K3"]


def compute_growth_rate(
    case_dir: str | Path,
    time_idx: int | None = None,
) -> float:
    """Estimate the linear growth rate from the kinetic energy time trace.

    Computes the mean instantaneous growth rate

        γ(t) = 0.5 · d(ln E_K3) / dt

    over the trace, optionally truncated at the timestep corresponding to
    snapshot ``time_idx``. Returns the mean of all instantaneous γ values,
    in units of 1/τ_A.

    Args:
        case_dir:  Path to the M3D-C1 case directory.
        time_idx:  Snapshot index (the NNN in ``time_NNN.h5``). If given, the
                   trace is truncated at the ``ntimestep`` value stored in that
                   snapshot's HDF5 attributes before computing the rate. If
                   ``None``, the full available trace is used.

    Returns:
        Mean growth rate in 1/τ_A. Returns ``nan`` if fewer than two non-zero
        kinetic energy values are found.
    """
    time, ke = compute_ke_growth_trace(case_dir)

    if time_idx is not None:
        snap_file = Path(case_dir) / f"time_{time_idx:03d}.h5"
        if snap_file.exists():
            with h5py.File(snap_file, "r") as f:
                ntimestep = f.attrs.get("ntimestep")
            if ntimestep is not None:
                mask = time <= float(ntimestep)
                time = time[mask]
                ke = ke[mask]

    nonzero = ke > 0
    if nonzero.sum() < 2:
        return float("nan")

    first = int(np.argmax(nonzero))
    ke_nz = ke[first:]
    time_nz = time[first:]
    dt = np.diff(time_nz)
    dt[dt == 0] = np.nan
    gamma = 0.5 * np.diff(np.log(ke_nz)) / dt
    return float(np.nanmean(gamma))


# ---------------------------------------------------------------------------
# Equilibrium profiles (requires m3dc1 + fpy)
# ---------------------------------------------------------------------------

def compute_flux_average_profiles(
    case_dir: str | Path,
    fields: list[str] | None = None,
    fcoords: str = "pest",
    points: int = 200,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute flux-surface-averaged radial profiles for equilibrium quantities.

    Requires the ``m3dc1`` and ``fpy`` libraries. Uses the equilibrium
    (``time=-1``) state from ``C1.h5``.

    Args:
        case_dir: Path to the M3D-C1 case directory.
        fields:   Field names to compute. Default: ``["p", "j", "ne", "q"]``.
                  Any name accepted by ``m3dc1.flux_average()`` is valid.
        fcoords:  Flux coordinate system: ``"pest"`` (default) or
                  ``"equal_arc"``.
        points:   Number of radial grid points in the output profiles.

    Returns:
        Dict mapping field name to ``(psi_norm, profile)`` — two 1-D float64
        arrays of length ``points``. ``psi_norm`` runs from 0 (magnetic axis)
        to 1 (LCFS). Fields that cannot be computed are omitted.
    """
    import fpy
    import m3dc1 as m1

    if fields is None:
        fields = ["p", "j", "ne", "q"]

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with _in_case_dir(case_dir):
        sim = fpy.sim_data(c1h5, time=-1)
        for field in fields:
            try:
                psin, profile = m1.flux_average(field, sim=sim, fcoords=fcoords, points=points)
                if psin is not None and profile is not None:
                    result[field] = (np.asarray(psin, dtype=float), np.asarray(profile, dtype=float))
            except Exception:
                pass

    return result


def compute_q95(psin: np.ndarray, q_profile: np.ndarray) -> float:
    """Interpolate the safety factor at normalised flux psi_norm = 0.95.

    Args:
        psin:      1-D array of normalised poloidal flux values spanning
                   0 (magnetic axis) to at least 0.95.
        q_profile: 1-D array of safety factor values at each ``psin`` point.

    Returns:
        q95 as a float. Returns ``nan`` if the profile does not cover
        psi_norm = 0.95.
    """
    psin = np.asarray(psin, dtype=float)
    q_profile = np.asarray(q_profile, dtype=float)
    if len(psin) == 0 or psin.max() < 0.95:
        return float("nan")
    return float(np.interp(0.95, psin, q_profile))


def compute_miller_geometry(
    case_dir: str | Path,
    res: int = 250,
) -> dict[str, float]:
    """Compute Miller geometry parameters of the last closed flux surface.

    Requires the ``m3dc1`` library. Uses ``m3dc1.get_shape()`` on the
    equilibrium state.

    Args:
        case_dir: Path to the M3D-C1 case directory.
        res:      Poloidal resolution used to trace the LCFS shape.

    Returns:
        Dict with keys ``"R0"`` (major radius, m), ``"a"`` (minor radius, m),
        ``"kappa"`` (elongation), ``"delta"`` (triangularity). Returns an
        empty dict if the computation fails.
    """
    import fpy
    import m3dc1 as m1

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")

    with _in_case_dir(case_dir):
        sim = fpy.sim_data(c1h5, time=-1)
        try:
            shape = m1.get_shape(sim, res=res)
            return {k: float(shape[k]) for k in ("R0", "a", "kappa", "delta")}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Perturbed fields (requires m3dc1 + fpy)
# ---------------------------------------------------------------------------

_DEFAULT_SKIP = frozenset({"alpha", "gradA", "kprad_rad"})
_VECTOR_FIELDS = frozenset({"B", "E", "A", "j", "v"})


def compute_perturbed_fields(
    case_dir: str | Path,
    time_idx: int,
    R: np.ndarray,
    Z: np.ndarray,
    phi: np.ndarray,
    fields: str | list[str] = "all",
    skip_fields: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate perturbed fields at given (R, Z, phi) points for one snapshot.

    Computes Δf = f(time_idx) − f(equilibrium) by calling ``eval_field()`` at
    both the equilibrium and the perturbed state and subtracting. Vector fields
    (B, E, A, j, v) are decomposed into R, PHI, and Z components stored under
    keys such as ``"BR"``, ``"BPHI"``, ``"BZ"``.

    Note: ``eval_field`` expects the argument order ``(name, R, phi, Z, ...)``,
    which is handled internally.

    Args:
        case_dir:    Path to the M3D-C1 case directory.
        time_idx:    Snapshot index to use for the perturbed state.
        R:           Array of R evaluation coordinates (any broadcastable shape).
        Z:           Array of Z evaluation coordinates (same shape as R).
        phi:         Array of toroidal angles in radians (same shape as R).
        fields:      ``"all"`` to compute every available field, or a list of
                     specific field names.
        skip_fields: Field names to exclude even when ``fields="all"``.
                     Default: ``{"alpha", "gradA", "kprad_rad"}``.

    Returns:
        Dict mapping field name → NumPy array with the same shape as R. Fields
        that fail evaluation are omitted (a warning is printed).
    """
    import fpy
    from m3dc1.eval_field import eval_field

    skip = set(skip_fields) if skip_fields is not None else set(_DEFAULT_SKIP)

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    sim_eq = fpy.sim_data(c1h5, time=-1)
    sim_lin = fpy.sim_data(c1h5, time=time_idx)

    available = set(sim_lin.typedict.keys())
    out: dict[str, np.ndarray] = {}

    def _add_scalar(name: str) -> None:
        try:
            eq_val = eval_field(name, R, phi, Z, coord="scalar",
                                sim=sim_eq, time=sim_eq.timeslice, quiet=True)
            lin_val = eval_field(name, R, phi, Z, coord="scalar",
                                 sim=sim_lin, time=sim_lin.timeslice, quiet=True)
            out[name] = lin_val - eq_val
        except Exception as exc:
            print(f"WARNING: skipping scalar '{name}': {exc}")

    def _add_vector(base: str) -> None:
        try:
            eq_vec = eval_field(base, R, phi, Z, coord="vector",
                                sim=sim_eq, time=sim_eq.timeslice, quiet=True)
            lin_vec = eval_field(base, R, phi, Z, coord="vector",
                                 sim=sim_lin, time=sim_lin.timeslice, quiet=True)
            delta = lin_vec - eq_vec
            tag = base.upper()
            out[f"{tag}R"] = delta[0]
            out[f"{tag}PHI"] = delta[1]
            out[f"{tag}Z"] = delta[2]
        except Exception as exc:
            print(f"WARNING: skipping vector '{base}': {exc}")

    if fields == "all":
        for name in sorted(available - _VECTOR_FIELDS):
            if name not in skip:
                _add_scalar(name)
        for base in sorted(_VECTOR_FIELDS & available):
            if base not in skip:
                _add_vector(base)
    else:
        for name in fields:
            if name in skip:
                continue
            if name in _VECTOR_FIELDS:
                _add_vector(name)
            else:
                _add_scalar(name)

    return out


# ---------------------------------------------------------------------------
# Spectral analysis (requires m3dc1 + fpy)
# ---------------------------------------------------------------------------

def compute_poloidal_spectrum(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    coord: str = "scalar",
    fcoords: str = "pest",
    points: int = 200,
    full_fft: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the poloidal mode spectrum of a field at a given time snapshot.

    Decomposes the field along the poloidal direction using
    ``m3dc1.eigenfunction()`` with Fourier=True. The result gives the amplitude
    of each poloidal harmonic m as a function of normalised flux psi_norm.

    Requires the ``m3dc1`` and ``fpy`` libraries.

    Args:
        case_dir:  Path to the M3D-C1 case directory.
        time_idx:  Snapshot index to analyse.
        field:     Field name: ``"p"`` (pressure), ``"B"`` (magnetic field), etc.
        coord:     Component: ``"scalar"`` for scalar fields; ``"R"``, ``"Z"``,
                   or ``"phi"`` for vector components.
        fcoords:   Flux coordinate system: ``"pest"`` (default) or
                   ``"equal_arc"``.
        points:    Radial resolution (number of psi_norm points).
        full_fft:  If ``True``, return the full two-sided FFT with all m modes.
                   If ``False`` (default), return a symmetric spectrum mirrored
                   to include both positive and negative m.

    Returns:
        ``(m_modes, psi_norm, spectrum)`` where:

        - ``m_modes``:  1-D int array of poloidal mode numbers.
        - ``psi_norm``: 1-D float array of length ``points``.
        - ``spectrum``: 2-D float array, shape ``(len(m_modes), points)``.
          ``spectrum[i, j]`` is the amplitude of mode ``m_modes[i]`` at
          ``psi_norm[j]``.
    """
    import fpy
    import m3dc1 as m1

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")

    with _in_case_dir(case_dir):
        sim_eq = fpy.sim_data(c1h5, time=-1)
        sim_lin = fpy.sim_data(c1h5, time=time_idx)

        sim_eq_fc = m1.flux_coordinates(
            sim=sim_eq, fcoords=fcoords, phit=0.0, points=points
        )
        psi_norm = np.asarray(sim_eq_fc.fc.psi_norm, dtype=float)

        spec = m1.eigenfunction(
            sim=[sim_eq_fc, sim_lin],
            field=field,
            coord=coord,
            fcoords=fcoords,
            points=points,
            makeplot=False,
            fourier=True,
            full_fft=full_fft,
            norm_to_unity=True,
            quiet=True,
        )

    spec = np.asarray(spec, dtype=float)

    if full_fft:
        m_modes = (np.fft.fftshift(np.fft.fftfreq(points, d=1.0)) * points).astype(int)
        return m_modes, psi_norm, spec

    m_max = spec.shape[0] - 1
    m_modes = np.arange(-m_max, m_max + 1, dtype=int)
    spec_full = np.concatenate([spec[1:][::-1], spec], axis=0)
    return m_modes, psi_norm, spec_full


_STANDARD_SPEC_MAP = {
    "p":    ("p", "scalar"),
    "br":   ("B", "R"),
    "bz":   ("B", "Z"),
    "bphi": ("B", "phi"),
}


def compute_standard_spectra(
    case_dir: str | Path,
    time_idx: int,
    fcoords: str = "pest",
    points: int = 200,
    full_fft: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute poloidal spectra for the standard set of fields: p, B_R, B_Z, B_phi.

    Calls :func:`compute_poloidal_spectrum` for each of the four default
    spectrum fields used in ``postprocess_ndarray.py``. Any field that fails is
    omitted from the result without raising an exception.

    Requires the ``m3dc1`` and ``fpy`` libraries.

    Args:
        case_dir:  Path to the M3D-C1 case directory.
        time_idx:  Snapshot index to analyse.
        fcoords:   Flux coordinate system.
        points:    Radial resolution.
        full_fft:  If ``True``, use the full two-sided FFT.

    Returns:
        Dict with keys ``"p"``, ``"br"``, ``"bz"``, ``"bphi"``. Each value is
        a ``(m_modes, psi_norm, spectrum)`` tuple as returned by
        :func:`compute_poloidal_spectrum`. Keys for fields that fail are absent.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, (field, coord) in _STANDARD_SPEC_MAP.items():
        try:
            result[key] = compute_poloidal_spectrum(
                case_dir, time_idx, field, coord=coord,
                fcoords=fcoords, points=points, full_fft=full_fft,
            )
        except Exception as exc:
            print(f"WARNING: spectrum failed for '{key}': {exc}")
    return result
