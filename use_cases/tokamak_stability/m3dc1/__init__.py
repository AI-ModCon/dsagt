"""m3dc1 — Python interface to M3D-C1 simulation data via fusion-io.

Public API
----------
eval_field(name, R, phi, Z, coord="scalar", sim=..., time=..., quiet=True)
get_time_of_slice(time_idx, filename="C1.h5", units="mks", quiet=True)
get_timetrace(name, sim=..., units="m3dc1", growth=False, renorm=False, quiet=True)
flux_average(field, sim=..., fcoords="pest", points=200)
get_shape(sim, res=250)
flux_coordinates(sim=..., fcoords="pest", phit=0.0, points=200)
eigenfunction(sim=[fc_obj, sim_lin], field="p", coord="scalar", ...)

Plotting (Priority 2) — require matplotlib:
plot_field, plot_shape, plot_mesh, plot_diagnostics, plot_signal,
plot_time_trace_fast, plot_flux_average, plot_line, plot_field_vs_phi,
plot_mag_probes, plot_gfile, tpf, run_trace, plot_poincare, poincare_movie

Submodule aliases (for `from m3dc1.eval_field import eval_field` etc.):
  m3dc1.eval_field, m3dc1.get_time_of_slice, m3dc1.get_timetrace
"""

import warnings
from pathlib import Path

import h5py
import numpy as np

from . import _neo_input as _ni

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_KE_ALIASES = {
    "ke": "E_K3",
    "ke_ion": "E_K3",
}

# Fields pre-computed by write_neo_input (map caller name → nc variable)
_NEO_FSA_FIELDS = {
    "q": "q",
    "ne": "ne0",
    "te": "Te0",
    "ni": "ni0",
    "ti": "Ti0",
}

# Mapping from typedict short name to ftype for fields we know about
_VECTOR_FIELDS = {"j", "v", "B", "A", "E", "gradA"}


def _ftype_of(field_name, sim):
    """Return 'vector' or 'scalar' for a field name."""
    if field_name in sim.typedict:
        return sim.typedict[field_name][1]
    return "scalar"


# ---------------------------------------------------------------------------
# eval_field
# ---------------------------------------------------------------------------

def eval_field(field_name, R, phi, Z, coord="scalar", sim=None, time=None,
               quiet=True):
    """Evaluate a fusion-io field at arbitrary (R, phi, Z) coordinates.

    Parameters
    ----------
    field_name : str
        Field name accepted by fpy.sim_data (e.g. 'psi', 'B', 'p').
    R, phi, Z : array_like
        Evaluation coordinates (cylindrical, SI). All must be broadcastable.
    coord : str
        'scalar' — return the scalar value (or magnitude for vector fields).
        'R', 'phi', 'Z' — return that cylindrical component (vector fields).
        'vector' — return (3, *shape) array of (R, phi, Z) components.
    sim : fpy.sim_data
        Open simulation object.
    time : int or None
        Timeslice to evaluate at.  If None, uses sim.timeslice.
    quiet : bool
        Suppress informational output.

    Returns
    -------
    np.ndarray
        Shape matches R (or (3, *R.shape) for coord='vector').
        Out-of-domain points are NaN.
    """
    if sim is None:
        raise ValueError("eval_field: sim must be provided")

    if time is not None:
        sim.set_timeslice(int(time))

    R = np.asarray(R, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Z = np.asarray(Z, dtype=float)
    R, phi, Z = np.broadcast_arrays(R, phi, Z)
    shape = R.shape
    R_flat = R.ravel()
    phi_flat = phi.ravel()
    Z_flat = Z.ravel()
    n = len(R_flat)

    ftype = _ftype_of(field_name, sim)
    fld = sim.get_field(field_name, time=None)

    if ftype == "vector" or coord in ("R", "phi", "Z", "vector"):
        out = np.full((3, n), np.nan)
        for i in range(n):
            result = fld.evaluate((float(R_flat[i]), float(phi_flat[i]),
                                   float(Z_flat[i])))
            if result[0] is not None:
                out[0, i] = result[0]
            if len(result) > 1 and result[1] is not None:
                out[1, i] = result[1]
            if len(result) > 2 and result[2] is not None:
                out[2, i] = result[2]

        if coord == "vector":
            return out.reshape((3,) + shape)
        comp_map = {"R": 0, "phi": 1, "Z": 2}
        if coord in comp_map:
            return out[comp_map[coord]].reshape(shape)
        # coord == 'scalar' for a vector field → return magnitude
        return np.sqrt(np.nansum(out ** 2, axis=0)).reshape(shape)

    else:
        out = np.full(n, np.nan)
        for i in range(n):
            result = fld.evaluate((float(R_flat[i]), float(phi_flat[i]),
                                   float(Z_flat[i])))
            if result[0] is not None:
                out[i] = float(result[0])
        return out.reshape(shape)


# ---------------------------------------------------------------------------
# get_time_of_slice
# ---------------------------------------------------------------------------

def get_time_of_slice(time_idx, filename="C1.h5", units="mks", quiet=True):
    """Return the simulation time of a snapshot.

    Parameters
    ----------
    time_idx : int
        Snapshot index (the NNN in time_NNN.h5).
    filename : str or Path
        Path to C1.h5.
    units : str
        'alfven' or 'm3dc1' — return Alfvén times (raw).
        'mks' or 's' — convert to physical seconds.
    quiet : bool
        Suppress warnings.

    Returns
    -------
    float
        Simulation time in the requested units.  Returns nan on failure.
    """
    try:
        with h5py.File(str(filename), "r") as f:
            key = f"time_{int(time_idx):03d}"
            if key not in f:
                if not quiet:
                    print(f"get_time_of_slice: group {key!r} not found in {filename}")
                return float("nan")
            t_alfven = float(f[key].attrs["time"])

            if units in ("alfven", "m3dc1"):
                return t_alfven

            # MKS conversion: tau_A = l0 / v_A  (compute in CGS, result in seconds)
            try:
                n0 = float(f.attrs["n0_norm"])     # cm^-3
                l0 = float(f.attrs["l0_norm"])     # cm
                b0 = float(f.attrs["b0_norm"])     # Gauss
                ion_mass = float(f.attrs["ion_mass"])  # proton masses
                m_p_cgs = 1.6726e-24  # grams
                v_A_cgs = b0 / np.sqrt(4 * np.pi * ion_mass * m_p_cgs * n0)
                tau_A = l0 / v_A_cgs  # seconds
                return t_alfven * tau_A
            except KeyError as exc:
                if not quiet:
                    warnings.warn(
                        f"get_time_of_slice: cannot compute MKS time "
                        f"(missing attr {exc}); returning Alfvén time.",
                        stacklevel=2,
                    )
                return t_alfven

    except Exception as exc:
        if not quiet:
            print(f"get_time_of_slice: error reading {filename}: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# get_timetrace
# ---------------------------------------------------------------------------

def get_timetrace(name, sim=None, units="m3dc1", growth=False, renorm=False,
                  quiet=True):
    """Read a scalar time trace from C1.h5.

    Parameters
    ----------
    name : str
        Scalar name ('ke' aliases to 'E_K3'; otherwise must be in C1.h5/scalars/).
    sim : fpy.sim_data
        Open simulation object.
    units : str
        Time units: 'm3dc1'/'alfven' (Alfvén times) or 'mks'/'s' (seconds).
    growth : bool
        If True, return instantaneous growth rate gamma = 0.5 * d(ln values)/dt
        instead of the raw values.
    renorm : bool
        If True, divide values by the first nonzero value before returning.
    quiet : bool
        Suppress output.

    Returns
    -------
    (time, values, label, units_str) : tuple of (ndarray, ndarray, str, str)
    """
    if sim is None:
        raise ValueError("get_timetrace: sim must be provided")

    raw_name = _KE_ALIASES.get(name.lower(), name)
    try:
        time = np.asarray(sim._all_traces["time"], dtype=float)
        values = np.asarray(sim._all_traces[raw_name], dtype=float)
    except KeyError:
        raise KeyError(f"get_timetrace: trace {raw_name!r} not found in C1.h5/scalars/")

    # Trim to equal length if needed
    n = min(len(time), len(values))
    time = time[:n]
    values = values[:n]

    # Remove NaNs
    valid = ~np.isnan(values)
    time = time[valid]
    values = values[valid]

    label = raw_name
    units_str = units

    if renorm:
        nonzero = values != 0
        if nonzero.any():
            values = values / values[nonzero][0]

    if growth:
        # gamma = 0.5 * d(ln |values|) / dt
        log_vals = np.log(np.abs(values))
        dt = np.diff(time)
        gamma = 0.5 * np.diff(log_vals) / np.where(dt != 0, dt, np.nan)
        time = 0.5 * (time[:-1] + time[1:])
        values = gamma
        label = f"gamma({raw_name})"
        units_str = "1/tau_A"

    if units in ("mks", "s") and not growth:
        # Scale time axis only; values are dimensionless or in code units
        # Try to get the Alfvén time from the file
        try:
            with h5py.File(sim.filename, "r") as f:
                n0 = float(f.attrs["n0_norm"])
                l0 = float(f.attrs["l0_norm"])
                b0 = float(f.attrs["b0_norm"])
                ion_mass_val = float(f.attrs["ion_mass"])
            m_p_cgs = 1.6726e-24
            v_A_cgs = b0 / np.sqrt(4 * np.pi * ion_mass_val * m_p_cgs * n0)
            tau_A = l0 / v_A_cgs
            time = time * tau_A
            units_str = "s"
        except Exception:
            if not quiet:
                warnings.warn("get_timetrace: cannot convert time to MKS; "
                              "returning Alfvén times.", stacklevel=2)

    return time, values, label, units_str


# ---------------------------------------------------------------------------
# flux_average
# ---------------------------------------------------------------------------

def flux_average(field, sim=None, fcoords="pest", points=200, ntheta=300, nphi=4):
    """Compute the flux-surface average of a field.

    Parameters
    ----------
    field : str
        Field name.  For 'q', 'ne', 'te', 'ni', 'ti' the pre-computed values
        from write_neo_input are returned directly.  All other fields are
        evaluated on the 3-D flux surface grid and averaged with the Jacobian.
    sim : fpy.sim_data
        Equilibrium simulation object.
    fcoords : str
        Flux coordinate system passed to write_neo_input (currently unused;
        reserved for future coordinate choices).
    points : int
        Number of radial grid points (nr).
    ntheta : int
        Poloidal grid points per surface for the FSA integration.
    nphi : int
        Toroidal grid points for the FSA integration.  Use nphi=1 for a
        strictly axisymmetric equilibrium (exact); nphi>=4 if the field
        has toroidal variation.

    Returns
    -------
    (psi_norm, profile) : (ndarray, ndarray), each of shape (nr,)
    """
    if sim is None:
        raise ValueError("flux_average: sim must be provided")

    c1h5 = Path(sim.filename)
    timeslice = sim.timeslice
    nc_path, tmpdir_obj = _ni.run_write_neo_input(
        c1h5, timeslice,
        psi_start=0.01, psi_end=0.99,
        nr=points, ntheta=ntheta, nphi=nphi,
    )
    try:
        neo = _ni.read_neo_input(nc_path)
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()

    psi_norm = neo["psi_norm"]

    # Direct lookup for pre-computed averages
    neo_key = _NEO_FSA_FIELDS.get(field.lower())
    if neo_key is not None and neo_key in neo:
        return psi_norm, neo[neo_key]

    # General case: evaluate field on 3-D grid and integrate
    R_grid = neo["R"]       # (ntheta, nphi, nr)
    Z_grid = neo["Z"]
    Jac = neo["Jac"]
    Phi = neo["Phi"]        # (nphi,) toroidal angles

    ntheta, nphi, nr = R_grid.shape
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

    # Evaluate field at every grid point
    R_flat = R_grid.ravel()
    Z_flat = Z_grid.ravel()
    phi_3d = np.broadcast_to(
        Phi[np.newaxis, :, np.newaxis], (ntheta, nphi, nr)
    ).copy()
    phi_flat = phi_3d.ravel()

    f_flat = eval_field(field, R_flat, phi_flat, Z_flat,
                        coord="scalar", sim=sim, time=timeslice, quiet=True)
    f_grid = f_flat.reshape(ntheta, nphi, nr)

    profile = np.empty(nr)
    for s in range(nr):
        profile[s] = _ni.flux_surface_average(
            f_grid[:, :, s], Jac[:, :, s], theta, Phi
        )

    return psi_norm, profile


# ---------------------------------------------------------------------------
# get_shape
# ---------------------------------------------------------------------------

def get_shape(sim, res=250):
    """Compute Miller geometry parameters of the last closed flux surface.

    Parameters
    ----------
    sim : fpy.sim_data
        Equilibrium simulation object.
    res : int
        Grid resolution per axis for psi evaluation (res × res points).

    Returns
    -------
    dict with keys 'R0', 'a', 'kappa', 'delta' (all in metres).
    Returns {} on failure.
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        # Read LCFS psi value
        scalars = sim._all_attrs["scalars"]
        psi_lcfs = float(np.asarray(scalars["psi_lcfs"])[0])

        # Get mesh bounding box
        mesh = sim.get_mesh(quiet=True)
        R_mesh = mesh.elements[:, 4]
        Z_mesh = mesh.elements[:, 5]
        R_min, R_max = R_mesh.min(), R_mesh.max()
        Z_min, Z_max = Z_mesh.min(), Z_mesh.max()

        # Regular grid for psi evaluation
        R_1d = np.linspace(R_min, R_max, res)
        Z_1d = np.linspace(Z_min, Z_max, res)
        R_2d, Z_2d = np.meshgrid(R_1d, Z_1d)
        phi_2d = np.zeros_like(R_2d)

        psi_2d = eval_field("psi", R_2d, phi_2d, Z_2d,
                            coord="scalar", sim=sim,
                            time=sim.timeslice, quiet=True)

        # Extract LCFS contour (collect paths before closing figure)
        fig, ax = plt.subplots()
        cs = ax.contour(R_2d, Z_2d, psi_2d, levels=[psi_lcfs])
        try:
            paths = list(cs.get_paths())          # matplotlib >= 3.8
        except AttributeError:
            paths = []
            for coll in cs.collections:           # matplotlib < 3.8
                paths.extend(coll.get_paths())
        plt.close(fig)
        if not paths:
            return {}
        longest = max(paths, key=lambda p: len(p.vertices))
        R_lcfs = longest.vertices[:, 0]
        Z_lcfs = longest.vertices[:, 1]

        R0 = (R_lcfs.max() + R_lcfs.min()) / 2.0
        a = (R_lcfs.max() - R_lcfs.min()) / 2.0
        kappa = (Z_lcfs.max() - Z_lcfs.min()) / (2.0 * a)
        R_at_Zmax = R_lcfs[np.argmax(Z_lcfs)]
        delta = (R0 - R_at_Zmax) / a

        return {"R0": float(R0), "a": float(a),
                "kappa": float(kappa), "delta": float(delta)}

    except Exception as exc:
        warnings.warn(f"get_shape failed: {exc}", stacklevel=2)
        return {}


# ---------------------------------------------------------------------------
# flux_coordinates
# ---------------------------------------------------------------------------

class _FCSummary:
    """Holds the flux-coordinate summary accessible via flux_coordinates().fc."""
    def __init__(self, neo_data):
        self.psi_norm = neo_data["psi_norm"]   # (nr,)
        self.q = neo_data["q"]                 # (nr,)
        self.rpath = neo_data["R"][:, 0, :]   # (ntheta, nr) at phi plane 0
        self.zpath = neo_data["Z"][:, 0, :]   # (ntheta, nr)


class _FluxCoordsResult:
    """Return value of flux_coordinates().

    Attributes
    ----------
    fc : _FCSummary
        Exposes .psi_norm, .q, .rpath, .zpath
    """
    def __init__(self, neo_data, sim, phit, tmpdir_obj):
        self.fc = _FCSummary(neo_data)
        self._neo = neo_data
        self._sim = sim
        self._phit = float(phit)
        self._tmpdir_obj = tmpdir_obj   # keep temp dir alive


def flux_coordinates(sim=None, fcoords="pest", phit=0.0, points=200, ntheta=128):
    """Compute flux coordinates using write_neo_input.

    Parameters
    ----------
    sim : fpy.sim_data
        Equilibrium simulation object.
    fcoords : str
        Flux coordinate system (reserved; currently 'pest' only).
    phit : float
        Toroidal angle in radians for the poloidal cross-section.
    points : int
        Number of radial grid points (nr).
    ntheta : int
        Number of poloidal grid points per flux surface.  Controls the highest
        resolvable poloidal mode number (max_m = ntheta//2) and the cost of
        subsequent eigenfunction() calls.  128 resolves up to m=64, which
        exceeds typical M3D-C1 mesh resolution.

    Returns
    -------
    _FluxCoordsResult
        Object with .fc.psi_norm, passed as sim[0] to eigenfunction().
    """
    if sim is None:
        raise ValueError("flux_coordinates: sim must be provided")

    c1h5 = Path(sim.filename)
    timeslice = sim.timeslice
    nc_path, tmpdir_obj = _ni.run_write_neo_input(
        c1h5, timeslice,
        psi_start=0.01, psi_end=0.99,
        nr=points, ntheta=ntheta, nphi=1,
    )
    neo = _ni.read_neo_input(nc_path)
    return _FluxCoordsResult(neo, sim, phit, tmpdir_obj)


# ---------------------------------------------------------------------------
# eigenfunction
# ---------------------------------------------------------------------------

def eigenfunction(sim=None, field="p", coord="scalar", fcoords="pest",
                  points=200, makeplot=False, fourier=True, full_fft=False,
                  norm_to_unity=True, quiet=True):
    """Compute the poloidal mode spectrum of a perturbed field.

    Parameters
    ----------
    sim : [_FluxCoordsResult, fpy.sim_data]
        sim[0] — flux coordinate result from flux_coordinates().
        sim[1] — perturbed simulation object (linear timeslice).
    field : str
        Field to decompose (e.g. 'p', 'B').
    coord : str
        'scalar', 'R', 'phi', or 'Z'.
    fcoords : str
        Flux coordinate system (reserved).
    points : int
        Number of radial points (inherited from flux_coordinates).
    makeplot : bool
        If True, plot amplitude vs psi_norm.
    fourier : bool
        If True, return Fourier spectrum in poloidal mode number.
        If False, return the raw (theta, nr) field.
    full_fft : bool
        If True, return full two-sided FFT (using np.fft.fft + fftshift).
        If False, return one-sided amplitude (using np.fft.rfft).
    norm_to_unity : bool
        If True, divide the spectrum by its global maximum.
    quiet : bool
        Suppress output.

    Returns
    -------
    np.ndarray
        Shape (ntheta//2 + 1, nr) for fourier=True, full_fft=False.
        Shape (ntheta, nr)         for fourier=True, full_fft=True.
        Shape (ntheta, nr)         for fourier=False.
    """
    if sim is None or not hasattr(sim, "__len__") or len(sim) < 2:
        raise ValueError(
            "eigenfunction: sim must be [flux_coords_result, sim_data_linear]"
        )

    fc_result = sim[0]
    sim_lin = sim[1]

    # Poloidal cross-section at the stored phi plane (index 0 in neo nphi axis)
    R_grid = fc_result._neo["R"][:, 0, :]   # (ntheta, nr)
    Z_grid = fc_result._neo["Z"][:, 0, :]   # (ntheta, nr)
    ntheta, nr = R_grid.shape
    phi_arr = np.full_like(R_grid, fc_result._phit)

    f_flat = eval_field(
        field,
        R_grid.ravel(), phi_arr.ravel(), Z_grid.ravel(),
        coord=coord, sim=sim_lin,
        time=sim_lin.timeslice, quiet=quiet,
    )
    f_grid = f_flat.reshape(ntheta, nr)

    if not fourier:
        spec = f_grid
    elif full_fft:
        spec = np.abs(np.fft.fftshift(np.fft.fft(f_grid, axis=0), axes=0))
    else:
        spec = np.abs(np.fft.rfft(f_grid, axis=0))

    if norm_to_unity:
        peak = spec.max()
        if peak > 0:
            spec = spec / peak

    if makeplot:
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
            psin = fc_result.fc.psi_norm
            n_modes = spec.shape[0]
            fig, ax = plt.subplots()
            for m in range(min(n_modes, 20)):
                ax.plot(psin, spec[m, :], label=f"m={m}")
            ax.set_xlabel(r"$\psi_\mathrm{norm}$")
            ax.set_ylabel("Mode amplitude")
            ax.set_title(f"Eigenfunction: {field} ({coord})")
            ax.legend(fontsize="small", ncol=2)
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            if not quiet:
                warnings.warn(f"eigenfunction makeplot failed: {exc}", stacklevel=2)

    return spec


# ---------------------------------------------------------------------------
# Priority 2 — Plotting functions
# ---------------------------------------------------------------------------

def _require_sim(filename, time, filetype="m3dc1"):
    """Open a sim_data object from a filename."""
    try:
        import sys
        import os
        fio_lib = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        if fio_lib not in sys.path:
            sys.path.insert(0, fio_lib)
        from fpy import sim_data  # noqa: PLC0415
    except ImportError:
        raise ImportError("fpy.sim_data not importable; check your fusion-io installation.")
    return sim_data(filename=str(filename), filetype=filetype, time=time)


def plot_field(field, filename, time=0, coord="scalar", phi=0.0, points=250,
               tor_av=1, units="mks", mesh=False, bound=False, lcfs=False,
               coils=False, save=False, savedir="./", quiet=True):
    """Plot a field on a regular (R, Z) grid as a filled contour map.

    Evaluates *field* on a *points* × *points* grid at fixed toroidal angle
    *phi*.  If *tor_av* > 1, averages over that many equally-spaced phi values.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, time)
    mesh_obj = sim.get_mesh(quiet=True)
    R_mesh = mesh_obj.elements[:, 4]
    Z_mesh = mesh_obj.elements[:, 5]
    R_1d = np.linspace(R_mesh.min(), R_mesh.max(), points)
    Z_1d = np.linspace(Z_mesh.min(), Z_mesh.max(), points)
    R_2d, Z_2d = np.meshgrid(R_1d, Z_1d)

    if tor_av > 1:
        phis = np.linspace(0, 2 * np.pi, tor_av, endpoint=False)
        f_sum = np.zeros_like(R_2d)
        for p in phis:
            f_sum += eval_field(field, R_2d, np.full_like(R_2d, p), Z_2d,
                                coord=coord, sim=sim, time=time, quiet=True)
        f_2d = f_sum / tor_av
    else:
        f_2d = eval_field(field, R_2d, np.full_like(R_2d, phi), Z_2d,
                          coord=coord, sim=sim, time=time, quiet=True)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(R_2d, Z_2d, f_2d, shading="auto")
    plt.colorbar(pcm, ax=ax, label=field)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")

    if lcfs:
        try:
            psi_lcfs = float(np.asarray(sim._all_attrs["scalars"]["psi_lcfs"])[0])
            psi_2d = eval_field("psi", R_2d, np.zeros_like(R_2d), Z_2d,
                                coord="scalar", sim=sim, time=time, quiet=True)
            ax.contour(R_2d, Z_2d, psi_2d, levels=[psi_lcfs], colors="w")
        except Exception:
            pass

    ax.set_title(f"{field}  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"{field}_{time:03d}.png", dpi=150)
    plt.show()


def plot_shape(filename, time=0, points=200, save=False, savedir="./",
               quiet=True):
    """Plot psi contours (flux surfaces)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, time)
    mesh_obj = sim.get_mesh(quiet=True)
    R_mesh = mesh_obj.elements[:, 4]
    Z_mesh = mesh_obj.elements[:, 5]
    R_1d = np.linspace(R_mesh.min(), R_mesh.max(), points)
    Z_1d = np.linspace(Z_mesh.min(), Z_mesh.max(), points)
    R_2d, Z_2d = np.meshgrid(R_1d, Z_1d)
    psi_2d = eval_field("psi", R_2d, np.zeros_like(R_2d), Z_2d,
                        coord="scalar", sim=sim, time=time, quiet=True)

    fig, ax = plt.subplots()
    ax.contour(R_2d, Z_2d, psi_2d, levels=20)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title(f"Flux surfaces  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"shape_{time:03d}.png", dpi=150)
    plt.show()


def plot_mesh(filename, time=0, boundary=False, save=False, savedir="./",
              quiet=True):
    """Plot the M3D-C1 triangular mesh."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.tri as tri  # noqa: PLC0415

    sim = _require_sim(filename, time)
    mesh_obj = sim.get_mesh(quiet=True)
    el = mesh_obj.elements
    R = el[:, 4]
    Z = el[:, 5]
    # Element connectivity: columns 0-2 are node indices
    triang = tri.Triangulation(R, Z, el[:, :3].astype(int))

    fig, ax = plt.subplots()
    ax.triplot(triang, lw=0.3, color="k")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title("Mesh")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / "mesh.png", dpi=150)
    plt.show()


def plot_diagnostics(filename, save=False, savedir="./", quiet=True):
    """Plot M3D-C1 iteration counts and timing diagnostics."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, 0)
    try:
        diag_iter = sim.get_diagnostic("iterations")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(diag_iter.x_axis, diag_iter.diagnostic[:, 0])
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Iterations")
        axes[0].set_title("GMRES iterations")
    except Exception:
        fig, axes = plt.subplots(1, 1)
        axes = [axes]

    try:
        diag_time = sim.get_diagnostic("timings")
        total = diag_time.diagnostic.get("t_onestep", np.array([]))
        if len(total):
            axes[-1].plot(diag_time.x_axis, total)
            axes[-1].set_xlabel("Step")
            axes[-1].set_ylabel("Wall time (s)")
            axes[-1].set_title("Step timing")
    except Exception:
        pass

    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / "diagnostics.png", dpi=150)
    plt.show()


def plot_signal(filename, signame, save=False, savedir="./", quiet=True):
    """Plot a diagnostic signal (probe/flux loop) time trace."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, 0)
    sig = sim.get_signal(signame)
    fig, ax = plt.subplots()
    time = np.arange(len(sig.sigvalues))
    ax.plot(time, sig.sigvalues)
    ax.set_xlabel("Time step")
    ax.set_ylabel(signame)
    ax.set_title(signame)
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"signal_{signame}.png", dpi=150)
    plt.show()


def plot_time_trace_fast(trace, filename, units="m3dc1", save=False,
                         savedir="./", quiet=True):
    """Plot log(E_K3) vs time, optionally with fitted growth rate."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, 0)
    time, values, label, units_str = get_timetrace(
        trace, sim=sim, units=units, quiet=quiet
    )
    fig, ax = plt.subplots()
    ax.semilogy(time, np.abs(values))
    ax.set_xlabel(f"t ({units_str})")
    ax.set_ylabel(label)
    ax.set_title(f"{label} time trace")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"timetrace_{trace}.png", dpi=150)
    plt.show()


def plot_flux_average(field, filename, time=0, fcoords="pest", points=200,
                      save=False, savedir="./", quiet=True):
    """Plot the flux-surface average of a field vs psi_norm."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, time)
    psin, profile = flux_average(field, sim=sim, fcoords=fcoords, points=points)
    fig, ax = plt.subplots()
    ax.plot(psin, profile)
    ax.set_xlabel(r"$\psi_\mathrm{norm}$")
    ax.set_ylabel(field)
    ax.set_title(f"FSA of {field}  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"fsa_{field}_{time:03d}.png", dpi=150)
    plt.show()


def plot_line(field, filename, time=0, angle=0.0, Zoff=0.0, coord="scalar",
              dist_from_magax=False, points=200, save=False, savedir="./",
              quiet=True):
    """Plot a field along a radial line in (R, Z) at fixed phi=0."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, time)
    mesh_obj = sim.get_mesh(quiet=True)
    R_max = mesh_obj.elements[:, 4].max()
    R_min = mesh_obj.elements[:, 4].min()
    theta_rad = np.deg2rad(angle)
    R_line = np.linspace(R_min, R_max, points)
    Z_line = np.full_like(R_line, Zoff) + R_line * np.tan(theta_rad)
    phi_line = np.zeros_like(R_line)
    f_line = eval_field(field, R_line, phi_line, Z_line,
                        coord=coord, sim=sim, time=time, quiet=True)

    x_axis = R_line
    xlabel = "R (m)"
    if dist_from_magax:
        try:
            R_mag = float(np.asarray(sim._all_attrs["scalars"]["xmag"])[0])
            x_axis = np.abs(R_line - R_mag)
            xlabel = "|R - R_mag| (m)"
        except Exception:
            pass

    fig, ax = plt.subplots()
    ax.plot(x_axis, f_line)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{field} ({coord})")
    ax.set_title(f"Radial profile: {field}  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"line_{field}_{time:03d}.png", dpi=150)
    plt.show()


def plot_field_vs_phi(field, filename, time=0, R=None, Z=None, phi_res=64,
                      coord="scalar", save=False, savedir="./", quiet=True):
    """Plot a field at fixed (R, Z) as a function of toroidal angle phi."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = _require_sim(filename, time)
    if R is None or Z is None:
        R_mag = float(np.asarray(sim._all_attrs["scalars"]["xmag"])[0])
        Z_mag = float(np.asarray(sim._all_attrs["scalars"]["zmag"])[0])
        R = R_mag
        Z = Z_mag
    phis = np.linspace(0, 2 * np.pi, phi_res, endpoint=False)
    R_arr = np.full_like(phis, R)
    Z_arr = np.full_like(phis, Z)
    f_arr = eval_field(field, R_arr, phis, Z_arr,
                       coord=coord, sim=sim, time=time, quiet=True)
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(phis), f_arr)
    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel(f"{field} ({coord})")
    ax.set_title(f"{field} vs phi at R={R:.3f}, Z={Z:.3f}  t={time}")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / f"vs_phi_{field}_{time:03d}.png", dpi=150)
    plt.show()


def tpf(field, sim, filename, units="m3dc1", res=64):
    """Return (time_value, toroidal_peaking_factor) for the midplane.

    The toroidal peaking factor is max(|f|) / mean(|f|) evaluated toroidally
    at the outboard midplane.
    """
    R_mag = float(np.asarray(sim._all_attrs["scalars"]["xmag"])[0])
    Z_mag = float(np.asarray(sim._all_attrs["scalars"]["zmag"])[0])
    phis = np.linspace(0, 2 * np.pi, res, endpoint=False)
    R_arr = np.full_like(phis, R_mag)
    Z_arr = np.full_like(phis, Z_mag)
    f_arr = eval_field(field, R_arr, phis, Z_arr,
                       coord="scalar", sim=sim, time=sim.timeslice, quiet=True)
    f_abs = np.abs(f_arr)
    mean_f = f_abs.mean()
    pf = f_abs.max() / mean_f if mean_f > 0 else float("nan")

    time_arr = np.asarray(sim._all_traces["time"])
    t = float(time_arr[sim.timeslice]) if sim.timeslice < len(time_arr) else float("nan")
    return t, pf


def run_trace(filename, time=0, nparticles=1000, nsteps=500, verbose=False):
    """Run the fusion-io field-line tracer and return the output path."""
    import os as _os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess as sp  # noqa: PLC0415

    exe = shutil.which("trace")
    fio_dir = _os.environ.get("FIO_INSTALL_DIR")
    if exe is None and fio_dir:
        candidate = Path(fio_dir) / "bin" / "trace"
        if candidate.is_file():
            exe = str(candidate)
    if exe is None:
        raise RuntimeError(
            "trace executable not found. Set $FIO_INSTALL_DIR or add trace to $PATH."
        )
    cmd = [exe, "-m3dc1", str(filename), str(time),
           "-nparticles", str(nparticles), "-nsteps", str(nsteps)]
    sp.run(cmd, check=True, capture_output=not verbose)
    return Path("poincare.dat")


def plot_poincare(poincare_file="poincare.dat", save=False, savedir="./"):
    """Plot a Poincaré section from a trace output file."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    data = np.loadtxt(str(poincare_file))
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=0.5, c="k")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title("Poincaré section")
    plt.tight_layout()
    if save:
        plt.savefig(Path(savedir) / "poincare.png", dpi=150)
    plt.show()


def poincare_movie(filename, time_indices=None, nparticles=500, nsteps=300,
                   savedir="./poincare_frames/", verbose=False):
    """Generate Poincaré sections for a sequence of timeslices."""
    import os as _os  # noqa: PLC0415
    _os.makedirs(savedir, exist_ok=True)
    with h5py.File(str(filename), "r") as f:
        ntime = int(f.attrs.get("ntime", 0))
    if time_indices is None:
        time_indices = list(range(ntime))
    for t in time_indices:
        poincare_path = run_trace(filename, time=t, nparticles=nparticles,
                                  nsteps=nsteps, verbose=verbose)
        plot_poincare(poincare_path, save=True, savedir=savedir)


def plot_mag_probes(filename, save=False, savedir="./", quiet=True):
    """Plot magnetic probe (R, Z) positions from coil.dat."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    coil_dat = Path(filename).parent / "coil.dat"
    if not coil_dat.exists():
        warnings.warn(f"plot_mag_probes: {coil_dat} not found", stacklevel=2)
        return
    R_list, Z_list = [], []
    with open(coil_dat) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    R_list.append(float(parts[0]))
                    Z_list.append(float(parts[1]))
                except ValueError:
                    pass
    fig, ax = plt.subplots()
    ax.scatter(R_list, Z_list, marker="x")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title("Magnetic probe positions")
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / "mag_probes.png", dpi=150)
    plt.show()


def _parse_geqdsk_grid(filename):
    """Parse an EFIT GEQDSK file and return a dict of equilibrium arrays."""
    with open(str(filename)) as fh:
        header = fh.readline()
        parts = header.split()
        nw = int(parts[-2])
        nh = int(parts[-1])

        def _read_floats(n):
            vals = []
            while len(vals) < n:
                line = fh.readline()
                vals.extend([float(x) for x in line.split()])
            return np.array(vals[:n])

        efit_data = {}
        scalars = _read_floats(20)
        keys = ["rdim", "zdim", "rcentr", "rleft", "zmid",
                "rmaxis", "zmaxis", "simag", "sibry", "bcentr",
                "current", "simag2", "xdum", "rmaxis2", "xdum2",
                "zmaxis2", "xdum3", "sibry2", "xdum4", "xdum5"]
        for k, v in zip(keys, scalars):
            efit_data[k] = v

        efit_data["fpol"] = _read_floats(nw)
        efit_data["pres"] = _read_floats(nw)
        efit_data["ffprim"] = _read_floats(nw)
        efit_data["pprime"] = _read_floats(nw)
        efit_data["psirz"] = _read_floats(nw * nh).reshape(nh, nw)
        efit_data["qpsi"] = _read_floats(nw)
        nbbbs = int(fh.readline().split()[0])
        bbbs = _read_floats(2 * nbbbs)
        efit_data["rbbbs"] = bbbs[::2]
        efit_data["zbbbs"] = bbbs[1::2]
        efit_data["nw"] = nw
        efit_data["nh"] = nh
        efit_data["rdim"] = efit_data["rdim"]
        return efit_data


def plot_gfile(gfile_path, save=False, savedir="./", quiet=True):
    """Parse a GEQDSK file and plot psi contours and LCFS."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    try:
        g = _parse_geqdsk_grid(gfile_path)
    except Exception as exc:
        warnings.warn(f"plot_gfile: could not parse {gfile_path}: {exc}", stacklevel=2)
        return

    nw, nh = g["nw"], g["nh"]
    rdim, zdim = g["rdim"], g["zdim"]
    rleft, zmid = g["rleft"], g["zmid"]
    R_1d = np.linspace(rleft, rleft + rdim, nw)
    Z_1d = np.linspace(zmid - zdim / 2, zmid + zdim / 2, nh)
    R_2d, Z_2d = np.meshgrid(R_1d, Z_1d)

    fig, ax = plt.subplots()
    ax.contour(R_2d, Z_2d, g["psirz"], levels=20)
    ax.plot(g["rbbbs"], g["zbbbs"], "r-", lw=2, label="LCFS")
    ax.plot(g["rmaxis"], g["zmaxis"], "r+", ms=10)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.set_title(str(gfile_path))
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(Path(savedir) / "gfile.png", dpi=150)
    plt.show()


def plot_vector_field(field, filename, time=0, points=30, phi=0.0,
                      quiet=True):
    """Plot a vector field using mayavi (requires mayavi installation)."""
    try:
        from mayavi import mlab  # noqa: F401, PLC0415
    except ImportError:
        raise ImportError(
            "plot_vector_field requires mayavi. "
            "Install it with: conda install -c conda-forge mayavi"
        )
    raise NotImplementedError(
        "plot_vector_field mayavi rendering is not yet implemented."
    )


# ---------------------------------------------------------------------------
# Public API list
# ---------------------------------------------------------------------------

__all__ = [
    "eval_field",
    "get_time_of_slice",
    "get_timetrace",
    "flux_average",
    "get_shape",
    "flux_coordinates",
    "eigenfunction",
    "plot_field",
    "plot_shape",
    "plot_mesh",
    "plot_diagnostics",
    "plot_signal",
    "plot_time_trace_fast",
    "plot_flux_average",
    "plot_line",
    "plot_field_vs_phi",
    "tpf",
    "run_trace",
    "plot_poincare",
    "poincare_movie",
    "plot_mag_probes",
    "plot_gfile",
    "plot_vector_field",
]
