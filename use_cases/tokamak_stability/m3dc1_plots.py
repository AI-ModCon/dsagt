"""Plotting library for M3D-C1 simulation output.

Each public function produces exactly one figure, saves it to a caller-specified
path, closes the figure, and returns the output Path.  No plt.show() calls are
made; the module is headless-safe.

Functions
---------
Category A — pure matplotlib, data via m3dc1_tools.py (no m3dc1 plotting):
    plot_kinetic_energy          Semi-log KE vs time with optional γ annotation
    plot_growth_rate_vs_time     Instantaneous γτ_A(t) trace
    plot_flux_average_profiles   Multi-panel radial flux-averaged profiles
    plot_safety_factor           q(ψ_norm) with reference lines and q95
    plot_poloidal_spectrum       2-D heatmap of m-spectrum vs ψ_norm
    plot_standard_spectra        4-panel spectra: p, B_R, B_Z, B_φ
    plot_perturbed_field_map     δf filled contour on R,Z plane
    plot_geqdsk                  Flux contours from a GEQDSK file
    plot_geqdsk_compare          3-panel GEQDSK vs C1.h5 psi comparison
    plot_tpf_vs_time             Total poloidal flux trace over time snapshots

Category B — m3dc1 library wrappers (figure capture / tempdir pattern):
    plot_field                   2-D field contour via m3.plot_field
    plot_mesh                    Triangular mesh via m3.plot_mesh
    plot_flux_surface_shape      Flux surface shape via m3.plot_shape
    plot_field_basic             Field via m3dc1.plot_field_basic
    plot_field_mesh              Field on mesh via m3dc1.plot_field_mesh
    plot_field_vs_phi            Field vs φ at fixed R or Z
    plot_flux_average_m3         Flux average via m3.plot_flux_average
    plot_line                    Field along a line via m3.plot_line
    plot_eigenfunction           Eigenfunction via m3.eigenfunction
    plot_poincare                Poincaré plot via m3.plot_poincare
    plot_diagnostics             Diagnostic timing via m3.plot_diagnostics
    plot_signal                  Diagnostic signal via m3.plot_signal
    plot_time_trace              Time trace via m3.plot_time_trace_fast

Convenience wrappers:
    plot_stability_summary       KE + growth rate + spectra + field map
    plot_equilibrium_overview    Profiles + q + mesh
    plot_case_summary            Calls both wrappers above
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

# Set a headless default before importing pyplot; respected if user has already
# set MPLBACKEND or imported matplotlib with a different backend.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from m3dc1_tools import (
    compute_flux_average_profiles,
    compute_growth_rate,
    compute_ke_growth_trace,
    compute_perturbed_fields,
    compute_poloidal_spectrum,
    compute_q95,
    compute_standard_spectra,
    make_evaluation_grid,
    read_case_metadata,
    read_mesh_vertices,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _in_case_dir(path: Path):
    """Temporarily change CWD to path; restore on exit even if an exception is raised."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _import_m3dc1():
    """Return the m3dc1 module or raise ImportError with a helpful message."""
    try:
        import m3dc1 as m3
        return m3
    except ImportError as exc:
        raise ImportError(
            "m3dc1 Python package not found. Add its location to sys.path "
            "before calling Category B plot functions."
        ) from exc


def _save_first_new_fig(before: set, output_path: Path, dpi: int) -> None:
    """Save the first figure created since `before` was recorded; close all new figures."""
    after = set(plt.get_fignums())
    new = sorted(after - before)
    if not new:
        raise RuntimeError("m3dc1 library call produced no new figure")
    fig = plt.figure(new[0])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    for n in new:
        plt.close(plt.figure(n))


def _parse_geqdsk(gfile_path: Path) -> dict:
    """Parse a GEQDSK equilibrium file; return a dict of grid arrays and scalars."""
    number_re = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")
    with open(gfile_path) as f:
        header = f.readline()
        parts = header.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid GEQDSK header in {gfile_path}")
        nw, nh = int(parts[-2]), int(parts[-1])
        nums = [float(x) for line in f for x in number_re.findall(line)]

    idx = 0

    def take(n: int) -> list:
        nonlocal idx
        out = nums[idx: idx + n]
        idx += n
        return out

    rdim, zdim, _rcentr, rleft, zmid = take(5)
    rmaxis, zmaxis, simag, sibry, _bcentr = take(5)
    current = take(5)[0]
    take(nw)  # fpol
    take(nw)  # pres
    take(nw)  # ffprim
    take(nw)  # pprim
    psirz_flat = take(nw * nh)
    take(nw)  # qpsi

    _nbbbs_limitr = take(2)
    nbbbs, limitr = int(_nbbbs_limitr[0]), int(_nbbbs_limitr[1])

    rbbbs, zbbbs, rlim, zlim = [], [], [], []
    if nbbbs > 0:
        bdry = take(2 * nbbbs)
        rbbbs, zbbbs = bdry[0::2], bdry[1::2]
    if limitr > 0:
        lim = take(2 * limitr)
        rlim, zlim = lim[0::2], lim[1::2]

    rg = np.linspace(rleft, rleft + rdim, nw)
    zg = np.linspace(zmid - zdim / 2.0, zmid + zdim / 2.0, nh)
    psirz = np.reshape(psirz_flat, (nh, nw))
    psirzn = (psirz - simag) / (sibry - simag) if sibry != simag else np.zeros_like(psirz)

    return {
        "nw": nw, "nh": nh,
        "rg": rg, "zg": zg,
        "psirz": psirz, "psirzn": psirzn,
        "simag": simag, "sibry": sibry,
        "rmaxis": rmaxis, "zmaxis": zmaxis,
        "current": current,
        "rbbbs": rbbbs, "zbbbs": zbbbs,
        "rlim": rlim, "zlim": zlim,
    }


# ============================================================================
# CATEGORY A — PURE MATPLOTLIB
# Data is loaded and computed via m3dc1_tools.py functions.
# No m3dc1 library plotting routines are called in this section.
# ============================================================================

def plot_kinetic_energy(
    case_dir: str | Path,
    output_path: str | Path,
    annotate_growth_rate: bool = True,
    dpi: int = 150,
) -> Path:
    """Semi-log plot of total kinetic energy vs time.

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        annotate_growth_rate: If True, annotate with mean γτ_A.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    time, ke = compute_ke_growth_trace(case_dir)
    mask = ke > 0
    if mask.sum() < 2:
        raise ValueError("Fewer than 2 non-zero KE points in the trace.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(time[mask], ke[mask], lw=1.5)
    ax.set_xlabel("time (τ_A)")
    ax.set_ylabel("E_K3 (normalised)")
    ax.set_title(f"Kinetic Energy — {Path(case_dir).name}")
    ax.grid(True, which="both", alpha=0.4)

    if annotate_growth_rate:
        gamma = compute_growth_rate(case_dir)
        if math.isfinite(gamma):
            ax.annotate(
                f"γτ_A = {gamma:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                va="top", ha="left", fontsize=9, color="C1",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_growth_rate_vs_time(
    case_dir: str | Path,
    output_path: str | Path,
    smooth_window: int = 5,
    dpi: int = 150,
) -> Path:
    """Instantaneous growth rate γτ_A(t) = 0.5 · d(ln E_K3)/dt.

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        smooth_window: Running-average window width for the smoothed overlay.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    time, ke = compute_ke_growth_trace(case_dir)
    mask = ke > 0
    if mask.sum() < 3:
        raise ValueError("Fewer than 3 non-zero KE points; cannot compute growth rate trace.")

    t = time[mask]
    k = ke[mask]
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    gamma = 0.5 * np.diff(np.log(k)) / dt
    t_mid = 0.5 * (t[:-1] + t[1:])

    w = min(smooth_window, len(gamma))
    gamma_smooth = np.convolve(gamma, np.ones(w) / w, mode="same")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_mid, gamma, alpha=0.35, lw=0.8, color="C0")
    ax.plot(t_mid, gamma_smooth, lw=1.5, color="C0", label=f"smoothed (w={w})")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("time (τ_A)")
    ax.set_ylabel("γτ_A")
    ax.set_title(f"Instantaneous Growth Rate — {Path(case_dir).name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_flux_average_profiles(
    case_dir: str | Path,
    output_path: str | Path,
    fields: list[str] | None = None,
    fcoords: str = "pest",
    points: int = 200,
    dpi: int = 150,
) -> Path:
    """Multi-panel flux-surface-averaged radial profiles vs ψ_norm.

    Uses the equilibrium state (time=-1) from C1.h5.  Requires m3dc1 + fpy.

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        fields: Field names to plot; default ``["p", "j", "ne", "q"]``.
        fcoords: Flux coordinate system (``"pest"`` or ``"equal_arc"``).
        points: Radial grid resolution.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    profiles = compute_flux_average_profiles(
        case_dir, fields=fields, fcoords=fcoords, points=points
    )
    if not profiles:
        raise ValueError("No profiles computed; check case_dir and field names.")

    _labels = {"p": "Pressure", "j": "Current density", "ne": "Electron density nₑ", "q": "Safety factor q"}
    n = len(profiles)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, (name, (psin, profile)) in zip(axes.flat, profiles.items()):
        ax.plot(psin, profile, lw=1.5)
        ax.set_xlabel("ψ_norm")
        ax.set_ylabel(_labels.get(name, name))
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.4)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Flux-averaged profiles — {Path(case_dir).name}")
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_safety_factor(
    case_dir: str | Path,
    output_path: str | Path,
    fcoords: str = "pest",
    points: int = 200,
    dpi: int = 150,
) -> Path:
    """Safety factor q(ψ_norm) with q=1,2,3 reference lines and q95 annotation.

    Uses the equilibrium state (time=-1).  Requires m3dc1 + fpy.

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        fcoords: Flux coordinate system.
        points: Radial grid resolution.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    profiles = compute_flux_average_profiles(
        case_dir, fields=["q"], fcoords=fcoords, points=points
    )
    if "q" not in profiles:
        raise ValueError("q profile not available for this case.")

    psin, q = profiles["q"]
    q95 = compute_q95(psin, q)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(psin, q, lw=1.8)
    for q_ref in (1, 2, 3):
        ax.axhline(q_ref, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.text(0.99, q_ref + 0.05, f"q={q_ref}", ha="right", va="bottom",
                fontsize=8, color="gray")
    if math.isfinite(q95):
        ax.axvline(0.95, color="C1", lw=0.8, ls=":", alpha=0.8)
        ax.annotate(
            f"q₉₅ = {q95:.2f}",
            xy=(0.95, q95), xytext=(0.78, q95 + 0.4),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="C1"),
            fontsize=8, color="C1",
        )
    ax.set_xlabel("ψ_norm")
    ax.set_ylabel("q")
    ax.set_xlim(0, 1)
    ax.set_title(f"Safety Factor — {Path(case_dir).name}")
    ax.grid(True, alpha=0.4)

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _norm_and_m_max(
    m_modes: np.ndarray, spectrum: np.ndarray, min_amplitude: float
) -> tuple[np.ndarray, int]:
    """Normalize spectrum to its peak and return (norm_spec, m_max).

    m_max is the outermost |m| where any psi_norm value exceeds min_amplitude,
    giving a symmetric mode range [-m_max, m_max] that contains all visible structure.
    """
    peak = float(np.nanmax(spectrum))
    norm_spec = spectrum / peak if peak > 0 else spectrum.copy()
    visible = norm_spec.max(axis=1) >= min_amplitude
    m_max = int(np.max(np.abs(m_modes[visible]))) if visible.any() else int(np.max(np.abs(m_modes)))
    return norm_spec, m_max


def plot_poloidal_spectrum(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    fcoords: str = "pest",
    points: int = 200,
    min_amplitude: float = 1e-5,
    dpi: int = 150,
) -> Path:
    """2-D heatmap of poloidal mode spectrum: normalized amplitude[m, ψ_norm].

    The mode range is chosen automatically: all modes where the normalized
    amplitude exceeds ``min_amplitude`` at any ψ_norm value are shown, with the
    range kept symmetric around m=0.  Color scale is logarithmic from
    ``min_amplitude`` to 1.

    Requires m3dc1 + fpy.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index to analyse.
        field: Field name, e.g. ``"p"``, ``"B"``.
        output_path: Destination PNG file.
        coord: Field component (``"scalar"`` for scalars; ``"R"``, ``"Z"``, ``"phi"`` for vectors).
        fcoords: Flux coordinate system.
        points: Radial grid resolution.
        min_amplitude: Normalized amplitude threshold; modes below this at all ψ_norm
            values are excluded, and it sets the colorbar minimum.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m_modes, psi_norm, spectrum = compute_poloidal_spectrum(
        case_dir, time_idx, field, coord=coord, fcoords=fcoords, points=points
    )

    norm_spec, m_max = _norm_and_m_max(m_modes, spectrum, min_amplitude)
    mask = (m_modes >= -m_max) & (m_modes <= m_max)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(
        psi_norm, m_modes[mask], norm_spec[mask, :],
        cmap="viridis", shading="auto",
        norm=LogNorm(vmin=min_amplitude, vmax=1),
    )
    fig.colorbar(im, ax=ax, label="Normalized Amplitude")
    ax.set_xlabel("ψ_norm")
    ax.set_ylabel("poloidal mode $m$")
    ax.set_title(f"Poloidal spectrum — {field} (t={time_idx}) — {Path(case_dir).name}")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_standard_spectra(
    case_dir: str | Path,
    time_idx: int,
    output_path: str | Path,
    fcoords: str = "pest",
    points: int = 200,
    min_amplitude: float = 1e-5,
    dpi: int = 150,
) -> Path:
    """2×2 panel figure: poloidal spectra for p, B_R, B_Z, B_φ.

    All panels share the same mode range, determined by the widest visible range
    across all four fields (modes where normalized amplitude >= ``min_amplitude``
    at any ψ_norm).  Color scale is logarithmic from ``min_amplitude`` to 1,
    independently normalized per panel.

    Requires m3dc1 + fpy.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        output_path: Destination PNG file.
        fcoords: Flux coordinate system.
        points: Radial grid resolution.
        min_amplitude: Normalized amplitude threshold; modes below this at all ψ_norm
            values are excluded, and it sets the colorbar minimum for all panels.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    spectra = compute_standard_spectra(case_dir, time_idx, fcoords=fcoords, points=points)
    if not spectra:
        raise ValueError("No spectra computed; check case_dir and time_idx.")

    _panel_labels = {"p": "Pressure (p)", "br": "B_R", "bz": "B_Z", "bphi": "B_φ"}
    keys = [k for k in ("p", "br", "bz", "bphi") if k in spectra]
    n = len(keys)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)

    # First pass: normalize each panel and find the widest visible mode range.
    prepped: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    m_max_global = 0
    for key in keys:
        m_modes, psi_norm, spectrum = spectra[key]
        norm_spec, m_max = _norm_and_m_max(m_modes, spectrum, min_amplitude)
        m_max_global = max(m_max_global, m_max)
        prepped[key] = (m_modes, psi_norm, norm_spec)

    # Second pass: draw all panels with the shared mode range.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for ax, key in zip(axes.flat, keys):
        m_modes, psi_norm, norm_spec = prepped[key]
        mask = (m_modes >= -m_max_global) & (m_modes <= m_max_global)
        im = ax.pcolormesh(
            psi_norm, m_modes[mask], norm_spec[mask, :],
            cmap="viridis", shading="auto",
            norm=LogNorm(vmin=min_amplitude, vmax=1),
        )
        fig.colorbar(im, ax=ax, label="Normalized Amplitude")
        ax.set_xlabel("ψ_norm")
        ax.set_ylabel("poloidal mode $m$")
        ax.set_title(_panel_labels.get(key, key))

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Poloidal spectra — t={time_idx} — {Path(case_dir).name}")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_perturbed_field_map(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    mode: str = "grid",
    grid_res: int = 200,
    phi: float = 0.0,
    dpi: int = 150,
) -> Path:
    """Filled contour of a perturbed field δf on the R,Z plane.

    Requires m3dc1 + fpy.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index for the perturbed state.
        field: Scalar field name, e.g. ``"psi"``, ``"p"``, ``"ne"``.
        output_path: Destination PNG file.
        mode: Evaluation grid: ``"grid"`` (regular Cartesian) or ``"mesh"`` (mesh vertices).
        grid_res: Points per axis when ``mode="grid"``.
        phi: Toroidal angle in radians.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    case_dir = Path(case_dir)
    R_verts, Z_verts = read_mesh_vertices(case_dir / "C1.h5")
    R, Z, phi_arr = make_evaluation_grid(R_verts, Z_verts, mode=mode, grid_res=grid_res, phi=phi)
    fields_data = compute_perturbed_fields(case_dir, time_idx, R, Z, phi_arr, fields=[field])

    if field not in fields_data:
        raise ValueError(f"Field '{field}' not available. Got: {list(fields_data.keys())}")

    delta_f = fields_data[field]
    vmax = float(np.nanquantile(np.abs(delta_f), 0.99)) or 1.0

    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.pcolormesh(R, Z, delta_f, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label=f"δ{field}")
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Perturbed {field} — t={time_idx}, φ={phi:.2f} rad\n{case_dir.name}")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_geqdsk(
    gfile_path: str | Path,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Flux surface contours and boundary from a GEQDSK equilibrium file.

    No m3dc1 or fpy dependency.

    Args:
        gfile_path: Path to the GEQDSK file (typically named ``geqdsk``).
        output_path: Destination PNG file.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    geo = _parse_geqdsk(Path(gfile_path))
    psirzn = geo["psirzn"]

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.contour(geo["rg"], geo["zg"], psirzn,
               levels=np.linspace(0.1, 0.9, 9), linewidths=0.7, colors="C0")
    out_max = float(np.nanmax(psirzn))
    if out_max > 1.05:
        ax.contour(geo["rg"], geo["zg"], psirzn,
                   levels=np.linspace(1.05, out_max, 20), linewidths=0.7, colors="C0")
    ax.plot(geo["rmaxis"], geo["zmaxis"], lw=0, marker="+", markersize=10, color="C0")
    if geo["rbbbs"]:
        ax.plot(geo["rbbbs"], geo["zbbbs"], color="m", lw=1.2, label="LCFS")
    if geo["rlim"]:
        ax.plot(geo["rlim"], geo["zlim"], color="C1", lw=1.2, label="limiter")
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{Path(gfile_path).name}  (I = {geo['current']:.3e} A)")
    ax.grid(True, alpha=0.4)
    if geo["rbbbs"] or geo["rlim"]:
        ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_geqdsk_compare(
    case_dir: str | Path,
    output_path: str | Path,
    gfile: str = "geqdsk",
    dpi: int = 150,
) -> Path:
    """3-panel comparison: GEQDSK ψ_norm, C1.h5 ψ_norm, and their difference.

    Requires m3dc1 + fpy (evaluates ψ on the GEQDSK grid from C1.h5).

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        gfile: GEQDSK filename relative to case_dir (default ``"geqdsk"``).
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    import fpy
    m3 = _import_m3dc1()

    case_dir = Path(case_dir)
    gfile_path = case_dir / gfile
    if not gfile_path.exists():
        raise FileNotFoundError(f"GEQDSK file not found: {gfile_path}")

    geo = _parse_geqdsk(gfile_path)
    rg, zg, psirzn_geq = geo["rg"], geo["zg"], geo["psirzn"]

    c1h5 = str(case_dir / "C1.h5")
    with _in_case_dir(case_dir):
        sim = fpy.sim_data(c1h5, time=0)
    psi_axis = float(sim.get_time_trace("psimin").values[0])
    psi_lcfs = float(sim.get_time_trace("psi_lcfs").values[0])

    R, Z = np.meshgrid(rg, zg)
    phi_grid = np.zeros_like(R)
    with _in_case_dir(case_dir):
        psi_c1 = m3.eval_field("psi", R, phi_grid, Z, coord="scalar",
                                sim=sim, time=0, quiet=True)
    if psi_lcfs != psi_axis:
        psin_c1 = (psi_c1 - psi_axis) / (psi_lcfs - psi_axis)
    else:
        psin_c1 = np.zeros_like(psi_c1)

    diff = psin_c1 - psirzn_geq
    levels = np.linspace(0.05, 1.2, 18)

    fig, axs = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    axs[0].contour(rg, zg, psirzn_geq, levels=levels, colors="r", linewidths=0.7)
    axs[0].contour(rg, zg, psirzn_geq, levels=[1.0], colors="k", linewidths=2)
    axs[0].set_title("GEQDSK ψ_norm")

    axs[1].contour(rg, zg, psin_c1, levels=levels, colors="C0", linewidths=0.7)
    axs[1].contour(rg, zg, psin_c1, levels=[1.0], colors="k", linewidths=2)
    axs[1].set_title("C1.h5 ψ_norm")

    im = axs[2].contourf(rg, zg, diff, levels=30, cmap="coolwarm")
    axs[2].contour(rg, zg, psin_c1, levels=[1.0], colors="k", linewidths=1.5)
    axs[2].set_title("ψ_norm difference (C1 − GEQDSK)")
    fig.colorbar(im, ax=axs[2], shrink=0.8)

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("Z (m)")
    fig.suptitle(f"GEQDSK comparison — {case_dir.name}")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_tpf_vs_time(
    case_dir: str | Path,
    field: str,
    output_path: str | Path,
    ts_list: list[int] | None = None,
    units: str = "mks",
    points: int = 250,
    dpi: int = 150,
) -> Path:
    """Total poloidal flux of a field as a function of time snapshot.

    Iterates over time snapshots, calling m3.tpf for each.  Requires m3dc1 + fpy.
    A sidecar .dat file with columns (ts, time, tpf) is written alongside output_path.

    Args:
        case_dir: M3D-C1 case directory.
        field: Field name, e.g. ``"prad"``.
        output_path: Destination PNG file.
        ts_list: Time-slice indices to include; None means all available snapshots.
        units: ``"mks"`` or ``"alfven"`` — affects axis label only.
        points: Spatial resolution passed to m3.tpf.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    import fpy
    m3 = _import_m3dc1()

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")

    if ts_list is None:
        from m3dc1_tools import list_time_snapshots
        ts_list = list_time_snapshots(case_dir)
    if not ts_list:
        raise ValueError("No time snapshots found.")

    time_vals, tpf_vals = [], []
    with _in_case_dir(case_dir):
        for ts in ts_list:
            try:
                sim = fpy.sim_data(filename=c1h5, time=ts)
                t, pf = m3.tpf(field=field, sim=sim, filename=c1h5, units=units, res=points)
                time_vals.append(t)
                tpf_vals.append(pf)
            except Exception as exc:
                print(f"WARNING: tpf failed for ts={ts}: {exc}")

    if not time_vals:
        raise RuntimeError("TPF computation failed for all time slices.")

    output_path = Path(output_path)
    dat_path = output_path.with_suffix(".dat")
    with open(dat_path, "w") as f:
        f.write("# ts  time  tpf\n")
        for ts, t, pf in zip(ts_list, time_vals, tpf_vals):
            f.write(f"{ts}    {t}    {pf}\n")

    time_label = "time (s)" if units == "mks" else "time (τ_A)"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_vals, tpf_vals, lw=1.5)
    ax.set_xlabel(time_label)
    ax.set_ylabel(f"TPF ({field})")
    ax.set_title(f"Total Poloidal Flux — {field} — {case_dir.name}")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# CATEGORY B — M3DC1 LIBRARY WRAPPERS
# These functions call m3dc1's own plotting routines and redirect their output
# to a caller-specified path using either the figure-capture pattern or a
# temporary directory.  All require m3dc1 (and usually fpy) to be importable.
# ============================================================================

def plot_field(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    phi: float = 0.0,
    points: int = 250,
    tor_av: int = 1,
    units: str = "mks",
    mesh: bool = False,
    bound: bool = False,
    lcfs: bool = False,
    dpi: int = 150,
) -> Path:
    """2-D field contour on the R,Z plane via m3dc1.plot_field.

    m3dc1 saves the figure internally; this function redirects that save to a
    temporary directory then moves the result to output_path.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name, e.g. ``"psi"``, ``"te"``, ``"ne"``, ``"jphi"``.
        output_path: Destination PNG file.
        coord: Field component (``"scalar"``, ``"R"``, ``"Z"``, ``"phi"``).
        phi: Toroidal angle for the slice (radians).
        points: R/Z grid resolution.
        tor_av: Toroidal average over N planes.
        units: ``"mks"`` or ``"alfven"``.
        mesh: Overlay mesh triangulation.
        bound: Overlay boundary only (no full mesh).
        lcfs: Overlay last closed flux surface.
        dpi: Figure resolution (applied when moving the saved file; m3.plot_field
             controls its own dpi internally).

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        before_files = set(tmpdir_path.iterdir())
        m3.plot_field(
            field=field,
            filename=c1h5,
            time=time_idx,
            coord=coord,
            phi=phi,
            points=points,
            tor_av=tor_av,
            units=units,
            mesh=mesh,
            bound=bound,
            lcfs=lcfs,
            save=True,
            savedir=str(tmpdir_path) + os.sep,
            quiet=True,
        )
        plt.close("all")
        new_files = sorted(set(tmpdir_path.iterdir()) - before_files)
        if not new_files:
            raise RuntimeError("m3.plot_field produced no output file in tmpdir.")
        shutil.copy2(new_files[0], output_path)

    return output_path


def plot_mesh(
    case_dir: str | Path,
    output_path: str | Path,
    boundary: bool = False,
    dpi: int = 150,
) -> Path:
    """Triangular mesh visualisation via m3dc1.plot_mesh.

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        boundary: If True, show boundary outline only (no interior mesh).
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_mesh(filename=c1h5, save=False, boundary=boundary)
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_flux_surface_shape(
    case_dir: str | Path,
    time_idx: int,
    output_path: str | Path,
    phi: float = 0.0,
    points: int = 250,
    mesh: bool = False,
    bound: bool = False,
    lcfs: bool = False,
    dpi: int = 150,
) -> Path:
    """Flux surface shape on the R,Z plane via m3dc1.plot_shape.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        output_path: Destination PNG file.
        phi: Toroidal angle in radians.
        points: Resolution used to trace flux surfaces.
        mesh: Overlay mesh triangulation.
        bound: Overlay boundary.
        lcfs: Overlay last closed flux surface.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_shape(
        filename=c1h5,
        time=time_idx,
        phi=phi,
        res=points,
        mesh=mesh,
        bound=bound,
        lcfs=lcfs,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_field_basic(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    phi: float = 0.0,
    points: int = 250,
    tor_av: int = 1,
    units: str = "mks",
    mesh: bool = False,
    dpi: int = 150,
) -> Path:
    """2-D field plot via m3dc1.plot_field_basic.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name.
        output_path: Destination PNG file.
        coord: Field component.
        phi: Toroidal angle in radians.
        points: R/Z resolution.
        tor_av: Toroidal average planes.
        units: Unit system.
        mesh: Overlay mesh.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    import importlib
    m3 = _import_m3dc1()  # ensure m3dc1 is available before submodule import
    pfb = importlib.import_module("m3dc1.plot_field_basic")
    pfb.shortlbl = False

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    pfb.plot_field_basic(
        field=field,
        coord=coord,
        filename=c1h5,
        time=time_idx,
        phi=phi,
        mesh=mesh,
        linear=False,
        diff=False,
        tor_av=tor_av,
        units=units,
        res=points,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_field_mesh(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    phi: float = 0.0,
    tor_av: int = 1,
    units: str = "mks",
    mesh: bool = False,
    mesh_res: int = 0,
    dpi: int = 150,
) -> Path:
    """Field plotted on the mesh triangulation via m3dc1.plot_field_mesh.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name.
        output_path: Destination PNG file.
        coord: Field component.
        phi: Toroidal angle in radians.
        tor_av: Toroidal average planes.
        units: Unit system.
        mesh: Overlay mesh edges.
        mesh_res: Mesh refinement level (0 = native resolution).
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    import importlib
    m3 = _import_m3dc1()
    pfm = importlib.import_module("m3dc1.plot_field_mesh")
    pfm.shortlbl = False

    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    pfm.plot_field_mesh(
        field=field,
        coord=coord,
        filename=c1h5,
        time=time_idx,
        phi=phi,
        mesh=mesh,
        linear=False,
        diff=False,
        tor_av=tor_av,
        units=units,
        res=mesh_res,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_field_vs_phi(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    cutr: float | None = None,
    cutz: float | None = None,
    coord: str = "scalar",
    phi_res: int = 100,
    points: int = 250,
    mesh: bool = False,
    bound: bool = False,
    units: str = "mks",
    dpi: int = 150,
) -> Path:
    """Field amplitude vs toroidal angle φ at a fixed R or Z cut.

    Exactly one of cutr or cutz must be supplied.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name.
        output_path: Destination PNG file.
        cutr: Fixed R value (metres) for the cut; mutually exclusive with cutz.
        cutz: Fixed Z value (metres) for the cut; mutually exclusive with cutr.
        coord: Field component.
        phi_res: Number of toroidal angle points.
        points: Poloidal resolution.
        mesh: Overlay mesh.
        bound: Overlay boundary.
        units: Unit system.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    if (cutr is None) == (cutz is None):
        raise ValueError("Exactly one of cutr or cutz must be provided.")

    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_field_vs_phi(
        field=field,
        cutr=cutr,
        cutz=cutz,
        coord=coord,
        filename=c1h5,
        time=time_idx,
        phi_res=phi_res,
        res=points,
        mesh=mesh,
        bound=bound,
        units=units,
        save=False,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_flux_average_m3(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    fcoords: str = "pest",
    units: str = "mks",
    dpi: int = 150,
) -> Path:
    """Flux-averaged profile via m3dc1.plot_flux_average.

    The ``_m3`` suffix distinguishes this from the pure-matplotlib
    ``plot_flux_average_profiles`` in Category A.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name, e.g. ``"p"``.
        output_path: Destination PNG file.
        coord: Field component.
        fcoords: Flux coordinate system.
        units: Unit system.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_flux_average(
        field=field,
        coord=coord,
        fcoords=fcoords,
        filename=c1h5,
        time=time_idx,
        units=units,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_line(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    angle: float = 0.0,
    zoff: float = 0.0,
    phi: float = 0.0,
    tor_av: int = 1,
    units: str = "mks",
    dist_from_magax: bool = False,
    dpi: int = 150,
) -> Path:
    """Field profile along a poloidal line via m3dc1.plot_line.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name.
        output_path: Destination PNG file.
        coord: Field component.
        angle: Poloidal angle of the line in degrees.
        zoff: Z offset of the line origin (metres).
        phi: Toroidal angle in radians.
        tor_av: Toroidal average planes.
        units: Unit system.
        dist_from_magax: If True, use distance from magnetic axis as x-axis.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_line(
        field=field,
        coord=coord,
        angle=angle,
        Zoff=zoff,
        dist_from_magax=dist_from_magax,
        filename=c1h5,
        time=time_idx,
        phi=phi,
        tor_av=tor_av,
        units=units,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_eigenfunction(
    case_dir: str | Path,
    time_idx: int,
    field: str,
    output_path: str | Path,
    coord: str = "scalar",
    phit: float = 0.0,
    fcoords: str | None = None,
    points: int = 200,
    units: str = "mks",
    dpi: int = 150,
) -> Path:
    """Eigenfunction and poloidal spectrum via m3dc1.eigenfunction.

    Must be called from case_dir (handled internally).

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index.
        field: Field name, e.g. ``"p"``.
        output_path: Destination PNG file.
        coord: Field component.
        phit: Toroidal angle in radians.
        fcoords: Flux coordinate system (None uses m3dc1 default).
        points: Grid resolution.
        units: Unit system.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    with _in_case_dir(case_dir):
        m3.eigenfunction(
            field=field,
            coord=coord,
            filename=c1h5,
            time=time_idx,
            phit=phit,
            fcoords=fcoords,
            points=points,
            units=units,
            makeplot=True,
            save=False,
        )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_poincare(
    case_dir: str | Path,
    time_idx: int,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Poincaré section via m3dc1.plot_poincare.

    Poincaré data must already exist in case_dir (generated by m3.run_trace or
    equivalent).  m3dc1 saves the figure to a directory internally; this
    function redirects it to a temporary directory then moves the result to
    output_path.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index to plot.
        output_path: Destination PNG file.
        dpi: Not applied (m3dc1 controls its own dpi).

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with _in_case_dir(case_dir):
            m3.plot_poincare(
                time=time_idx,
                savefig=True,
                savepath=str(tmpdir_path) + os.sep,
            )
        plt.close("all")
        new_files = sorted(tmpdir_path.iterdir())
        if not new_files:
            raise RuntimeError("m3.plot_poincare produced no output file.")
        shutil.copy2(new_files[0], output_path)

    return output_path


def plot_diagnostics(
    case_dir: str | Path,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Diagnostic timing and iteration history via m3dc1.plot_diagnostics.

    Args:
        case_dir: M3D-C1 case directory.
        output_path: Destination PNG file.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_diagnostics(filename=c1h5)
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_signal(
    case_dir: str | Path,
    signal: str,
    output_path: str | Path,
    pspec: bool = False,
    pts_per_probe: int = 1,
    dpi: int = 150,
) -> Path:
    """Diagnostic signal time traces via m3dc1.plot_signal.

    Args:
        case_dir: M3D-C1 case directory.
        signal: Signal type, e.g. ``"mag_probes"``.
        output_path: Destination PNG file.
        pspec: If True, also plot the power spectrum.
        pts_per_probe: Points per probe.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_signal(
        signal=signal,
        filename=c1h5,
        pspec=pspec,
        pts_per_probe=pts_per_probe,
    )
    _save_first_new_fig(before, output_path, dpi)
    return output_path


def plot_time_trace(
    case_dir: str | Path,
    trace: str,
    output_path: str | Path,
    units: str = "mks",
    dpi: int = 150,
) -> Path:
    """Global scalar time trace via m3dc1.plot_time_trace_fast.

    Args:
        case_dir: M3D-C1 case directory.
        trace: Trace name, e.g. ``"ke"``.
        output_path: Destination PNG file.
        units: ``"mks"`` or ``"alfven"``.
        dpi: Figure resolution.

    Returns:
        Path to the saved figure.
    """
    m3 = _import_m3dc1()
    case_dir = Path(case_dir)
    c1h5 = str(case_dir / "C1.h5")
    output_path = Path(output_path)

    before = set(plt.get_fignums())
    m3.plot_time_trace_fast(trace=trace, filename=c1h5, units=units, save=False)
    _save_first_new_fig(before, output_path, dpi)
    return output_path


# ============================================================================
# CONVENIENCE WRAPPERS
# These call individual functions above to produce standard sets of figures.
# Each returns a list of Paths for all files created.
# ============================================================================

def plot_stability_summary(
    case_dir: str | Path,
    time_idx: int,
    output_dir: str | Path,
    prefix: str = "",
    dpi: int = 150,
) -> list[Path]:
    """Standard linear stability figure set for one time snapshot.

    Produces: KE trace, growth rate trace, standard spectra, p spectrum, δpsi map.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index for spectral and field-map figures.
        output_dir: Directory where figures are written.
        prefix: Optional filename prefix for all output files.
        dpi: Figure resolution.

    Returns:
        List of Paths created (skips any figure that raises an exception).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name, func, kwargs in [
        ("ke.png",          plot_kinetic_energy,       {"case_dir": case_dir}),
        ("growth_rate.png", plot_growth_rate_vs_time,  {"case_dir": case_dir}),
    ]:
        try:
            p = output_dir / f"{prefix}{name}"
            results.append(func(output_path=p, dpi=dpi, **kwargs))
        except Exception as exc:
            print(f"WARNING: {name} failed: {exc}")

    for name, func, kwargs in [
        ("spectra.png",   plot_standard_spectra, {"case_dir": case_dir, "time_idx": time_idx}),
        ("spectrum_p.png", plot_poloidal_spectrum, {"case_dir": case_dir, "time_idx": time_idx, "field": "p"}),
        ("field_psi.png", plot_perturbed_field_map, {"case_dir": case_dir, "time_idx": time_idx, "field": "psi"}),
    ]:
        try:
            p = output_dir / f"{prefix}{name}"
            results.append(func(output_path=p, dpi=dpi, **kwargs))
        except Exception as exc:
            print(f"WARNING: {name} failed: {exc}")

    return results


def plot_equilibrium_overview(
    case_dir: str | Path,
    output_dir: str | Path,
    prefix: str = "",
    dpi: int = 150,
) -> list[Path]:
    """Equilibrium figure set: flux-averaged profiles, q profile, and mesh.

    Args:
        case_dir: M3D-C1 case directory.
        output_dir: Directory where figures are written.
        prefix: Optional filename prefix.
        dpi: Figure resolution.

    Returns:
        List of Paths created.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name, func in [
        ("profiles.png",  plot_flux_average_profiles),
        ("q_profile.png", plot_safety_factor),
        ("mesh.png",      plot_mesh),
    ]:
        try:
            p = output_dir / f"{prefix}{name}"
            results.append(func(case_dir=case_dir, output_path=p, dpi=dpi))
        except Exception as exc:
            print(f"WARNING: {name} failed: {exc}")

    return results


def plot_case_summary(
    case_dir: str | Path,
    time_idx: int,
    output_dir: str | Path,
    prefix: str = "",
    dpi: int = 150,
) -> list[Path]:
    """Complete figure set: stability summary + equilibrium overview.

    Args:
        case_dir: M3D-C1 case directory.
        time_idx: Snapshot index for spectral and field-map figures.
        output_dir: Directory where figures are written.
        prefix: Optional filename prefix.
        dpi: Figure resolution.

    Returns:
        Combined list of Paths created by both wrappers.
    """
    results = plot_stability_summary(case_dir, time_idx, output_dir, prefix=prefix, dpi=dpi)
    results += plot_equilibrium_overview(case_dir, output_dir, prefix=prefix, dpi=dpi)
    return results
