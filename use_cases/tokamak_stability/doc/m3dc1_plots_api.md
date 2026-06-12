# m3dc1_plots.py — API Reference

Plotting library for M3D-C1 simulation output.  All public functions produce
exactly one figure, save it to a caller-specified path, close the figure, and
return the output `Path`.  No `plt.show()` calls are made; the module is
headless-safe.

---

## Architecture

Functions are divided into two categories and one layer of convenience wrappers,
all clearly labelled with section headers in the source file.

### Category A — Pure matplotlib

Data is loaded and computed by calling functions from `m3dc1_tools.py`.  No
m3dc1 library plotting routines are called.  These functions can be used
anywhere h5py and numpy are available; those that call fpy-dependent
`m3dc1_tools` functions will emit C/Fortran-level diagnostics to stdout (see
[stdout note](#stdout-note)).

### Category B — m3dc1 library wrappers

These wrap plotting routines from the `m3dc1` Python package.  The module uses
one of two patterns to redirect output to a caller-supplied path:

- **Figure-capture pattern** (most functions): records `plt.get_fignums()`
  before and after the m3dc1 call, saves the first new figure to `output_path`,
  closes all new figures.
- **Tempdir pattern** (`plot_field`, `plot_poincare`): calls the m3dc1 function
  with its own save mechanism directed to a temporary directory, then moves the
  resulting file to `output_path`.

All Category B functions raise `ImportError` with a helpful message if `m3dc1`
is not importable.

### Stdout note

Category A functions that call fpy-dependent `m3dc1_tools` functions
(`plot_flux_average_profiles`, `plot_safety_factor`, `plot_poloidal_spectrum`,
`plot_standard_spectra`, `plot_perturbed_field_map`, `plot_geqdsk_compare`) will
emit C/Fortran-level diagnostic messages (e.g. `"deleting simulation object"`)
directly to the OS-level stdout.  This is harmless in library use — Python
output is not captured.  It becomes an issue only when wrapping these functions
in a CLI tool that captures stdout; use `--output-json FILE` as described in
`m3dc1_tools.py`.

---

## Category A Functions

### `plot_kinetic_energy`

```python
plot_kinetic_energy(
    case_dir, output_path,
    annotate_growth_rate=True, dpi=150
) -> Path
```

Semi-log plot of total kinetic energy (E_K3) vs time in Alfvén units.
Optionally annotates with the mean linear growth rate γτ_A computed by
`compute_growth_rate`.

---

### `plot_growth_rate_vs_time`

```python
plot_growth_rate_vs_time(
    case_dir, output_path,
    smooth_window=5, dpi=150
) -> Path
```

Instantaneous growth rate γτ_A(t) = 0.5 · d(ln E_K3)/dt.  Shows both the raw
trace (faint) and a running-average smoothed overlay.

---

### `plot_flux_average_profiles`

```python
plot_flux_average_profiles(
    case_dir, output_path,
    fields=None, fcoords="pest", points=200, dpi=150
) -> Path
```

Multi-panel figure with one subplot per field vs ψ_norm.  Uses the equilibrium
state (time = −1).  Default fields: `["p", "j", "ne", "q"]`.  Requires m3dc1 + fpy.

---

### `plot_safety_factor`

```python
plot_safety_factor(
    case_dir, output_path,
    fcoords="pest", points=200, dpi=150
) -> Path
```

q(ψ_norm) with horizontal reference lines at q = 1, 2, 3 and a q₉₅ annotation.
Uses the equilibrium state.  Requires m3dc1 + fpy.

---

### `plot_poloidal_spectrum`

```python
plot_poloidal_spectrum(
    case_dir, time_idx, field, output_path,
    coord="scalar", fcoords="pest", points=200, n_modes=20, dpi=150
) -> Path
```

2-D heatmap: spectrum amplitude as a function of poloidal mode number m (y-axis)
and ψ_norm (x-axis).  Shows ±`n_modes` harmonics around m = 0.  Requires m3dc1 + fpy.

---

### `plot_standard_spectra`

```python
plot_standard_spectra(
    case_dir, time_idx, output_path,
    fcoords="pest", points=200, n_modes=20, dpi=150
) -> Path
```

2 × 2 panel figure: poloidal spectra for p, B_R, B_Z, B_φ.  Requires m3dc1 + fpy.

---

### `plot_perturbed_field_map`

```python
plot_perturbed_field_map(
    case_dir, time_idx, field, output_path,
    mode="grid", grid_res=200, phi=0.0, dpi=150
) -> Path
```

Filled contour of δf = f(time_idx) − f(equilibrium) on the R,Z plane at a
given toroidal angle φ.  Uses a diverging colormap (RdBu_r) symmetric about
zero.  Requires m3dc1 + fpy.

| Parameter | Description |
|-----------|-------------|
| `field`   | Scalar field name: `"psi"`, `"p"`, `"ne"`, etc. |
| `mode`    | `"grid"` (regular Cartesian grid) or `"mesh"` (mesh vertices) |
| `grid_res`| Points per axis for `mode="grid"` |
| `phi`     | Toroidal angle in radians |

---

### `plot_geqdsk`

```python
plot_geqdsk(gfile_path, output_path, dpi=150) -> Path
```

Flux surface contours and plasma/limiter boundary from a GEQDSK equilibrium
file.  No m3dc1 or fpy dependency.

---

### `plot_geqdsk_compare`

```python
plot_geqdsk_compare(
    case_dir, output_path,
    gfile="geqdsk", dpi=150
) -> Path
```

3-panel figure: GEQDSK ψ_norm (left), C1.h5 ψ_norm (centre), difference
(right).  Requires m3dc1 + fpy to evaluate ψ from C1.h5 on the GEQDSK grid.

---

### `plot_tpf_vs_time`

```python
plot_tpf_vs_time(
    case_dir, field, output_path,
    ts_list=None, units="mks", points=250, dpi=150
) -> Path
```

Total poloidal flux of a field as a function of time snapshot.  Iterates over
`ts_list` (default: all available snapshots), calling `m3.tpf` for each.
Writes a sidecar `.dat` file (columns: `ts  time  tpf`) alongside `output_path`.
Requires m3dc1 + fpy.

---

## Category B Functions

All Category B functions call `_import_m3dc1()` at entry and raise `ImportError`
if m3dc1 is not available.

### `plot_field`

```python
plot_field(
    case_dir, time_idx, field, output_path,
    coord="scalar", phi=0.0, points=250, tor_av=1,
    units="mks", mesh=False, bound=False, lcfs=False, dpi=150
) -> Path
```

2-D field contour on the R,Z plane via `m3.plot_field`.  Uses the tempdir
pattern (m3dc1 saves the file; this function moves it to `output_path`).

---

### `plot_mesh`

```python
plot_mesh(case_dir, output_path, boundary=False, dpi=150) -> Path
```

Triangular mesh visualisation via `m3.plot_mesh`.

---

### `plot_flux_surface_shape`

```python
plot_flux_surface_shape(
    case_dir, time_idx, output_path,
    points=200, dpi=150
) -> Path
```

Flux surface psi contours on the R,Z plane via `m3.plot_shape`.

---

### `plot_field_basic`

```python
plot_field_basic(
    case_dir, time_idx, field, output_path,
    coord="scalar", phi=0.0, points=250, tor_av=1,
    units="mks", mesh=False, dpi=150
) -> Path
```

2-D filled-contour field plot via `m3.plot_field`.

---

### `plot_field_mesh`

```python
plot_field_mesh(
    case_dir, time_idx, field, output_path,
    coord="scalar", phi=0.0, dpi=150
) -> Path
```

Field plotted on the mesh triangulation via `m3dc1.plot_field_mesh`.

---

### `plot_field_vs_phi`

```python
plot_field_vs_phi(
    case_dir, time_idx, field, output_path,
    R=None, Z=None, coord="scalar", phi_res=64, dpi=150
) -> Path
```

Field vs toroidal angle φ at a fixed (R, Z) point.  `R` and `Z` default to the
magnetic axis when not supplied.

---

### `plot_flux_average_m3`

```python
plot_flux_average_m3(
    case_dir, time_idx, field, output_path,
    fcoords="pest", points=200, dpi=150
) -> Path
```

Flux-averaged profile via `m3.plot_flux_average`.  The `_m3` suffix
distinguishes this from the pure-matplotlib `plot_flux_average_profiles` in
Category A.

---

### `plot_line`

```python
plot_line(
    case_dir, time_idx, field, output_path,
    coord="scalar", angle=0.0, zoff=0.0,
    dist_from_magax=False, dpi=150
) -> Path
```

Field profile along a radial line at fixed phi=0 via `m3.plot_line`.

---

### `plot_eigenfunction`

```python
plot_eigenfunction(
    case_dir, time_idx, field, output_path,
    coord="scalar", phit=0.0, fcoords="pest",
    points=200, dpi=150
) -> Path
```

Eigenfunction and poloidal spectrum via `m3.eigenfunction`.  Internally runs
`flux_coordinates` on the equilibrium (time=0) then evaluates the perturbed
field at `time_idx`, producing the `[fc_result, lin_sim]` pair that
`eigenfunction` requires.

---

### `plot_poincare`

```python
plot_poincare(case_dir, time_idx, output_path, dpi=150) -> Path
```

Poincaré section via `m3.plot_poincare`.  Requires pre-computed Poincaré data
in `case_dir` (generated by `m3.run_trace` or equivalent).  Uses the tempdir
pattern.

---

### `plot_signal`

```python
plot_signal(
    case_dir, signal, output_path,
    pspec=False, pts_per_probe=1, dpi=150
) -> Path
```

Diagnostic signal time traces via `m3.plot_signal`.

---

### `plot_time_trace`

```python
plot_time_trace(
    case_dir, trace, output_path,
    units="mks", dpi=150
) -> Path
```

Global scalar time trace via `m3.plot_time_trace_fast`.

---

## Convenience Wrappers

### `plot_stability_summary`

```python
plot_stability_summary(
    case_dir, time_idx, output_dir,
    prefix="", dpi=150
) -> list[Path]
```

Produces the standard linear stability figure set:

| Filename | Function called |
|----------|----------------|
| `{prefix}ke.png` | `plot_kinetic_energy` |
| `{prefix}growth_rate.png` | `plot_growth_rate_vs_time` |
| `{prefix}spectra.png` | `plot_standard_spectra` |
| `{prefix}spectrum_p.png` | `plot_poloidal_spectrum(field="p")` |
| `{prefix}field_psi.png` | `plot_perturbed_field_map(field="psi")` |

Individual failures are caught and printed as warnings; successful outputs are
collected in the returned list.

---

### `plot_equilibrium_overview`

```python
plot_equilibrium_overview(
    case_dir, output_dir,
    prefix="", dpi=150
) -> list[Path]
```

Equilibrium figure set:

| Filename | Function called |
|----------|----------------|
| `{prefix}profiles.png` | `plot_flux_average_profiles` |
| `{prefix}q_profile.png` | `plot_safety_factor` |
| `{prefix}mesh.png` | `plot_mesh` |

---

### `plot_case_summary`

```python
plot_case_summary(
    case_dir, time_idx, output_dir,
    prefix="", dpi=150
) -> list[Path]
```

Calls `plot_stability_summary` then `plot_equilibrium_overview` and returns
the combined list of Paths.

---

## Usage Examples

```python
import sys
sys.path.insert(0, "/path/to/tokamak_stability")
import m3dc1_plots

case = "/data/runs/sparc_1425"

# Quick KE check (no m3dc1 needed)
m3dc1_plots.plot_kinetic_energy(case, "ke.png")

# Safety factor (requires m3dc1 + fpy)
m3dc1_plots.plot_safety_factor(case, "q_profile.png")

# Full stability figure set for snapshot 1
paths = m3dc1_plots.plot_stability_summary(case, time_idx=1, output_dir="figs/")
print("Created:", paths)
```
