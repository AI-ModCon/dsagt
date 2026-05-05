# m3dc1_tools — API Reference

All functions live in `m3dc1_tools.py` in the `use_cases/tokamak_stability/`
directory. Import them directly:

```python
from m3dc1_tools import read_case_metadata, compute_growth_rate, ...
```

Functions are grouped by dependency level. The first three groups require only
`h5py` and `numpy`. The final two groups additionally require the `m3dc1` and
`fpy` libraries.


---

## Case Inspection

### `read_c1input(case_dir)`

Parse the `C1input` Fortran namelist and return the key simulation parameters
as a plain dict.

| Parameter  | Type   | Description                                  |
|------------|--------|----------------------------------------------|
| `case_dir` | path   | M3D-C1 case directory containing `C1input`   |

Returns `dict` with keys: `ntor` (int), `pscale`, `batemanscale`, `dt`
(float), `ntimemax`, `ntimepr`, `linear`, `numvar` (int), `ion_mass`, `zeff`
(float). Any absent parameter is `None`.

```python
params = read_c1input("sparc_1425")
# {'ntor': 9, 'pscale': 0.815, 'linear': 1, ...}
```

---

### `list_time_snapshots(case_dir)`

Return a sorted list of integer indices for the `time_NNN.h5` snapshot files
present in the case directory.

| Parameter  | Type | Description               |
|------------|------|---------------------------|
| `case_dir` | path | M3D-C1 case directory     |

Returns `list[int]`, e.g. `[0, 1]`. Returns `[]` if no snapshots are found.

```python
snaps = list_time_snapshots("sparc_1425")
# [0, 1]
```

---

### `read_snapshot_time(case_dir, time_idx, units="alfven")`

Return the simulation time of a specific snapshot.

| Parameter  | Type | Description                                              |
|------------|------|----------------------------------------------------------|
| `case_dir` | path | M3D-C1 case directory                                    |
| `time_idx` | int  | Snapshot file index (the NNN in `time_NNN.h5`)           |
| `units`    | str  | `"alfven"` (default) or `"s"`/`"mks"` for SI seconds    |

Returns `float`. `"s"` requires `m3dc1`; falls back to Alfvén times with a
warning if unavailable. Returns `nan` when the snapshot is not found.

```python
t = read_snapshot_time("sparc_1425", 1)          # → 1000.0 (Alfvén times)
t_s = read_snapshot_time("sparc_1425", 1, "mks") # → physical seconds
```

---

### `read_scalar_traces(case_dir, names=None)`

Read global time-trace diagnostics from `C1.h5/scalars/`.

| Parameter  | Type            | Description                                              |
|------------|-----------------|----------------------------------------------------------|
| `case_dir` | path            | M3D-C1 case directory                                    |
| `names`    | list[str] | None | Scalar names to read. `None` returns all (~97 traces).   |

Returns `dict[str, np.ndarray]` — each array has length `ntimestep + 1`. The
`"time"` key gives Alfvén times.

```python
traces = read_scalar_traces("sparc_1425", names=["time", "E_K3", "W_M"])
t   = traces["time"]   # Alfvén times, shape (1001,)
ke  = traces["E_K3"]   # kinetic energy, shape (1001,)
```

---

### `read_case_metadata(case_dir)`

Return a combined summary of a case's parameters and equilibrium geometry.
Useful as the first call when an agent encounters an unfamiliar case.

| Parameter  | Type | Description           |
|------------|------|-----------------------|
| `case_dir` | path | M3D-C1 case directory |

Returns `dict` with keys: `params`, `snapshots`, `final_time`, `R_mag`,
`Z_mag`, `R_xpoint`, `Z_xpoint`, `psi_min`, `psi_lcfs`.

```python
meta = read_case_metadata("sparc_1425")
# meta["params"]["ntor"]  → 9
# meta["R_mag"]           → 1.855 m
# meta["snapshots"]       → [0, 1]
```

---

## Mesh and Evaluation Grid

### `read_mesh_vertices(c1h5_path)`

Extract the unique (R, Z) mesh vertex positions from the `mesh/elements`
array in an M3D-C1 HDF5 file.

| Parameter    | Type | Description                                          |
|--------------|------|------------------------------------------------------|
| `c1h5_path`  | path | Path to `C1.h5`, `equilibrium.h5`, or `time_NNN.h5` |

Returns `(R, Z)` — two 1-D `float32` arrays sorted lexicographically.

```python
R, Z = read_mesh_vertices("sparc_1425/C1.h5")
# R shape: (n_vertices,)  R ∈ [1.25, 2.55] m
```

---

### `make_evaluation_grid(R_mesh, Z_mesh, mode="mesh", grid_res=200, phi=0.0)`

Build `(R, Z, phi)` arrays ready for passing to `eval_field()`.

| Parameter  | Type    | Description                                                  |
|------------|---------|--------------------------------------------------------------|
| `R_mesh`   | ndarray | Mesh vertex R coordinates from `read_mesh_vertices`          |
| `Z_mesh`   | ndarray | Mesh vertex Z coordinates                                    |
| `mode`     | str     | `"mesh"` (use vertices) or `"grid"` (regular Cartesian grid) |
| `grid_res` | int     | Grid points per axis for `mode="grid"`                       |
| `phi`      | float   | Toroidal angle in radians applied to all points              |

Returns `(R, Z, phi_arr)` with identical shapes. **Note:** `eval_field` takes
arguments as `(name, R, phi, Z, ...)` — pass `phi_arr` before `Z`.

```python
R, Z = read_mesh_vertices("sparc_1425/C1.h5")
R_eval, Z_eval, phi_eval = make_evaluation_grid(R, Z, mode="mesh")
# or: R_eval, Z_eval, phi_eval = make_evaluation_grid(R, Z, mode="grid", grid_res=200)
```

---

## Growth Rate

### `compute_ke_growth_trace(case_dir)`

Return the total kinetic energy `E_K3` and corresponding time array directly
from `C1.h5/scalars/`. No m3dc1 dependency.

| Parameter  | Type | Description           |
|------------|------|-----------------------|
| `case_dir` | path | M3D-C1 case directory |

Returns `(time, ke)` — two 1-D float arrays of length `ntimestep + 1`.

```python
t, ke = compute_ke_growth_trace("sparc_1425")
# ke[-1] / ke[1]  ≈ 7 × 10⁵  (mode amplitude growth over 1000 τ_A)
```

---

### `compute_growth_rate(case_dir, time_idx=None)`

Estimate the linear growth rate γ = 0.5 · d(ln E_K3)/dt, averaged over the
kinetic energy trace. No m3dc1 dependency.

| Parameter  | Type     | Description                                                |
|------------|----------|------------------------------------------------------------|
| `case_dir` | path     | M3D-C1 case directory                                      |
| `time_idx` | int|None | Snapshot index to truncate the trace at. `None` = full.    |

Returns `float` in units of 1/τ_A. Returns `nan` if insufficient data.

```python
gamma = compute_growth_rate("sparc_1425")
# → 0.0067 τ_A⁻¹  (n=9 mode growth rate)
```

---

## Equilibrium Profiles  *(requires m3dc1 + fpy)*

### `compute_flux_average_profiles(case_dir, fields=None, fcoords="pest", points=200)`

Compute flux-surface-averaged radial profiles for equilibrium quantities.

| Parameter  | Type      | Description                                     |
|------------|-----------|-------------------------------------------------|
| `case_dir` | path      | M3D-C1 case directory                           |
| `fields`   | list|None | Field names. Default: `["p", "j", "ne", "q"]`   |
| `fcoords`  | str       | Flux coordinate system: `"pest"` or `"equal_arc"` |
| `points`   | int       | Number of radial grid points                    |

Returns `dict[str, (psi_norm, profile)]` where each value is a pair of 1-D
`float64` arrays of length `points`. Fields that fail are omitted.

```python
profiles = compute_flux_average_profiles("sparc_1425")
psin, q = profiles["q"]   # safety factor profile vs psi_norm
```

---

### `compute_q95(psin, q_profile)`

Interpolate the safety factor at psi_norm = 0.95.

| Parameter   | Type    | Description                                   |
|-------------|---------|-----------------------------------------------|
| `psin`      | ndarray | Normalised flux values (0 to ≥ 0.95)          |
| `q_profile` | ndarray | Safety factor at each psin                    |

Returns `float`. Returns `nan` if the profile does not reach psi_norm = 0.95.

```python
psin, q = profiles["q"]
q95 = compute_q95(psin, q)   # → e.g. 3.2
```

---

### `compute_miller_geometry(case_dir, res=250)`

Compute the Miller geometry parameters of the last closed flux surface.

| Parameter  | Type | Description                           |
|------------|------|---------------------------------------|
| `case_dir` | path | M3D-C1 case directory                 |
| `res`      | int  | Poloidal resolution for LCFS tracing  |

Returns `dict` with keys `"R0"` (m), `"a"` (m), `"kappa"`, `"delta"`.
Returns `{}` on failure.

```python
shape = compute_miller_geometry("sparc_1425")
# {"R0": 1.85, "a": 0.57, "kappa": 1.85, "delta": 0.42}
```

---

## Perturbed Fields  *(requires m3dc1 + fpy)*

### `compute_perturbed_fields(case_dir, time_idx, R, Z, phi, fields="all", skip_fields=None)`

Evaluate perturbed fields (Δf = f_linear − f_equilibrium) at arbitrary
(R, Z, φ) points for one time snapshot.

| Parameter     | Type        | Description                                             |
|---------------|-------------|---------------------------------------------------------|
| `case_dir`    | path        | M3D-C1 case directory                                   |
| `time_idx`    | int         | Snapshot index                                          |
| `R`           | ndarray     | R evaluation coordinates                               |
| `Z`           | ndarray     | Z evaluation coordinates (same shape as R)              |
| `phi`         | ndarray     | Toroidal angles in radians (same shape as R)            |
| `fields`      | str or list | `"all"` or list of field names                         |
| `skip_fields` | list|None   | Fields to skip. Default: `{"alpha", "gradA", "kprad_rad"}` |

Returns `dict[str, ndarray]`. Vector fields (B, E, A, j, v) produce keys like
`"BR"`, `"BPHI"`, `"BZ"`. Shapes match R.

```python
R, Z = read_mesh_vertices("sparc_1425/C1.h5")
R, Z, phi = make_evaluation_grid(R, Z)
perts = compute_perturbed_fields("sparc_1425", 1, R, Z, phi, fields=["psi", "B"])
# perts["psi"]  → 1-D array, same length as R
# perts["BR"]   → R-component of perturbed B
```

---

## Spectral Analysis  *(requires m3dc1 + fpy)*

### `compute_poloidal_spectrum(case_dir, time_idx, field, coord="scalar", fcoords="pest", points=200, full_fft=False)`

Compute the poloidal mode (m) spectrum of a field at a given snapshot.

| Parameter   | Type | Description                                              |
|-------------|------|----------------------------------------------------------|
| `case_dir`  | path | M3D-C1 case directory                                    |
| `time_idx`  | int  | Snapshot index                                           |
| `field`     | str  | Field name: `"p"`, `"B"`, etc.                           |
| `coord`     | str  | Component: `"scalar"`, `"R"`, `"Z"`, `"phi"`            |
| `fcoords`   | str  | Flux coordinate system                                   |
| `points`    | int  | Radial resolution                                        |
| `full_fft`  | bool | `True` = full two-sided FFT; `False` = mirrored spectrum |

Returns `(m_modes, psi_norm, spectrum)`:
- `m_modes`: 1-D int array of poloidal mode numbers
- `psi_norm`: 1-D float array of length `points`
- `spectrum`: 2-D float array, shape `(len(m_modes), points)`

```python
m, psi, spec = compute_poloidal_spectrum("sparc_1425", 1, "p", points=200)
# spec[i, j] = amplitude of mode m[i] at psi_norm[j]
```

---

### `compute_standard_spectra(case_dir, time_idx, fcoords="pest", points=200, full_fft=False)`

Compute poloidal spectra for the default set of four fields: pressure (p),
and the R, Z, and toroidal components of B.

| Parameter  | Type | Description           |
|------------|------|-----------------------|
| `case_dir` | path | M3D-C1 case directory |
| `time_idx` | int  | Snapshot index        |
| `fcoords`  | str  | Flux coordinate system |
| `points`   | int  | Radial resolution     |
| `full_fft` | bool | Full two-sided FFT    |

Returns `dict` with keys `"p"`, `"br"`, `"bz"`, `"bphi"`. Each value is a
`(m_modes, psi_norm, spectrum)` tuple. Failed fields are absent.

```python
spectra = compute_standard_spectra("sparc_1425", 1, points=200)
m, psi, spec_p = spectra["p"]
```
