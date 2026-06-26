# m3dc1 Python Module — User Guide

The `m3dc1` package provides a Python interface to M3D-C1 simulation data.
It wraps `fpy.py` (the fusion-io Python bindings) and adds flux-surface
analysis functions that rely on the `write_neo_input` C++ executable.

---

## Setup

Three environment variables are required:

```bash
export FIO_INSTALL_DIR=/path/to/fusion-io/install
export PYTHONPATH=$FIO_INSTALL_DIR/lib:$PYTHONPATH
export DYLD_LIBRARY_PATH=$FIO_INSTALL_DIR/lib:$DYLD_LIBRARY_PATH   # macOS
# export LD_LIBRARY_PATH=$FIO_INSTALL_DIR/lib:$LD_LIBRARY_PATH     # Linux
```

`PYTHONPATH` makes `fio_py`, `fpy`, and `m3dc1` importable. `DYLD_LIBRARY_PATH`
(or `LD_LIBRARY_PATH` on Linux) lets the dynamic linker find `libm3dc1.so` and
`libfusionio.so` at runtime. `FIO_INSTALL_DIR` tells `m3dc1` where to find the
`write_neo_input` and `trace` executables; if you prefer, you can instead add
`$FIO_INSTALL_DIR/bin` to `PATH`.

The source directory `fusion_io/` can be used in place of the install tree by
pointing `PYTHONPATH` at it directly — no build step is needed for the pure-Python
parts of `m3dc1`.

---

## Import patterns

All of the following import styles are supported:

```python
import m3dc1 as m1                                  # main namespace
from m3dc1.eval_field import eval_field             # submodule alias
from m3dc1.get_time_of_slice import get_time_of_slice
from m3dc1.get_timetrace import get_timetrace
import m3dc1.time_trace_fast as ttf                 # time trace submodule
```

---

## Opening a simulation

`m3dc1` functions take a `fpy.sim_data` object rather than a filename.  Open
one with:

```python
from fpy import sim_data

sim_eq  = sim_data("path/to/C1.h5", time=0)    # equilibrium / timeslice 0
sim_lin = sim_data("path/to/C1.h5", time=1)    # linear perturbation, slice 1
```

The `time` argument sets the active timeslice.  Use `sim.set_timeslice(n)` to
change it later.  Useful attributes:

| Attribute              | Description                                    |
|------------------------|------------------------------------------------|
| `sim.filename`         | Absolute path to `C1.h5`                       |
| `sim.timeslice`        | Currently active timeslice index               |
| `sim.ntime`            | Total number of timeslices                     |
| `sim.ntor`             | Toroidal mode number                           |
| `sim._all_traces`      | h5py group for `C1.h5/scalars/` (time traces)  |
| `sim._all_attrs`       | Open h5py file handle for direct HDF5 access   |

---

## Core functions

### `eval_field`

Evaluate a field at arbitrary (R, φ, Z) coordinates.

```python
from m3dc1.eval_field import eval_field

psi = eval_field("psi", R, phi, Z, sim=sim_eq, time=0)
BR  = eval_field("B",   R, phi, Z, coord="R",      sim=sim_eq)
Bvec = eval_field("B",  R, phi, Z, coord="vector", sim=sim_eq)
```

`R`, `phi`, `Z` are NumPy arrays (or scalars) in SI units and cylindrical
coordinates.  They are broadcast against each other so mixed shapes work.

| `coord`    | Return shape       | Meaning                                |
|------------|--------------------|----------------------------------------|
| `"scalar"` | `R.shape`          | Scalar value, or magnitude for vectors |
| `"R"`      | `R.shape`          | R-component of a vector field          |
| `"phi"`    | `R.shape`          | φ-component of a vector field          |
| `"Z"`      | `R.shape`          | Z-component of a vector field          |
| `"vector"` | `(3, *R.shape)`    | All three cylindrical components       |

Out-of-domain points return `NaN`.  Field names accepted by `sim.typedict`:
`"psi"`, `"B"`, `"p"`, `"j"`, `"v"`, `"ne"`, `"ni"`, `"te"`, `"ti"`,
`"pe"`, `"pi"`, `"A"`, `"E"`, `"kprad_rad"`.

**Performance note:** `eval_field` calls the C extension once per grid point
in a Python loop.  For large arrays this is the main bottleneck.  Prefer the
pre-computed values from `flux_average` where possible.

---

### `get_time_of_slice`

Return the simulation time of a snapshot.

```python
from m3dc1.get_time_of_slice import get_time_of_slice

t_alfven = get_time_of_slice(1, filename="C1.h5", units="alfven")
t_s      = get_time_of_slice(1, filename="C1.h5", units="mks")
```

| `units`              | Returns                     |
|----------------------|-----------------------------|
| `"alfven"`, `"m3dc1"` | Alfvén times (code units)  |
| `"mks"`, `"s"`       | Physical seconds            |

The MKS conversion reads `n0_norm`, `l0_norm`, `b0_norm`, and `ion_mass` from
the `C1.h5` global attributes.  Returns `nan` if those attributes are absent
(older file versions).

---

### `get_timetrace`

Read a scalar time trace.

```python
from m3dc1.get_timetrace import get_timetrace

t, ke, label, units_str = get_timetrace("ke", sim=sim_eq)
t, gamma, _, _ = get_timetrace("ke", sim=sim_eq, growth=True)
```

`"ke"` is an alias for `"E_K3"` (total kinetic energy).  Any name present in
`C1.h5/scalars/` is valid.

| Parameter   | Effect                                                     |
|-------------|------------------------------------------------------------|
| `growth`    | Return γ(t) = 0.5 · d(ln\|values\|)/dt instead of values   |
| `renorm`    | Divide values by the first non-zero value                  |
| `units`     | `"m3dc1"` (Alfvén) or `"mks"` for the time axis            |

Returns `(time, values, label, units_str)`.

---

## Flux-surface functions

These functions require `write_neo_input` to be installed and findable (see
Setup above).  Each call launches the executable as a subprocess, which
re-opens `C1.h5`, loads the mesh, and traces flux surfaces.  **Expect a few
seconds of startup cost per call** even before the tracing begins.

### `flux_average`

Compute the flux-surface average ⟨f⟩ of a field as a function of ψ_norm.

```python
psin, q_prof = m1.flux_average("q",  sim=sim_eq)
psin, p_prof = m1.flux_average("p",  sim=sim_eq)
psin, ne_prof = m1.flux_average("ne", sim=sim_eq)
```

For the fields below, `write_neo_input` pre-computes the average internally
and the result is read directly — no secondary `eval_field` loop:

| Field arg | neo_input.nc variable |
|-----------|-----------------------|
| `"q"`     | `q`                   |
| `"ne"`    | `ne0`                 |
| `"te"`    | `Te0`                 |
| `"ni"`    | `ni0`                 |
| `"ti"`    | `Ti0`                 |

All other fields are evaluated on the full 3-D flux surface grid with
`eval_field` (slow for large grids) and integrated with the Jacobian.

Key parameters:

| Parameter | Default | Effect                                           |
|-----------|---------|--------------------------------------------------|
| `points`  | 200     | Number of radial surfaces (nr)                   |
| `ntheta`  | 128     | Poloidal points per surface                      |
| `nphi`    | 4       | Toroidal planes; use 1 for axisymmetric fields   |

---

### `get_shape`

Compute Miller geometry parameters of the last closed flux surface.

```python
shape = m1.get_shape(sim_eq)
# {"R0": 1.855, "a": 0.570, "kappa": 1.85, "delta": 0.42}
```

Uses `matplotlib.contour` on a regular psi grid — no `write_neo_input`
needed.  The `res` parameter (default 250) sets the grid resolution per axis.
Returns `{}` on failure.

| Key      | Description                                  |
|----------|----------------------------------------------|
| `"R0"`   | Major radius of LCFS midpoint (m)            |
| `"a"`    | Minor radius (m)                             |
| `"kappa"`| Elongation                                   |
| `"delta"`| Triangularity                                |

---

### `flux_coordinates` and `eigenfunction`

These two functions are designed to be used together.  `flux_coordinates`
runs `write_neo_input` once and caches the flux surface grid; `eigenfunction`
uses that grid to decompose a perturbed field into poloidal modes.

```python
# Step 1 — compute flux coordinates from the equilibrium (runs write_neo_input)
fc = m1.flux_coordinates(sim=sim_eq, points=100, ntheta=128)

# fc.fc.psi_norm  → normalised flux grid, shape (nr,)
# fc.fc.q         → safety factor profile, shape (nr,)

# Step 2 — decompose a perturbed field (runs eval_field loop)
spec = m1.eigenfunction(
    sim=[fc, sim_lin],
    field="p", coord="scalar",
    fourier=True, full_fft=False,
    norm_to_unity=True,
)
# spec shape: (ntheta//2 + 1, nr)  — poloidal mode amplitude vs psi_norm
```

Reuse the same `fc` object for multiple `eigenfunction` calls on different
fields or components — `write_neo_input` only runs once.

`flux_coordinates` parameters:

| Parameter | Default | Effect                                                    |
|-----------|---------|-----------------------------------------------------------|
| `points`  | 200     | Number of radial surfaces (nr)                            |
| `ntheta`  | 128     | Poloidal points; sets max resolvable mode (m ≤ ntheta//2) |
| `phit`    | 0.0     | Toroidal angle (radians) of the cross-section             |

`eigenfunction` parameters:

| Parameter       | Default    | Effect                                            |
|-----------------|------------|---------------------------------------------------|
| `field`         | `"p"`      | Field to decompose                                |
| `coord`         | `"scalar"` | Component: `"scalar"`, `"R"`, `"phi"`, `"Z"`      |
| `fourier`       | `True`     | Return Fourier spectrum; `False` returns raw grid |
| `full_fft`      | `False`    | One-sided amplitude vs full two-sided FFT         |
| `norm_to_unity` | `True`     | Divide spectrum by its global peak                |
| `makeplot`      | `False`    | Show amplitude vs ψ_norm plot                     |

---

## Performance guide

| Function                           | Mechanism                  | Typical cost          |
|------------------------------------|----------------------------|-----------------------|
| `get_time_of_slice`                | h5py read                  | Milliseconds          |
| `get_timetrace`                    | h5py read                  | Milliseconds          |
| `flux_average` (q, ne, te, ni, ti) | Subprocess + nc read       | Seconds               |
| `flux_average` (other fields)      | Subprocess + eval_field loop | Minutes             |
| `get_shape`                        | eval_field on 2-D grid     | Seconds (res=250)     |
| `flux_coordinates`                 | Subprocess                 | Seconds–minutes       |
| `eigenfunction`                    | eval_field loop (ntheta×nr) | Minutes              |

The subprocess cost is dominated by:
1. Process launch and HDF5 file re-open (~1–2 s fixed overhead)
2. Flux surface tracing: scales as `nr × ntheta`

The `eval_field` loop cost scales as `ntheta × nr` Python→C extension calls.
Reduce both by lowering `points` and `ntheta`.  For the pre-computed fields
in `flux_average`, only the subprocess cost matters.

---

## Plotting functions

All plotting functions follow the same pattern — they open a new `sim_data`
internally from a filename, evaluate the requested quantity, and display with
matplotlib.

```python
m1.plot_field("p",    "C1.h5", time=1, lcfs=True)
m1.plot_shape("C1.h5", time=0)
m1.plot_time_trace_fast("ke", "C1.h5")
m1.plot_flux_average("q", "C1.h5", time=0)
m1.plot_gfile("sparc.geqdsk")
```

Pass `save=True` and `savedir="./output/"` to write PNG files instead of
displaying interactively.  `plot_vector_field` requires mayavi and raises
`ImportError` if it is not installed.
