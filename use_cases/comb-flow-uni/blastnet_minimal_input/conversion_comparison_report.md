# Conversion Comparison Report

**Generated:** 2026-04-24  
**Our output:** `minimal_input/well_output/lifted_hydrogen_jet_traj_5000.hdf5`  
**Reference:** `holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5`  
**Source data:** `blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000/`

---

## Summary

The conversion script produces a structurally correct WELL HDF5 file, but contains one critical data bug (incorrect array transposition) and three metadata mismatches relative to the holdout reference.

---

## Matches

| Item | Detail |
|---|---|
| `dimensions/time` | 201 steps, 0 to 1.0×10⁻³ s — exact match |
| `dimensions/x` | 1600 points, ~7.8×10⁻⁶ to 0.025 m — exact match |
| `dimensions/y` | 2000 points, ~−0.015 m upward — exact match |
| `boundary_conditions/x_open/mask` | Shape (1600,), True at indices 0 and 1599 — values identical |
| `boundary_conditions/y_open/mask` | Shape (2000,), True at indices 0 and 1999 — values identical |
| All dataset shapes | `(1, 201, 1600, 2000)` for t0 scalars, `(1, 201, 1600, 2000, 2)` for velocity — correct |
| Group/attribute structure | All groups present with correct hierarchy |

---

## Differences

### 1. Field data values — critical (transposition bug)

All spatial field arrays are incorrect due to a wrong reshape in the data-loading step.

**Root cause:** Grid files (`X_m.dat`, `Y_m.dat`) are stored in `(Ny=2000, Nx=1600)` Fortran-style order, so extracting 1D coordinates requires `reshape(2000, 1600)`. However, the data field files (`*.dat`) are stored in `(Nx=1600, Ny=2000)` C order and should be reshaped directly as `reshape(1600, 2000)` with no transpose. The script incorrectly applied the grid file logic to data files, using `reshape(Ny, Nx).T` which shuffles x and y content.

**Observed errors:**

| Field | Max absolute error | Mean relative error |
|---|---|---|
| `pressure` | 381 Pa | 2.1×10⁻⁴ |
| `density` | 0.241 kg/m³ | 3.1×10⁻¹ |
| `temperature` | 2322 K | 3.1×10⁻¹ |
| `mass_fraction_h2` | 0.650 | 1.82 |
| `mass_fraction_o2` | 0.233 | 4.2×10⁻¹ |
| `mass_fraction_h2o` | 0.235 | 1.64 |
| `mass_fraction_oh` | 0.024 | 1.83 |
| `velocity` | 163 m/s | 8.0×10⁻¹ |

**Fix:** Replace `raw.reshape(Ny, Nx).T` with `raw.reshape(Nx, Ny)` in the data-loading loop.

---

### 2. Species mass fraction field names

| Our name | Reference name |
|---|---|
| `Y_H` | `mass_fraction_h` |
| `Y_H2` | `mass_fraction_h2` |
| `Y_O` | `mass_fraction_o` |
| `Y_O2` | `mass_fraction_o2` |
| `Y_OH` | `mass_fraction_oh` |
| `Y_H2O` | `mass_fraction_h2o` |
| `Y_HO2` | `mass_fraction_ho2` |
| `Y_H2O2` | `mass_fraction_h2o2` |

**Fix:** Update `SCALAR_MAP` in the script to use the `mass_fraction_<species>` naming convention.

---

### 3. Root attributes — extra fields

| Attribute | Ours | Reference |
|---|---|---|
| `simulation_parameters` | `['Re_jet']` | `[]` |
| `Re_jet` | `5000.0` | not present |

`Re_jet` is available in `info.json` but the reference does not promote it to a root attribute.  
**Fix:** Set `simulation_parameters = []` and omit the `Re_jet` root attribute.

---

### 4. Boundary condition mask dtype — minor

| Dataset | Ours | Reference |
|---|---|---|
| `boundary_conditions/*/mask` | `bool` | `int8` |

Values are identical after casting. Functionally equivalent; no impact on downstream reads.  
**Fix (optional):** Write masks as `int8` to match the reference exactly.

---

## Required script changes

```python
# 1. Fix data reshape (convert_to_well_format_v1.py, data-loading loop)
# Before:
arr = load_dat(paths[bname]).reshape(Ny, Nx).T
# After:
arr = load_dat(paths[bname]).reshape(Nx, Ny)

# 2. Fix species names in SCALAR_MAP
SCALAR_MAP = {
    ...
    "YH":    "mass_fraction_h",
    "YH2":   "mass_fraction_h2",
    "YO":    "mass_fraction_o",
    "YO2":   "mass_fraction_o2",
    "YOH":   "mass_fraction_oh",
    "YH2O":  "mass_fraction_h2o",
    "YHO2":  "mass_fraction_ho2",
    "YH2O2": "mass_fraction_h2o2",
}

# 3. Fix root attributes
f.attrs["simulation_parameters"] = []   # empty, no Re_jet
# remove: f.attrs["Re_jet"] = ...

# 4. (Optional) BC mask dtype
mask = np.zeros(size, dtype=np.int8)
mask[0] = mask[-1] = 1
```
