# BlastNet → WELL Conversion: Development Summary

**Date:** 2026-04-24  
**Working directory:** `minimal_input/`  
**Datasets converted:** `lifted_hydrogen_jet`, `nonreacting_channel_flow`

---

## Overview

Four conversion script versions were developed iteratively. Each version was validated against holdout reference files using a dedicated checker script. The table below summarises the progression.

| Version | Script | Status | Datasets |
|---|---|---|---|
| v1 | `convert_to_well_format_v1.py` | Fails (3 bugs) | `lifted_hydrogen_jet` 2D only |
| v2 | `convert_to_well_format_v2.py` | Passes | `lifted_hydrogen_jet` 2D only |
| v3 | `convert_to_well_format_v3.py` | Partial (velocity swap) | 2D + 3D |
| v4 | `convert_to_well_format_v4.py` | Passes | 2D + 3D |

---

## Version History

### v1 — `convert_to_well_format_v1.py`

Initial implementation targeting `lifted_hydrogen_jet/hydrogen-jet-5000`.

**Three bugs found on comparison against holdout:**

1. **Data transpose (critical):** `.dat` files were reshaped as `(Ny, Nx).T` based on the grid file layout. But data files use a different storage order to grid files — they are stored as `(Nx, Ny)` C-order directly, requiring no transpose. This caused all field values to be scrambled. Errors ranged from `max_abs=381 Pa` for pressure to `max_abs=163 m/s` for velocity.

2. **Species field names:** Species mass fractions were named `Y_H`, `Y_H2`, etc. The reference convention is `mass_fraction_h`, `mass_fraction_h2`, etc.

3. **Root attributes:** `Re_jet: 5000.0` was written as a root attribute and `simulation_parameters` was set to `['Re_jet']`. The reference has `simulation_parameters: []` and no `Re_jet` attribute.

Minor: BC masks written as `bool` dtype; reference uses `int8`.

---

### v2 — `convert_to_well_format_v2.py`

Fixes all three bugs from v1. Passes exact match against `lifted_hydrogen_jet` holdout.

**Changes from v1:**
- Data reshape changed from `.reshape(Ny, Nx).T` to `.reshape(Nx, Ny)` — no transpose
- Species names updated to `mass_fraction_<species>` convention
- Root attributes: `simulation_parameters=[]`, `Re_jet` removed
- BC mask dtype changed from `bool` to `int8`
- `main()` updated to accept an arbitrary `traj_dir` positional argument and derive the output filename from the parent directory name and trajectory subdirectory name

**Limitation:** hardcoded for 2D datasets only; cannot handle `nonreacting_channel_flow`.

---

### v3 — `convert_to_well_format_v3.py`

Extended to support 3D datasets and the `nonreacting_channel_flow` family. All structural elements pass; one data bug remains.

**Changes from v2:**
- `n_spatial_dims` auto-detected from `len(Nxyz)` in `info.json`
- Grid loading generalised: if the grid file contains exactly `N` elements (matching the 1D coordinate count), it is used directly; otherwise the 2D `(Ny, Nx)` Fortran-style layout from v2 is applied
- Time array generalised: uses global `time-step snapshot [s]` if present; otherwise reads per-snapshot `"time"` values from `local` entries (required for channel flow, where IDs start at 21 and physical times are non-uniform)
- BCs made configurable via a `BC_CONFIGS` dict keyed by dataset name, supporting `PERIODIC` (all-True mask), `WALL` (endpoint mask), and `OPEN` (endpoint mask)
- `dim_varying` attribute sized to `n_spatial_dims` rather than always `[True, True]`
- `dataset_name` derived from `traj_dir.parent.name` rather than hardcoded

**Bug found:** velocity components UX and UY were swapped for the channel flow dataset. The `info.json` for `nonreacting_channel_flow` contains mislabeled filename entries:
```
"UX_ms-1 filename": "./data/UY_ms-1_id0021.dat"   ← wrong
"UY_ms-1 filename": "./data/UX_ms-1_id0021.dat"   ← wrong
```
The v3 script resolved file paths through the JSON labels, so it read the UY file as UX and vice versa. UZ was unaffected. Pressure was exact.

---

### v4 — `convert_to_well_format_v4.py`

Fixes the velocity swap. Passes exact match against all three holdout references tested.

**Change from v3:**
- Data file paths are now constructed directly as `data/{varname}_id{id:04d}.dat`, bypassing the `info.json` filename fields entirely. This matches reference behaviour and is safe for all known datasets: for `lifted_hydrogen_jet` the JSON labels are correct, so constructed and JSON-derived paths are identical; for `nonreacting_channel_flow` the constructed paths bypass the mislabeled entries.

**Validated datasets:**
- `lifted_hydrogen_jet/hydrogen-jet-5000` — exact match (201 steps, 11 scalar fields + 2D velocity)
- `lifted_hydrogen_jet/nonreacting-hydrogen-jet-5000` — exact match (201 steps, 5 scalar fields + 2D velocity)
- `nonreacting_channel_flow/channelflow-dns-Re544-eq-p021-041` — exact match (21 steps, pressure + 3D velocity)

---

## Key Technical Findings

| Finding | Detail |
|---|---|
| Grid vs data storage order | Grid files for `lifted_hydrogen_jet` use `(Ny, Nx)` Fortran-style layout; data files use `(Nx, Ny)` C-order. Different conventions in the same dataset. |
| 3D grid files are 1D | `nonreacting_channel_flow` grid files contain only the 1D coordinate arrays (1536, 384, 1024 elements) rather than a full 3D field. |
| Time in info.json | `lifted_hydrogen_jet` stores a global `time-step snapshot [s]`; `nonreacting_channel_flow` stores physical time per snapshot in `local` entries, with snapshot IDs starting at 21. |
| Mislabeled filenames | `info.json` for `nonreacting_channel_flow` swaps the UX and UY filename entries. Constructing paths from variable names directly is more robust than trusting the JSON labels. |
| `field_names` ordering | The `field_names` attribute lists the same fields in different orders across candidate and reference files. Since each field is a separately named HDF5 dataset, order is cosmetic and should be compared as a set. |

---

## Checking Results

Output files are validated using `check_well_output.py`, a standalone comparison script that takes a candidate and a reference WELL HDF5 file and checks them at four levels. First, it compares all root attributes key-by-key and flags any that are missing, extra, or have differing values. Second, it checks the full group and dataset structure — any extra or missing items are reported, along with any shape or dtype mismatches. Third, it checks all per-dataset and per-group attributes, treating `field_names` and `spatial_dims` as unordered sets (since field ordering is cosmetic). Fourth, it compares numerical values either fully or via a fast spot-check mode. Full mode loads each dataset entirely and reports `max_abs`, `mean_abs`, and `mean_rel` errors; spot-check mode (`--spot-check`) samples a small number of random points per dataset (default 5, configurable with `--n-points` and `--seed`) and prints the candidate and reference values side-by-side with a `<-- DIFF` marker for any point that falls outside tolerance. Spot-check mode completes in seconds even for datasets in the hundreds of GB, making it suitable for a quick sanity check immediately after conversion, while full mode provides a complete numerical audit. The script exits with code 0 on a clean pass and 1 on any failure, making it straightforward to integrate into a pipeline or batch job.

```bash
# Full check
python3 check_well_output.py <candidate.hdf5> <reference.hdf5>

# Fast spot check (seconds, not minutes)
python3 check_well_output.py <candidate.hdf5> <reference.hdf5> --spot-check

# Spot check with more points and custom tolerances
python3 check_well_output.py <candidate.hdf5> <reference.hdf5> \
    --spot-check --n-points 20 --seed 42 --rtol 1e-4 --atol 1e-6
```
