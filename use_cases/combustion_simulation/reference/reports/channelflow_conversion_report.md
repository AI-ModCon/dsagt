# Channel Flow Conversion Report

**Generated:** 2026-04-24  
**Our output:** `well_output_v2/nonreacting_channel_flow_channelflow-dns-Re544-eq-p021-041.hdf5`  
**Reference:** `holdout/well_output/nonreacting_channel_flow_traj_eq_041.hdf5`  
**Source data:** `blastnet_data/nonreacting_channel_flow/channelflow-dns-Re544-eq-p021-041/`  
**Script:** `convert_to_well_format_v3.py`

---

## Summary

The v3 script was extended from v2 to support 3D datasets and the channel flow dataset family. Structure, attributes, coordinates, BC masks, and pressure are all correct. One failure: velocity components UX and UY are swapped due to mislabeled filenames in `info.json`.

---

## What's New in v3 (vs v2)

| Capability | v2 | v3 |
|---|---|---|
| Spatial dimensions | 2D only | 2D and 3D (auto-detected from `len(Nxyz)`) |
| Grid files | 2D `(Ny, Nx)` layout | 1D if file size matches dim size, else 2D fallback |
| Time array | Global `time-step snapshot [s]` only | Also reads per-snapshot `"time"` from `local` entries |
| BCs | Lifted hydrogen jet only | Configurable per dataset via `BC_CONFIGS` dict |
| `dim_varying` attribute | Always `[True, True]` | Sized to `n_spatial_dims` |
| `dataset_name` | Hardcoded | Derived from `traj_dir.parent.name` |

---

## Matches

| Item | Detail |
|---|---|
| Root attributes (5) | `dataset_name`, `grid_type`, `n_spatial_dims=3`, `n_trajectories=1`, `simulation_parameters=[]` — all match |
| Structure | 18 items, all shapes and dtypes correct |
| All group/dataset attributes | Exact match |
| `dimensions/time` | 21 steps, `[24.65 .. 33.9]` — exact |
| `dimensions/x` | 1536 points, `[0.0 .. 25.1]` m — exact |
| `dimensions/y` | 384 points, `[-1.0 .. 1.0]` — exact |
| `dimensions/z` | 1024 points, `[0.0 .. 9.4]` m — exact |
| `boundary_conditions/x_periodic/mask` | Shape (1536,), all ones — exact |
| `boundary_conditions/y_wall/mask` | Shape (384,), True at endpoints only — exact |
| `boundary_conditions/z_periodic/mask` | Shape (1024,), all ones — exact |
| `t0_fields/pressure` | Shape (1, 21, 1536, 384, 1024) — exact |

---

## Failure

### Velocity components UX and UY are swapped

**Observed error** (spot check, 5 points):

| Coord | Candidate | Reference |
|---|---|---|
| `[0, 6, 1166, 84, 128, 2]` (UZ) | `-0.03168` | `-0.03168` ✅ |
| `[0, 7, 147, 255, 168, 0]` (UX) | `-0.12114` | `1.02273` ❌ |
| `[0, 7, 789, 16, 77, 1]` (UY) | `0.19196` | `-0.00345` ❌ |

UZ (index 2) matches; UX (index 0) and UY (index 1) are swapped.

**Root cause:** `info.json` contains mislabeled filename entries for UX and UY:

```json
"UX_ms-1 filename": "./data/UY_ms-1_id0021.dat",
"UY_ms-1 filename": "./data/UX_ms-1_id0021.dat"
```

The v3 script looks up `entry[f"{varname} filename"]` to resolve paths, so it reads the UY file for UX and vice versa.

**Verification:** reading the actual files directly against the reference confirms the correct mapping:

| File | Value at `[0,0,10,5,7]` | Reference velocity index |
|---|---|---|
| `UX_ms-1_id0021.dat` | `7.768e-03` | 0 (UX) ✅ |
| `UY_ms-1_id0021.dat` | `5.657e-06` | 1 (UY) ✅ |
| `UZ_ms-1_id0021.dat` | `-2.032e-04` | 2 (UZ) ✅ |

**Conclusion:** The reference ignores `info.json` filename fields for velocity and constructs paths directly from the variable name and snapshot ID.

---

## Required Fix

Construct file paths from variable name and snapshot ID instead of reading the JSON filename field:

```python
# Before (follows JSON labels — broken for channel flow):
fpath = traj_dir / entry[f"{varname} filename"].lstrip("./")

# After (constructs path directly — matches reference):
fpath = traj_dir / "data" / f"{varname}_id{sid:04d}.dat"
```

This change is safe for all known datasets: the hydrogen jet JSON filename fields are not mislabeled, and the constructed paths produce identical filenames to what the JSON would give.
