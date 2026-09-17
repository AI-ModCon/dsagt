Now I have both files. Let me compare them systematically.

---

## Comparison: `convert_to_well_format_v4.py` vs `holdout/convert_to_well_format.py`

---

### Design Philosophy

| Dimension | v4 (ours) | Holdout |
|---|---|---|
| Scope | Single trajectory, explicit path arg | Batch: scans all datasets under `blastnet_data/` |
| Dataset support | 2 datasets | 7 datasets (`BC_MAPPINGS` + `canonical_decaying_hit` special handler) |
| Interface | `traj_dir` positional arg | `--dataset`, `--base-path`, `--output-path` |
| Resume/skip | None | Checks if output file exists and is valid before re-running |
| Filtering | None | `--only-trajectories`, `--only-corrupted-missing` |

---

### Logic / Correctness Issues

**1. UX/UY velocity swap — holdout has it, v4 fixed it.**

The holdout reads data file paths from `info.json` local entry filename fields (lines 975–1011) for the multi-file time-series case. Since `nonreacting_channel_flow`'s `info.json` has `"UX_ms-1 filename": "./data/UY_ms-1_id0021.dat"`, the holdout script has the same swap bug that v3 had and v4 fixed. Our v4 constructs paths as `data/{varname}_id{sid:04d}.dat`, bypassing the mislabeled fields entirely.

**2. Empty list attributes — holdout fails in h5py 3.x.**

The holdout writes:
```python
f.attrs["simulation_parameters"] = []        # line 449
bc_group.attrs["associated_fields"] = []     # line 290
```
In h5py 3.x, writing an empty Python list as an attribute raises `TypeError: No conversion path for dtype: dtype('<U1')`. Our v4 uses `_str_attr()` which writes empty lists as `np.array([], dtype=np.float64)`.

**3. Grid file path — holdout hardcodes `grid/` subdir, misses lifted_hydrogen_jet.**

```python
grid_dir = traj_dir / 'grid'   # holdout line 147
```
For `lifted_hydrogen_jet`, grid files (`X_m.dat`, `Y_m.dat`) live in the trajectory root, not in a `grid/` subdirectory. The holdout's `load_grid_coordinates()` would return `None` and fall back to `np.linspace(0, 1, dim_size)` — wrong coordinates. Our v4 reads the grid path from `info.json["global"]["grid"]`, which is correct for both datasets.

**4. Field name key truncation — holdout works by accident.**

The holdout strips variable names to just the first `_`-separated token:
```python
field_name = field_base.split('_')[0]   # P_Pa -> P, RHO_kgm-3 -> RHO
```
Then looks up `BLASTNET_TO_WELL_FIELD_MAP`. This works because the map has both `'P'` and `'P_Pa'` as separate keys. But it's fragile — a dataset with a field like `YH2` gets truncated to `YH2` (no underscore), which is fine, but it relies on the fallback keys being in the map. Our v4's `SCALAR_MAP` uses the full BlastNet variable names as keys (`"RHO_kgm-3"`, `"P_Pa"`, etc.) directly from `info.json`, which is more explicit.

**5. `time_varying` attribute on scalar fields — holdout is conditional, v4 is always True.**

```python
time_varying = (nt > 1)   # holdout line 1227
ds.attrs["time_varying"] = time_varying
```
The reference format always sets `time_varying = True` for `t0_fields` and `t1_fields` datasets, regardless of `nt`. Our v4 follows the reference. For single-snapshot datasets this is a minor schema divergence.

---

### Efficiency

**Memory: holdout loads everything into RAM; v4 writes one step at a time.**

The holdout accumulates all timesteps for all fields before writing:
```python
# holdout lines 1022-1073
for field_name, file_path in traj_files.items():
    all_data = []
    for fpath in sorted(time_series_files[field_name]):
        file_data = np.fromfile(...).reshape(1, nx, ny, nz)
        all_data.append(file_data)
    data = np.concatenate(all_data, axis=0)   # (n_steps, Nx, Ny, Nz) in RAM
    fields_data[field_name] = data            # kept in RAM
# then writes fields_data to HDF5 at once
```

For `nonreacting_channel_flow` (21 steps × 1536×384×1024, pressure + 3D velocity):
- **Holdout**: 4 fields × 21 steps × 1536×384×1024 × 4 bytes ≈ **200 GB RAM**
- **v4**: 1 step at a time, 4 fields × 1 × 1536×384×1024 × 4 bytes ≈ **9.5 GB RAM** (~21× less)

The holdout's `canonical_decaying_hit` special handler (lines 347–622) does use per-trajectory incremental writing with pre-allocated buffers and explicit `f.flush()` every 2 steps — that pattern is good, but the general-dataset path doesn't use it.

**Chunking: holdout's general path has none; v4 always uses timestep-aligned chunks.**

```python
# v4 — always chunked
t0_grp.create_dataset(wname, shape=field_shape, dtype=np.float32,
    chunks=tuple([1, 1] + list(Nxyz)))
```
The holdout writes the full in-memory array directly (`create_dataset(..., data=well_data)`), which creates an unchunked contiguous dataset. For large 3D arrays, unchunked datasets are fast to write once but slow to read back slice-by-slice (e.g., reading a single timestep requires scanning the whole file). Chunks of `(1,1,Nx,Ny,Nz)` make per-timestep reads O(1 chunk).

---

### Summary

| Issue | v4 | Holdout |
|---|---|---|
| UX/UY swap for channel flow | Fixed (constructs paths directly) | Present (follows mislabeled JSON fields) |
| Empty attrs in h5py 3.x | Fixed (`_str_attr` helper) | Fails at runtime |
| Grid coordinates for lifted_hydrogen_jet | Correct (reads path from info.json) | Wrong (hardcodes `grid/` subdir, falls back to `linspace`) |
| Peak memory for large 3D datasets | O(1 step) ≈ 9.5 GB | O(all steps) ≈ 200 GB |
| HDF5 chunking | Always (timestep-aligned) | None in general path |
| `time_varying` attr | Always True | Conditional on nt > 1 |
| Dataset coverage | 2 datasets | 7 datasets |
| Batch/resume/skip | None | Full (existence check, corrupted detection) |
| Time key robustness | `"time"` only | Also tries `"time [s]"`, `time_step * dt` |

**The holdout is a more complete production batch tool** (7 datasets, resume, filters), but has correctness bugs and a severe memory scaling problem for 3D datasets. **v4 is correct and memory-efficient** but only handles 2 datasets and runs one trajectory at a time.
