# XGC Field Reference

## Mesh (xgc.mesh.bp) — static, one file per case

| Variable | Shape | Description |
|----------|-------|-------------|
| `rz` | [N, 2] | (R, Z) node coordinates in meters |
| `nd_connect_list` | [C, 3] | Triangle connectivity (0-based node indices) |
| `psi` | [N] | Poloidal magnetic flux at each node (Wb/rad) |
| `region` | [N] | Node region code (see table below) |
| `nextnode` | [N] | Next node along the magnetic field line |
| `n_n` | scalar | Total node count |
| `n_t` | scalar | Total triangle count |
| `nsurf` | scalar | Number of flux surfaces |

**Region codes**

| Code | Meaning |
|------|---------|
| 1 | Core (inside separatrix) |
| 2 | Edge / separatrix |
| 3 | Scrape-off layer (SOL) |
| 100 | Wall / boundary nodes |

## Fields (xgc.3d.NNNNN.bp) — one file per timestep

| Variable | Shape | Description |
|----------|-------|-------------|
| `dpot` | [nphi, N] or [N, nphi] | Perturbed electrostatic potential (V) |
| `eden` | [nphi, N] or [N, nphi] | Electron density perturbation |
| `iden` | [nphi, N] or [N, nphi] | Ion density perturbation |
| `pot0` | [N] | Flux-surface-averaged potential (V) |
| `nphi` | scalar | Number of toroidal planes |
| `time` | scalar | Physical time (s) |

## Fluid moments (xgc.f3d.NNNNN.bp) — subset of timesteps

| Variable | Shape | Description |
|----------|-------|-------------|
| `e_den` | [nphi, N] | Electron density |
| `e_T_para` | [nphi, N] | Electron parallel temperature (eV) |
| `e_T_perp` | [nphi, N] | Electron perpendicular temperature (eV) |
| `e_u_para` | [nphi, N] | Electron parallel flow velocity |
| `i_den` | [nphi, N] | Ion density |
| `i_T_para` | [nphi, N] | Ion parallel temperature (eV) |
| `i_T_perp` | [nphi, N] | Ion perpendicular temperature (eV) |
| `i_u_para` | [nphi, N] | Ion parallel flow velocity |
| `dpot` | [nphi, N] | Perturbed potential (same as xgc.3d) |

## Axis order

XGC field arrays appear in two layouts depending on the simulation:

| Case | Layout | Notes |
|------|--------|-------|
| ITER, KSTART | `[nphi, n_nodes]` | phi-first (most common) |
| NSTX | `[n_nodes, nphi]` | phi-last (transposed) |

`xgc_preprocess.py` normalizes all outputs to `[nphi, n_nodes]` float32.

## Dataset-level conventions

Each preprocessed case produces:
- `mesh.npz` — rz, conn, psi, region, nextnode
- `step_NNNNN.npz` — all fields at that step, shape `[nphi, n_nodes]`
- `meta.json` — n_nodes, nphi, step list, field_availability per field

Node feature vector assembled by `xgc_dataset.py`:
```
x = [R, Z, psi, region_oh(4), field_1, ..., field_F]   shape [N, 7+F]
```
Target `y = x[:, 7:]` (field values only at target timestep).
