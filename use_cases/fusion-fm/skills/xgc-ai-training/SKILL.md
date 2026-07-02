---
name: xgc-ai-training
description: >
  Convert XGC plasma turbulence simulation data (ADIOS2 BP5 format) into
  GNN-ready npz files and a PyTorch Dataset for AI/surrogate model training.
  Use when the user asks to preprocess XGC data, create training datasets from
  XGC simulations, or prepare fusion simulation data for machine learning.
---

# XGC → AI Training Pipeline

Converts raw XGC simulation directories into graph-structured training data
for MeshGraphNets-style surrogate models of plasma turbulence.

**Variable reference:** [`references/xgc_fields.md`](references/xgc_fields.md)

---

## Pipeline Checklist

Copy and track progress:

```
- [ ] 1. Check XGC case structure (pre-flight)
- [ ] 2. Summarize physics content (audit)
- [ ] 3. Preprocess BP files → npz
- [ ] 4. Validate preprocessed output (post-flight)
- [ ] 5. Confirm dataset class is usable
```

---

## Stage 1 — Check XGC Structure (check)

Run **before** preprocessing. Verifies mesh file, counts 3d/f3d steps, detects
nphi and axis order, flags inconsistencies.

```bash
python scripts/check_xgc_structure.py <case_dir> [--output audit/step1_pre.json]
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `case_dir` | yes | XGC simulation directory |
| `--output` | no | Save JSON report to file (audit trail) |

**Expect:** `"status": "ok"` and all checks passed before proceeding.  
**Watch for:** missing `xgc.mesh.bp`, axis order (`first` vs `last`), f3d alignment.

---

## Stage 2 — Physics Summary (operation → audit)

Reads mesh and field files; emits structured JSON with node counts, field
names, physical parameters, and time range.

```bash
python scripts/xgc_summarize.py <case_dir> [--output audit/xgc_summary.json]
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `case_dir` | yes | XGC simulation directory |
| `--output` | no | Save JSON report to file |

**Key output fields:** `mesh.n_nodes`, `mesh.n_edges`, `fields_3d.phi_field_names`,
`fields_f3d.phi_field_names`, `physical_params.sml_dt`.

---

## Stage 3 — Preprocess BP → npz (operation)

Reads all `xgc.3d.*.bp` (and `xgc.f3d.*.bp` where available). Normalizes
axis order to `[nphi, n_nodes]` float32. Writes `mesh.npz`, `step_NNNNN.npz`,
and `meta.json` under `out_dir`.

```bash
python scripts/xgc_preprocess.py <case_dir> <out_dir> \
    [--fields_3d dpot,eden,iden] \
    [--fields_f3d e_den,e_T_para,e_T_perp,e_u_para,i_den,i_T_para,i_T_perp,i_u_para] \
    [--no_f3d] \
    [--steps 10,20,30] \
    [--overwrite] \
    [--output audit/step3_op.json]
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `case_dir` | yes | — | XGC simulation directory |
| `out_dir` | yes | — | Output directory (created if absent) |
| `--fields_3d` | no | `dpot,eden,iden` | Fields to read from xgc.3d files |
| `--fields_f3d` | no | all 8 fluid moments | Fields to read from xgc.f3d files |
| `--no_f3d` | no | false | Skip f3d files entirely |
| `--steps` | no | all | Comma-separated step indices to process |
| `--overwrite` | no | false | Overwrite existing npz files |
| `--output` | no | — | Save JSON status to file |

**Output layout:**
```
out_dir/
  mesh.npz          rz[N,2], conn[C,3], psi[N], region[N], nextnode[N]
  meta.json         n_nodes, nphi, steps, field_names, field_availability
  step_00002.npz    dpot[nphi,N], eden[nphi,N], ...
  step_00004.npz    ...
```

**Typical run times (KSTART, 45K nodes, all 99 steps):** ~5 min on login node.

---

## Stage 4 — Validate Preprocessed Output (check)

Run **after** preprocessing. Spot-checks 5 step files for shape, dtype, field
completeness, and time monotonicity.

```bash
python scripts/check_xgc_preprocessed.py <out_dir> [--output audit/step3_post.json]
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `out_dir` | yes | Directory produced by `xgc_preprocess.py` |
| `--output` | no | Save JSON report to file |

**Expect:** `"status": "ok"`, all shapes `[nphi, n_nodes]`, dtype `float32`.

---

## Stage 5 — Dataset Class (library)

`scripts/xgc_dataset.py` provides `XGCGraphDataset`, a PyTorch Dataset
subclassing `BaseCFDGraphDataset` from the MATEY project.

On first use it builds a cached `topology.pt` (edge_index, edge_attr, static
node context) then reads npz files at getitem time — efficient for large meshes.

```python
from xgc_dataset import XGCGraphDataset, build_datasets

# Single case
ds = XGCGraphDataset(
    "/data/kstart_npz",
    field_names=["dpot", "e_den", "e_T_para", "e_T_perp"],
    n_steps=2,
    leadtime_max=5,
    split="train",
    train_val_test=[0.7, 0.15, 0.15],
)
sample, bcs = ds[0]
# sample.x     → [N, n_steps, 7+F]
# sample.y     → [N, F]
# sample.pos   → [N, 2]  (R, Z)

# Multi-case
splits = build_datasets(
    ["/data/kstart_npz", "/data/nstx_npz"],
    field_names=["dpot"],
    n_steps=1,
    leadtime_max=1,
)
train_ds = splits["train"]
```

**Smoke test:**
```bash
python scripts/xgc_dataset.py <out_dir> --n_steps 1 --leadtime_max 1
```

---

## Decisions the Agent Must Confirm With the User

1. **Which fields to include** — at minimum `dpot`; add fluid moments if f3d
   files are present and downstream model needs them.
2. **Which steps to process** — all (default) or a subset for a quick test.
3. **Output directory** — where the npz files will be written.
4. **`n_steps` and `leadtime_max`** — depend on the downstream GNN architecture.

---

## Notes

- ITER (1.28M nodes) is large; preprocessing all 99 steps takes ~30 min.
  Use `--steps 10,20,30` for a fast sanity check first.
- NSTX has axis order `[n_nodes, nphi]`; `xgc_preprocess.py` normalizes
  automatically (verified by `check_xgc_preprocessed.py`).
- f3d files in KSTART appear every 10 steps; 3d every 2 steps. Steps without
  f3d will only have `dpot`, `eden`, `iden`.
- `meta.json` records `field_availability` per field so the dataset class can
  automatically filter to steps where all requested fields are present.
