---
title: Plasma Turbulence Training Data (XGC)
domain: Plasma physics — gyrokinetic turbulence simulation (XGC) training-data prep
summary: >-
  Register the XGC preprocessing scripts as codes and drive them through
  DSAgt with provenance: check an ADIOS2 BP5 simulation case, summarize its
  physics content, preprocess it into GNN-ready npz files, validate the
  output, and reconstruct the pipeline. Advanced, HPC-scale data.
status: published
order: 80
---

# DSAgt Demo: XGC Training-Data Preparation

> **Estimated time:** ~30 minutes on an HPC login node — not a 10-minute demo.
> XGC output is HPC-scale ADIOS2 BP5 data (up to ~1.3M mesh nodes). No case is
> hosted with this demo: point the paths below at your own XGC run. The
> walkthrough needs only `adios2`; the training-ready dataset class in the
> appendix depends on a module that is not publicly available.

This guide prepares training data for machine-learning surrogates of plasma turbulence from raw
[XGC](https://xgc.pppl.gov/) (X-point Gyrokinetic Code) simulation output. The
four command-line scripts in [`skills/xgc-ai-training/`](skills/xgc-ai-training/)
form a check → operate → check pipeline: verify the case structure, summarize
its physics content, preprocess the BP5 files into `[nphi, n_nodes]` float32
npz files, and validate the result. The agent registers the scripts as DSAgt
codes and runs each stage through `dsagt-run`, so every step is recorded in
`trace_archive/` with its audit report. A fifth script exposes the npz files
as a PyTorch `Dataset`; see the appendix.

**Data format:** ADIOS2 BP5, one directory per simulation run, containing
`xgc.mesh.bp` (static 2D poloidal mesh), `xgc.3d.NNNNN.bp` (per-timestep
`dpot`, `eden`, `iden` fields) and, where available, `xgc.f3d.NNNNN.bp` (fluid
moments). See [`references/xgc_fields.md`](skills/xgc-ai-training/references/xgc_fields.md)
for the variable reference. Representative cases:

| Case | Machine | Nodes | nphi | Steps | Has f3d |
|------|---------|-------|------|-------|---------|
| `n560fr_ITER_PFPO_W_Ne` | ITER | 1,277,797 | 32 | 99 | yes (every 2) |
| `n613fr_KSTART_30306_q4_rmp_turbulence` | KSTART | 45,291 | 32 | 99 | yes (every 10) |
| `ti316_NSTX_small_dt_from_ti313` | NSTX | 86,513 | 16 | 999 | no |

## Prerequisites

- DSAgt installed with the `plasma-turbulence` extra
  (`pip install "dsagt[plasma-turbulence] @ git+https://github.com/AI-ModCon/dsagt.git"`),
  which brings `adios2`.
- An agent platform installed and **already authenticated**.
- An XGC case directory of your own (the KSTART case in the table above is the
  smallest of the three and the one the expected values below refer to).

## Setup

```bash
dsagt init
```

At the menu, name the project `xgc-training` and pick your agent; the defaults
are fine for the rest. Then:

```bash
PROJ=~/dsagt-projects/xgc-training
mkdir -p "$PROJ/data"
ln -s /path/to/your/xgc/<case_dir> "$PROJ/data/<case_dir>"   # or copy it
mkdir -p "$PROJ/skills"
cp -r use_cases/plasma_turbulence/skills/xgc-ai-training "$PROJ/skills/"
dsagt start xgc-training                       # mirrors the skill into the agent's native skills dir
```

Replace `<case_dir>` in the prompts below with the case directory name.

## Execution

Paste these prompts one at a time.

### 1. Register the pipeline scripts as codes

```text
Read the xgc-ai-training skill. Register each of its four command-line scripts
(check_xgc_structure.py, xgc_summarize.py, xgc_preprocess.py,
check_xgc_preprocessed.py) as a code, running --help on each to confirm its
interface. Use the skill's parameter tables for the descriptions.
```

**Verify:** `Search the registry for XGC codes.` → four specs under `codes/`.

### 2. Pre-flight check

```text
Run the XGC structure check on data/<case_dir> and save the report to
audit/step1_pre.json. Tell me the node count, nphi, axis order, and whether f3d
files are present.
```

**Expect:** `dsagt-run` wraps `check_xgc_structure.py`; the report has
`"status": "ok"` with all checks passed. For the KSTART case: 45,291 nodes,
nphi 32, f3d present every 10 steps.

### 3. Physics summary

```text
Summarize the physics content of data/<case_dir> and save it to
audit/xgc_summary.json. Then tell me which fields are available in the 3d and
f3d files and what the time step is.
```

**Expect:** `xgc_summarize.py` runs through `dsagt-run`; the agent reports
`mesh.n_nodes`, `fields_3d.phi_field_names`, `fields_f3d.phi_field_names`, and
`physical_params.sml_dt`.

### 4. Preprocess a subset of steps

```text
Preprocess data/<case_dir> into data/<case_dir>_npz with fields dpot, eden, iden
from the 3d files and all fluid moments from the f3d files. Do steps 10, 20,
and 30 only, save the status to audit/step3_op.json, and confirm the decisions
with me before running.
```

The skill lists the decisions the agent must confirm: fields, steps, output
directory. **Expect:** `xgc_preprocess.py` writes `mesh.npz`, `meta.json`, and
one `step_NNNNN.npz` per selected step, all `[nphi, n_nodes]` float32.

### 5. Post-flight validation

```text
Validate the preprocessed output in data/<case_dir>_npz and save the report to
audit/step3_post.json. Report any shape, dtype, or time-monotonicity problems.
```

**Expect:** `check_xgc_preprocessed.py` returns `"status": "ok"`.

### 6. Reconstruct the pipeline

```text
Reconstruct the preprocessing pipeline from the execution records as a bash
script whose case directory is a variable at the top, so it can be rerun on the
ITER and NSTX cases.
```

## Post-Conditions

1. Code registry contains four XGC specs (`codes/check-xgc-structure/`,
   `codes/xgc-summarize/`, `codes/xgc-preprocess/`, `codes/check-xgc-preprocessed/`).
2. `audit/` holds `step1_pre.json`, `xgc_summary.json`, `step3_op.json`, and
   `step3_post.json`, each with `"status": "ok"`.
3. `data/<case_dir>_npz/` contains `mesh.npz`, `meta.json`, and three step files.
4. `trace_archive/` holds one execution record per stage.
5. A reconstructed pipeline script replays steps 2–5 against a parameterized case directory.
6. MLflow traces (in the serverless `mlflow.db` store) capture the session —
   `dsagt traces xgc-training`.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Native discovery of a project skill placed before `dsagt start` | 1 |
| Registering several scripts as codes from a skill's documentation | 1 |
| Registry search | 1 |
| Paired check / operate / check execution with audit reports | 2–5 |
| Code execution with provenance through `dsagt-run` | 2–5 |
| User confirmation of pipeline decisions before execution | 4 |
| Pipeline reconstruction with a parameterized input | 6 |

## Cleanup

```bash
dsagt rm xgc-training -y
```

## Notes

- The ITER case (1.28M nodes) is large; preprocessing all 99 steps takes ~30
  minutes. Use a step subset (as in step 4) for a first pass.
- NSTX stores fields as `[n_nodes, nphi]`; `xgc_preprocess.py` normalizes the
  axis order automatically and `check_xgc_preprocessed.py` verifies it.
- `meta.json` records `field_availability` per field so the dataset class can
  filter to steps where all requested fields are present.

## Appendix: the training-ready dataset class

`scripts/xgc_dataset.py` wraps a preprocessed directory as `XGCGraphDataset`,
a PyTorch `Dataset` for MeshGraphNets-style surrogates. It subclasses
`BaseCFDGraphDataset` from the `graph_datasets` module of the MATEY project,
which is not publicly available, and it needs `torch` and `torch_geometric`.
Without that module the script does not import, so this step is outside the
walkthrough; with it, the step continues the session after step 5:

```text
Run the xgc_dataset.py smoke test on data/<case_dir>_npz with n_steps 1 and
leadtime_max 1, and report the sample tensor shapes.
```

**Expect:** `XGCGraphDataset` builds its cached `topology.pt` and returns
`sample.x` of shape `[N, n_steps, 7+F]`, `sample.y` of `[N, F]`, and
`sample.pos` of `[N, 2]`. The skill's Stage 5 section shows the `Dataset`
constructor and `build_datasets` for multi-case splits.
