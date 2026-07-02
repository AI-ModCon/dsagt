# Fusion Foundation Model — XGC AI Training Use Case

**Domain:** Plasma physics — gyrokinetic turbulence simulation  
**Simulation code:** XGC (X-point Gyrokinetic Code)  
**Data format:** ADIOS2 BP5 (one directory per simulation run)  
**Goal:** Produce GNN-ready npz files and a PyTorch Dataset for training plasma surrogate models

## Data

Three example simulation cases live in `example_xgc_data/`:

| Case | Machine | Nodes | nphi | Steps | Has f3d |
|------|---------|-------|------|-------|---------|
| `n560fr_ITER_PFPO_W_Ne` | ITER | 1,277,797 | 32 | 99 | yes (every 2) |
| `n613fr_KSTART_30306_q4_rmp_turbulence` | KSTART | 45,291 | 32 | 99 | yes (every 10) |
| `ti316_NSTX_small_dt_from_ti313` | NSTX | 86,513 | 16 | 999 | no |

Each case directory contains:
- `xgc.mesh.bp` — static 2D poloidal mesh (R/Z coordinates, triangle connectivity, flux surfaces)
- `xgc.3d.NNNNN.bp` — per-timestep field snapshots (`dpot`, `eden`, `iden`, shape `[nphi, n_nodes]` or `[n_nodes, nphi]`)
- `xgc.f3d.NNNNN.bp` — fluid moments (`e_den`, `e_T_para/perp`, `e_u_para`, `i_*` equivalents) where available

## Skill

See [`skills/xgc-ai-training/SKILL.md`](skills/xgc-ai-training/SKILL.md) for the full agent pipeline.

## Quick Start (manual)

```bash
# 1. Check case structure
python skills/xgc-ai-training/scripts/check_xgc_structure.py example_xgc_data/n613fr_KSTART_30306_q4_rmp_turbulence

# 2. Summarize physics content
python skills/xgc-ai-training/scripts/xgc_summarize.py example_xgc_data/n613fr_KSTART_30306_q4_rmp_turbulence

# 3. Preprocess to npz
python skills/xgc-ai-training/scripts/xgc_preprocess.py \
    example_xgc_data/n613fr_KSTART_30306_q4_rmp_turbulence \
    /tmp/kstart_npz

# 4. Validate output
python skills/xgc-ai-training/scripts/check_xgc_preprocessed.py /tmp/kstart_npz

# 5. Use the dataset in training
python skills/xgc-ai-training/scripts/xgc_dataset.py /tmp/kstart_npz
```

## Variable Reference

See [`skills/xgc-ai-training/references/xgc_fields.md`](skills/xgc-ai-training/references/xgc_fields.md).
