# Well Data Conversion Pipeline

Small, self-contained sandbox for developing and validating the BlastNet → WELL HDF5 conversion pipeline.

**Approach:**

Use the [DSAGT](https://github.com/AI-ModCon/dsagt) toolkit with Claude CLI to iteratively develop a pipeline to convert Blastnet Well Data into proposed format. This work was done with Claude-Sonnet-4.6 Model. 


## Original Reference docs

This project began with these 2 files as the set of files fed into LLM agent

| File | Description |
|------|-------------|
| [README-blastnet.md](README-blastnet.md) | Dataset layout, WELL output structure, and usage commands for the conversion script |
| [well_format.md](well_format.md) | WELL HDF5 format specification (schema, field shapes, attribute conventions) |

## Scripts

### [convert_to_well_format_v4.py](convert_to_well_format_v4.py)

Final converter. Takes a single BlastNet trajectory directory and writes a WELL HDF5 file. Supports 2D and 3D datasets.

```bash
python3 convert_to_well_format_v4.py <traj_dir> [--output-path DIR] [--output-file PATH] [--dry-run]
```

```bash
# Convert a trajectory
python3 convert_to_well_format_v4.py blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000

# Dry run (no HDF5 written)
python3 convert_to_well_format_v4.py blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 --dry-run

# Explicit output file
python3 convert_to_well_format_v4.py blastnet_data/nonreacting_channel_flow/channelflow-dns-Re544-eq-p021-041 \
  --output-file well_output/nonreacting_channel_flow_traj_eq_041.hdf5
```

### [check_well_output.py](check_well_output.py)

Compares a candidate WELL HDF5 file against a reference — checks structure, field shapes, and numerical values.

```bash
python3 check_well_output.py <candidate> <reference> [--rtol FLOAT] [--atol FLOAT] [--spot-check] [--n-points N] [--seed N]
```

```bash
# Full comparison
python3 check_well_output.py well_output/lifted_hydrogen_jet_traj_5000.hdf5 holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5

# Fast spot-check (sample 10 random points per dataset)
python3 check_well_output.py well_output/lifted_hydrogen_jet_traj_5000.hdf5 holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5 \
  --spot-check --n-points 10
```


## Version history

Four script versions were developed iteratively. See [development_summary_report.md](development_summary_report.md) for details.

| Script | Status | Notes |
|--------|--------|-------|
| [convert_to_well_format_v1.py](convert_to_well_format_v1.py) | Fails (3 bugs) | Initial implementation, 2D only |
| [convert_to_well_format_v2.py](convert_to_well_format_v2.py) | Passes | Bug fixes, 2D only |
| [convert_to_well_format_v3.py](convert_to_well_format_v3.py) | Partial | Added 3D support, velocity swap bug |
| [convert_to_well_format_v4.py](convert_to_well_format_v4.py) | Passes | 2D + 3D, final version |

## Validation reports

| File | Description |
|------|-------------|
| [conversion_comparison_report.md](conversion_comparison_report.md) | v1 vs holdout for `lifted_hydrogen_jet` |
| [channelflow_conversion_report.md](channelflow_conversion_report.md) | v3 vs holdout for `nonreacting_channel_flow` |
| [compare_holdout_conversion.md](compare_holdout_conversion.md) | v4 vs holdout `convert_to_well_format.py` (original script from COMB-FLOW team) |

## MLflow traces

[MLflow](https://mlflow.org/docs/latest/index.html) is an open-source platform for tracking experiments and agent traces. Agent reasoning traces from the development session are stored in [`mlruns/`](mlruns/).

> **Note:** `mlruns/0/meta.yaml` has `artifact_location` pointing to a `pscratch` path from the original machine. Always use `--backend-store-uri` to load traces locally.

### Launch the UI

```bash
# Install mlflow if you don't have it
pip install mlflow

# Run mlflow viewer
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open http://localhost:5000.

### Navigate to traces

1. Select the **Default** experiment in the left sidebar.
2. Click the **Traces** tab (next to Runs).
3. Click any trace row to expand the agent reasoning steps, tool calls, and LLM inputs/outputs.