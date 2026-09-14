---
title: BlastNet → WELL Conversion
domain: Combustion CFD — BlastNet DNS trajectories to the WELL HDF5 format
summary: >-
  Develop a BlastNet-to-WELL converter from the format documents: author a
  conversion skill that carries the specifications as references, register
  the agent-written converter and a checker as codes, convert a sample
  trajectory with provenance, and iterate against a holdout reference until
  the check passes.
status: published
order: 90
---

# DSAgt Demo: BlastNet → WELL Conversion

> **Estimated time:** ~45–60 minutes with the sample trajectory from the data
> bundle. The agent writes the converter and iterates against a checker, so
> the number of passes varies. Full BlastNet channel-flow cases are hundreds
> of GB and need an HPC node; this demo uses one small lifted-hydrogen-jet
> trajectory cut to three snapshots.

[BlastNet](https://blastnet.github.io/) publishes combustion DNS datasets as
per-trajectory directories of raw float32 arrays plus an `info.json`.
Machine-learning pipelines consume them in the [WELL](https://polymathic-ai.org/the_well/)
HDF5 layout. This walkthrough has the agent build the bridge between the two
from the two documents that define them, then prove it against a reference
file produced upstream. It reproduces the workflow that produced the converter
in [`reference/`](reference/); that development history, with the bugs each
version had, is in
[`reference/development_history.md`](reference/development_history.md).
The walkthrough has been run end to end with Claude Code on Sonnet 4.5.

Folder contents:

| Path | Role in the demo |
|------|------------------|
| [`docs/well_format.md`](docs/well_format.md), [`docs/blastnet_layout.md`](docs/blastnet_layout.md) | the two specifications the agent works from; they become the skill's `references/` |
| [`scripts/check_well_output.py`](scripts/check_well_output.py) | the checker: compares a candidate WELL file to a reference (structure, shapes, values) |
| [`scripts/make_demo_subset.py`](scripts/make_demo_subset.py) | builds the demo data bundle from a full trajectory |
| [`reference/`](reference/) | the converter this workflow produced, its earlier versions, and the validation reports — a reference solution, not an input to the demo |

## Prerequisites

- DSAgt installed (`uv sync --all-groups`) and an agent platform installed and
  **already authenticated** (BYOA — dsagt writes no credentials; the default
  local embedder needs no API key).
- `numpy` and `h5py` importable in the environment `dsagt` runs in
  (`uv sync --all-groups` installs them through the `use-cases` dependency group).

## Setup

```bash
dsagt init
```

At the menu, name the project `blastnet-well` and pick your agent; the defaults
are fine for the rest. Then:

```bash
PROJ=~/dsagt-projects/blastnet-well
# Demo data (one BlastNet trajectory, three snapshots, and its holdout WELL
# reference file) from the DSAgt use-case data folder:
# https://drive.google.com/drive/folders/1RWQAJeHaikIaD7CCf8ciJ71m55S1erp6
curl -L "https://drive.usercontent.google.com/download?id=1xUZhlr6uCahSbOLiLt5wcwMzehdpaUjL&export=download&confirm=t" \
  -o combustion_simulation_data.tar.gz
tar xzf combustion_simulation_data.tar.gz -C "$PROJ"
# creates $PROJ/data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000/
#     and $PROJ/data/holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5
mkdir -p "$PROJ/codes/scripts" "$PROJ/docs"
cp use_cases/combustion_simulation/docs/*.md "$PROJ/docs/"
cp use_cases/combustion_simulation/scripts/check_well_output.py "$PROJ/codes/scripts/"
dsagt start blastnet-well
```

The converter is deliberately not copied in. The agent writes it.

## Execution

Paste these prompts one at a time.

### 1. Author the conversion skill from the specifications

```text
Read docs/well_format.md (the WELL HDF5 format) and docs/blastnet_layout.md
(the BlastNet trajectory layout). Then use the skill-creator skill to author a
project skill named "blastnet-to-well". Its SKILL.md states the mapping rules:
the field-name mapping, which fields go into t0_fields versus t1_fields, how
the grid and time arrays are derived, how boundary conditions are represented,
and which root attributes are required. Copy both documents into the skill's
references/ directory. Under its scripts/ directory write
convert_to_well_format.py: a command-line converter taking a positional
BlastNet trajectory directory and the options --output-file and --dry-run,
reading info.json for dimensions, variables, snapshot ids, and grid paths, and
writing one WELL HDF5 file. The coordinate arrays must be read from the grid
files that info.json names, not generated. Save it with save_skill.
```

**Expect:** `save_skill` writes `<project>/skills/blastnet-to-well/` with a
`SKILL.md`, the two documents under `references/`, and
`scripts/convert_to_well_format.py`, mirrored into the agent's native skills
directory. The rules in `SKILL.md` should cover: scalar fields (pressure,
density, temperature, species mass fractions as `mass_fraction_*`) in
`t0_fields/`, velocity stacked as a vector in `t1_fields/`, per-type mask
groups under `boundary_conditions/`, coordinate arrays from the grid files,
time from `info.json`, and the root attributes `dataset_name`, `grid_type`,
`n_spatial_dims`, `n_trajectories`, `simulation_parameters`.

### 2. Register the converter and the checker as codes

```text
Register two codes. convert-to-well runs
`python skills/blastnet-to-well/scripts/convert_to_well_format.py` with a
positional trajectory directory and the options --output-file and --dry-run.
check-well-output runs `python codes/scripts/check_well_output.py` with
positional candidate and reference files and the options --rtol, --atol,
--spot-check, --n-points, and --seed. Run --help on each first to confirm.
```

**Verify:** `Search the registry for WELL conversion codes.` → both specs under `codes/`.

### 3. Dry run

```text
Do a dry run of convert-to-well on
data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 and tell me the grid
size, the number of snapshots, and which WELL fields it would write.
```

**Expect:** 1600 × 2000 grid, 3 snapshots, eleven `t0_fields` scalars and a
2-component velocity; no HDF5 written.

### 4. Convert the trajectory

```text
Convert data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 to
well_output/lifted_hydrogen_jet_traj_5000.hdf5 with the convert-to-well code.
```

### 5. Check against the holdout reference and iterate

```text
Spot-check well_output/lifted_hydrogen_jet_traj_5000.hdf5 against
data/holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5 with 10 random
points per dataset using the check-well-output code. If anything differs, fix
the converter in the skill, reconvert with the registered code, and check
again. When the spot-check passes, run the full comparison.
```

**Expect:** a first pass that fails on one or more of the pitfalls the
original development hit — all of them are visible in the checker's output:

| Pitfall | Checker symptom |
|---------|-----------------|
| data files reshaped with a transpose | every field value differs, errors of order 10²–10³ |
| species named `Y_H2` instead of `mass_fraction_h2` | datasets only in candidate / only in reference |
| an extra root attribute (`Re_jet`) or a non-empty `simulation_parameters` | root-attribute mismatch |
| boundary masks written as `bool` | dtype mismatch on `boundary_conditions/*/mask` |
| boundary-condition text not parsed (`inflow-outflow`, `pressure outlet`) | `boundary_conditions/` groups missing |
| coordinates generated as a uniform range instead of read from the grid files | none — the grid is uniform, so it passes within tolerance; read the converter, not only the checker output |

Each fix is a new version of the script inside the skill, each reconversion
and check a new record in `trace_archive/`. The loop ends with
`PASS — candidate matches reference exactly`.

### 6. Generate a datacard

```text
Use the datacard-generator skill to write a Level 1 datacard for the converted
WELL file to audit/. Take the values from info.json and the conversion, and
note anything unknown rather than asking.
```

### 7. Reconstruct the pipeline

```text
Reconstruct the conversion and validation pipeline from the execution records
as a bash script, save it as pipeline.sh, and put the trajectory directory in a
variable at the top so it can be rerun on the other BlastNet trajectories. Keep
only the final converter run and its checks.
```

**Expect:** `reconstruct_pipeline(format="bash")` orders the recorded runs by
their input/output files; `pipeline.sh` holds the dry run, the conversion, and
the two checks as plain `python` (or `uv run`) commands with `TRAJ_DIR` at the
top. The script calls the tools directly so it runs outside a DSAgt project.

## Post-Conditions

1. `skills/blastnet-to-well/` exists with a `SKILL.md` stating the mapping rules, both specifications under `references/`, and a converter under `scripts/`.
2. Code registry contains `convert-to-well` and `check-well-output` specs.
3. `well_output/lifted_hydrogen_jet_traj_5000.hdf5` exists and the full checker run reports an exact match to the holdout reference.
4. `trace_archive/` holds every converter and checker run, including the failed checks that drove the fixes.
5. A datacard exists for the converted dataset.
6. `pipeline.sh` replays conversion and validation for a parameterized trajectory directory, calling the tools directly.
7. MLflow traces (in the serverless `mlflow.db` store) capture the session —
   `mlflow ui --backend-store-uri sqlite:///$PROJ/mlflow.db`.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Skill authoring with `skill-creator` and `save_skill`, carrying its source documents as references | 1 |
| Agent-written code from documentation | 1 |
| Code registration (`save_code_spec`) and registry search | 2 |
| Code execution with provenance through `dsagt-run` | 3–5 |
| Check-driven iteration with failed runs on the record | 5 |
| Base-skill use (`datacard-generator`) | 6 |
| Pipeline reconstruction with a parameterized input | 7 |

## Cleanup

```bash
dsagt rm blastnet-well -y
rm combustion_simulation_data.tar.gz
```

## Notes

- [`reference/convert_to_well_format.py`](reference/convert_to_well_format.py)
  is the converter this workflow produced, verified against the holdout file.
  Compare the agent's converter to it after step 5, not before.
- The demo bundle is the first three snapshots of the `hydrogen-jet-5000`
  trajectory (a full trajectory is ~32 GB) with the reference WELL file sliced
  to the same steps, built with
  [`scripts/make_demo_subset.py`](scripts/make_demo_subset.py):

  ```bash
  python3 make_demo_subset.py <traj_dir> <reference.hdf5> <out_dir> --steps 3
  tar czf combustion_simulation_data.tar.gz -C <out_dir> data
  ```

  The subset converts and checks exactly like the full trajectory, since a
  converter enumerates snapshots from `info.json` and every time-varying WELL
  dataset carries time on axis 1.
