---
title: BlastNet → WELL Conversion
domain: Combustion CFD — BlastNet DNS trajectories to the WELL HDF5 format
summary: >-
  Develop a BlastNet-to-WELL converter from the format documents: author a
  conversion skill that carries the specifications as references, register
  the agent-written converter and a checker as codes, convert a sample
  trajectory with provenance, and compare the result with a reference WELL
  file held back from the agent, fixing the converter until the two match
  exactly.
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

Folder contents:

| Path | Role in the demo |
|------|------------------|
| [`docs/well_format.md`](docs/well_format.md), [`docs/blastnet_layout.md`](docs/blastnet_layout.md) | the two specifications the agent works from; they become the skill's `references/` |
| [`scripts/check_well_output.py`](scripts/check_well_output.py) | the checker: compares a candidate WELL file to a reference (structure, shapes, values) |
| [`scripts/make_demo_subset.py`](scripts/make_demo_subset.py) | builds the demo data bundle from a full trajectory |
| [`reference/`](reference/) | the converter this workflow produced, its earlier versions, and the validation reports — a reference solution, not an input to the demo |

## Prerequisites

- DSAgt installed with the `combustion-simulation` extra
  (`pip install "dsagt[combustion-simulation] @ git+https://github.com/AI-ModCon/dsagt.git"`),
  which brings `h5py`.
- An agent platform installed and authenticated.

## Setup

```bash
dsagt init
```

At the menu, name the project `blastnet-well` and pick your agent; the defaults
are fine for the rest. Then:

```bash
PROJ=~/dsagt-projects/blastnet-well
# From the DSAgt use-case data folder: https://drive.google.com/drive/folders/1RWQAJeHaikIaD7CCf8ciJ71m55S1erp6
# One bundle: the BlastNet trajectory (data/), the two format documents (docs/),
# the checker (skills/check-well-output/scripts/), and the holdout reference (holdout/).
curl -L "https://drive.usercontent.google.com/download?id=1dXFOocZ5Uep5DAmth_xMsYwIP5Gbpv83&export=download&confirm=t" \
  -o combustion_simulation.tar.gz
tar xzf combustion_simulation.tar.gz -C "$PROJ" --exclude='./holdout'
dsagt start blastnet-well
```

The holdout reference stays outside the project until step 5. An agent that reads the
reference while gathering context writes a converter that matches on the first try, and the
pitfall loop of step 5 never runs.

The agent writes the converter. [`reference/`](reference/) holds the converter this
walkthrough produced when it was developed, its earlier versions, and the validation
reports; compare the agent's converter with them when the walkthrough is done.

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
files that info.json names, not generated; a missing grid file is an error.
Save it with save_skill.
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

### 2. Register the checker beside the converter

```text
save_skill registered the converter as a code and its reply gave the command to
run it by. Register the checker as a second code: check-well-output runs `python skills/check-well-output/scripts/check_well_output.py` with
positional candidate and reference files and the options --rtol, --atol,
--spot-check, --n-points, and --seed. Run --help on both codes first to confirm
their options.
```

**Verify:** `Search the registry for WELL conversion codes.` → both specs under `skills/`.

### 3. Dry run

```text
Run the converter's registered code, with its exact command, as a dry run on
data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 and tell me the grid
size, the number of snapshots, and which WELL fields it would write. info.json
says 3 snapshots and 13 variables; tell me if the dry run disagrees.
```

**Expect:** 1600 × 2000 grid, 3 snapshots, eleven `t0_fields` scalars and a
2-component velocity; no HDF5 written.

### 4. Convert the trajectory

```text
Convert data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 to
well_output/lifted_hydrogen_jet_traj_5000.hdf5 by running the converter's registered
code with its exact command.
```

### 5. Check against the holdout reference and iterate

Unpack the reference into the project first:

```bash
tar xzf combustion_simulation.tar.gz -C "$PROJ/data" ./holdout
```

```text
Spot-check well_output/lifted_hydrogen_jet_traj_5000.hdf5 against
data/holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5 with 10 random
points per dataset by running the registered check-well-output code with its exact
command. If anything differs, fix the converter in the skill, reconvert with the
converter's registered code, and check again. When the spot-check passes, run the
full comparison the same way. After the check passes, update the skill's SKILL.md so
its rules match the converter.
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
| `bc_type` written in the source's case (`open`) | attribute mismatch on `boundary_conditions/*/bc_type` (`OPEN` expected) |
| `dataset_name` taken from the trajectory directory (`hydrogen-jet-5000`) instead of the family (`lifted_hydrogen_jet`) | `DIFF 'dataset_name'` on the root attributes |
| the two meshgrid files read in the wrong order | `dimensions/x` values differ while `dimensions/y` matches |
| `dimensions/time` written as snapshot indices (0, 1, 2) or a hard-coded step | `dimensions/time` values differ; the times come from `time-step snapshot [s]` in info.json |
| coordinates generated as a uniform range instead of read from the grid files | none — the grid is uniform, so it passes within tolerance; read the converter, not only the checker output |

Each fix is a new version of the script inside the skill, each reconversion
and check a new record in `trace_archive/`. The loop ends with
`PASS — candidate matches reference exactly`, and the skill's rules agree with
the converter that passed.

### 6. Generate a datacard

```text
Use the datacard-generator skill to write a Level 1 datacard for the converted
WELL file to audit/. Take the values from info.json and the conversion, and
note anything unknown rather than asking. Validate the card with the registered
datacard-validate code.
```

**Expect:** the skill's `introspect.py` runs as the registered `datacard-introspect`
code, so the introspection is an execution record; the card under `audit/` is in the
Genesis template and `datacard-validate` accepts it (`ok: true`).

### 7. Reconstruct the pipeline

```text
Reconstruct the conversion and validation pipeline from the execution records
as a bash script, save it as pipeline.sh, and put the trajectory directory in a
variable at the top so it can be rerun on the other BlastNet trajectories. Keep
the final conversion, the spot check, and the full check.
```

**Expect:** `reconstruct_pipeline` with `format="bash"` and `output="pipeline.sh"`
saves the script into the project and returns the recorded runs in the order they
ran; the agent then edits `pipeline.sh` down to the final conversion and the two
checks as plain commands, with the trajectory directory in one variable at the top
and the output and reference paths derived from it. The script calls the tools
directly so it runs outside a DSAgt project.

### 8. Review the project artifacts

```text
Show me the contents of my project folder in a tree format, with the artifacts dsagt recorded during this session highlighted.
```

**Expect:** a listing of the project directory that marks the execution records in
`trace_archive/`, the reports in `audit/`, the registered codes and installed skills under
`skills/`, the trace store `mlflow.db`, and the session's outputs, with a line on what each
is. The agent may print the tree through a command; the reply then summarizes it.

## Post-Conditions

1. `skills/blastnet-to-well/` exists with a `SKILL.md` whose mapping rules agree with the final converter, the two documents under `references/`, and the converter under `scripts/`.
2. Code registry contains the converter's code, registered by `save_skill` from the skill's script, and `check-well-output`, both under `skills/`.
3. `well_output/lifted_hydrogen_jet_traj_5000.hdf5` exists and the full checker run reports an exact match to the holdout reference.
4. `trace_archive/` holds every converter and checker run, including the failed checks that drove the fixes.
5. A datacard for the converted dataset exists under `audit/`, in the Genesis template, and `datacard-validate` accepts it.
6. `pipeline.sh`, saved by `reconstruct_pipeline`, replays the final conversion and both checks, calling the tools directly; the trajectory directory is the only variable to edit, and the output and reference paths are derived from it.
7. MLflow traces (in the serverless `mlflow.db` store) capture the session —
   `dsagt traces blastnet-well`.

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
| Review of the session's artifacts | 8 |

## Cleanup

```bash
dsagt rm blastnet-well -y
rm combustion_simulation.tar.gz
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
