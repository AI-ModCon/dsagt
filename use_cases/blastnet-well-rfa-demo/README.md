---
title: BlastNet → WELL, Full Capability Demo
domain: Combustion CFD, the BlastNet-to-WELL conversion driven through every DSAgt capability
summary: >-
  The BlastNet-to-WELL conversion run across the whole DSAgt surface: install
  the Well conversion and literature skills from the Genesis catalog, ground a
  combustion knowledge base in the format documents and the dataset's own
  catalog pages, review the literature on terms taken from that base, author
  the conversion skill from it, convert with provenance, measure the result
  with the generic AI-readiness baseline and with domain checks written for
  combustion data, iterate against a held-back reference, record the decisions
  in memory, and end at the datacard, the replayable pipeline, and the trace
  viewer.
status: published
order: 95
---

# DSAgt Demo: BlastNet → WELL, Full Capability Demo

> **Estimated time:** ~90 minutes with the sample trajectory from the data
> bundle, of which about 40 is the agent working. The agent writes the
> converter and iterates against a checker and a set of readiness checks, so
> the number of passes varies. Full BlastNet channel-flow cases are hundreds
> of GB and need an HPC node; this demo uses one small lifted-hydrogen-jet
> trajectory cut to three snapshots.
>
> The walkthrough is long enough to fill one session's context. Around step
> 15 a session that has carried every step so far runs out and compaction
> fails. Start a second session there and carry on: the knowledge base, the
> registered codes, the execution records, and the memories are on disk, so
> the only thing a new session lacks is the conversation, and step 17 then
> recalls across the break rather than within one session.

[BlastNet](https://blastnet.github.io/) publishes combustion DNS datasets as
per-trajectory directories of raw float32 arrays plus an `info.json`.
Machine-learning pipelines consume them in the [WELL](https://polymathic-ai.org/the_well/)
HDF5 layout. This walkthrough has the agent write the converter between the
two formats from the two documents that define them and the papers behind
them, check it against a reference file produced upstream, and measure the
converted data for AI-readiness.

It runs the same conversion as the [BlastNet → WELL](../combustion_simulation/README.md)
walkthrough and drives it through every DSAgt capability: the knowledge base,
skill discovery from an external catalog, the AI-readiness check with domain
checks written for combustion data, explicit and episodic memory, and the
trace viewer. Run that walkthrough for the conversion on its own, and this one
for the whole surface. The converter this workflow produced, its earlier
versions, and the development history with the bugs each version had are in
that use case's
[`reference/`](../combustion_simulation/reference/) folder.

Folder contents:

| Path | Role in the demo |
|------|------------------|
| [`well_format.md`](../combustion_simulation/docs/well_format.md), [`blastnet_layout.md`](../combustion_simulation/docs/blastnet_layout.md) | the two specifications the agent works from; they become the knowledge base's first collection and the skill's `references/` |
| [`check_well_output.py`](../combustion_simulation/scripts/check_well_output.py) | the checker: compares a candidate WELL file to a reference (structure, shapes, values) |
| [`make_demo_subset.py`](../combustion_simulation/scripts/make_demo_subset.py) | builds the demo data bundle from a full trajectory |
| [`reference/sample_well_points.py`](reference/sample_well_points.py) | the point sampler of step 12, as this workflow produced it: a reference solution, not an input to the demo |
| [`reference/species_closure.py`](reference/species_closure.py) | the custom AIDRIN module of step 13, the same way |
| [`../combustion_simulation/reference/`](../combustion_simulation/reference/) | the converter this workflow produced, its earlier versions, and the validation reports |

## Prerequisites

- DSAgt installed with the `combustion-simulation` extra
  (`pip install "dsagt[combustion-simulation] @ git+https://github.com/AI-ModCon/dsagt.git"`),
  which installs `h5py`.
- An agent platform installed and authenticated.
- Network access to `github.com` for the Genesis skill catalog, and to
  `arxiv.org` for the literature of steps 4 and 5.

## Setup

```bash
dsagt init
```

At the menu, name the project `blastnet-well-demo`, pick your agent, keep the
AI-readiness check on, and answer yes to episodic memory; the defaults are
fine for the rest. Then:

```bash
PROJ=~/dsagt-projects/blastnet-well-demo
# From the DSAgt use-case data folder: https://drive.google.com/drive/folders/1RWQAJeHaikIaD7CCf8ciJ71m55S1erp6
# One bundle: the BlastNet trajectory (data/), the two format documents (docs/),
# the checker (skills/check-well-output/scripts/), and the holdout reference (holdout/).
curl -L "https://drive.usercontent.google.com/download?id=1dXFOocZ5Uep5DAmth_xMsYwIP5Gbpv83&export=download&confirm=t" \
  -o combustion_simulation.tar.gz
tar xzf combustion_simulation.tar.gz -C "$PROJ" --exclude='./holdout'
dsagt start blastnet-well-demo
```

The holdout reference stays outside the project until step 14. An agent that reads the
reference while gathering context writes a converter that matches on the first try, and the
pitfall loop of step 14 never runs.

The agent writes the converter. The
[BlastNet → WELL](../combustion_simulation/README.md) use case's `reference/`
folder holds the converter this workflow produced when it was developed, its
earlier versions, and the validation reports; compare the agent's converter
with them when the walkthrough is done.

## Execution

Paste these prompts one at a time. Each is one step.

### 1. Install the catalog skills

```text
List the skill sources enabled for this project, then search the catalog for a
skill that converts simulation data to the Well HDF5 format and a skill that
searches scientific literature. Install both.
```

**Expect:** `list_skill_sources` names `genesis`, the source a fresh project enables. `search_skills` finds `well-convert`, an end-to-end Well conversion workflow, and `literature-search`; `install_skill` writes both under `skills/` and mirrors them into the agent's native skills directory. The prompt names no source, so an agent may sync another catalog before answering, which costs a clone and an index pass.

### 2. Ground the knowledge base

```text
Ingest docs/ into a knowledge collection named "combustion". Then retrieve these two BlastNet catalog pages with curl, convert
each to markdown under docs/reference/, and append them to the same
collection:
https://blastnet.github.io/diluted_partially_premixed_h2air_lifted_flame
https://blastnet.github.io/nonreacting_channel_flow_re544
List the collections when you are done.
```

**Expect:** `kb_ingest` embeds the two format documents into `combustion`, nine chunks between them, and `kb_append` adds each retrieved catalog page, about nineteen more. The pages carry what no format document states: the burner and grid dimensions, the case range, the inlet composition and temperatures, the mechanism, and a DOI for each family. They are retrieved with `curl` because everything downstream reads what lands here, and a platform's own fetch tool may return a model's summary of a page rather than the page.

### 3. Confirm what the base can answer

```text
Search the knowledge base for how the WELL format represents boundary
conditions, what the lifted hydrogen jet configuration is physically, and
which species BlastNet stores for a hydrogen-air case. Tell me which document
each answer came from.
```

**Expect:** `kb_search` returns the boundary groups and their `bc_type`, `mask` and `values` members from `well_format.md`, and the configuration from the catalog page: a 1.92 mm burner, a jet of 65% hydrogen at 400 K, co-flow air at 1100 K, Re 5000 to 11000. The species question is the one to read carefully. The collection gives the mechanism's nine species and does not name which eight BlastNet stores, so a grounded answer says the first and declines the second rather than completing the list.

### 4. Review the literature from the grounded base

```text
Before searching anything, query the knowledge base for what this dataset is:
the flame configuration, the Reynolds number range, the chemical mechanism, and
the target format. List the search terms those answers give you. Then use the
literature-search skill with those terms to find, first, the publications the
catalog pages cite, and second, work from the last two years on machine
learning or foundation models for this flame configuration and for the Well
format. Report the terms you used, the results grouped as foundational and
recent, and which of them you could not have found from my prompt alone.
```

**Expect:** The agent's search terms come off the collection rather than the prompt: the diluted partially-premixed lifted flame in an autoignitive heated coflow, the Reynolds range, the mechanism size, the target format. A live search returns different results on different days, so judge the step by whether the terms came from the knowledge base and whether the agent audits its own result, naming which findings the prompt alone would have reached. The useful result here is a negative one, and a run that states it has done the work.

### 5. Append what it found

```text
Download the open-access PDFs into docs/papers/ and append each to the
combustion collection. Then tell me what the collection can answer now that it
could not before.
```

**Expect:** Each PDF adds tens of chunks, taking the collection to roughly 500. Only what is reachable downloads: this flame's own publication is free to read under no open licence and refuses an automated fetch, while the earlier DNS of it sits on an author's faculty page that no DOI resolver indexes. A run that says which it could not fetch, and what the collection still cannot answer, has read the collection rather than counted it.

### 6. Map the fields

```text
Use the well-convert skill's inspect and cross-check steps on
data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000, taking the WELL
schema from the knowledge base. Register whatever you write to read the
trajectory as a code named blastnet-inspect and run it that way. Write the
field mapping table to audit/field_mapping.md: every BlastNet variable, the
WELL field it becomes, which group it belongs to, and its shape. Stop at the
mapping table.
```

**Expect:** The skill's inventory step reads `info.json` for the grid, the snapshot ids and the 13 variables, and `audit/field_mapping.md` names the eleven scalars that become `t0_fields`, the two velocity components that stack into `t1_fields/velocity`, and the coordinate and time arrays. The reading is a data operation, so it is a registered code and its run is an execution record; left unregistered it is the one step whose output nothing can reproduce. This is where the species question settles from the data: eight species with a data file, nine in the mechanism, N2 the complement.

### 7. Author the conversion skill

```text
Use the skill-creator skill to author a project skill named
"blastnet-to-well". Its SKILL.md states the mapping rules from
audit/field_mapping.md and the knowledge base: the field-name mapping, which
fields go into t0_fields versus t1_fields, how the grid and time arrays are
derived, how boundary conditions are represented, and which root attributes
are required. Copy docs/well_format.md and docs/blastnet_layout.md into the
skill's references/ directory. Under its scripts/ directory write
convert_to_well_format.py: a command-line converter taking a positional
BlastNet trajectory directory and the options --output-file and --dry-run,
reading info.json for dimensions, variables, snapshot ids, and grid paths, and
writing one WELL HDF5 file. The coordinate arrays must be read from the grid
files that info.json names, not generated; a missing grid file is an error.
Save it with save_skill.
```

**Expect:** `save_skill` writes `skills/blastnet-to-well/` with a `SKILL.md`, both documents under `references/`, and the converter under `scripts/`, mirrored into the agent's native skills directory. It registers the script as a code, deriving the spec from the script's own `argparse` calls, which declares no dependencies and no file roles. The rules should cover the field mapping, the grid and time derivation, the boundary groups, and the root attributes.

### 8. Register the checker

```text
save_skill registered the converter as a code and its reply gave the command to
run it by. Register the checker as a second code: check-well-output runs `python skills/check-well-output/scripts/check_well_output.py` with
positional candidate and reference files and the options --rtol, --atol,
--spot-check, --n-points, and --seed. Run --help on both codes first to confirm
their options.
```

**Expect:** A second code, `check-well-output`, with the candidate and reference as positional inputs. Running `--help` through `dsagt-run` records an execution like any other, so two runs that move no data land in `trace_archive/`. `Search the registry for WELL conversion codes.` should return both specs under `skills/`.

### 9. Dry run

```text
Run the converter's registered code, with its exact command, as a dry run on
data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 and tell me the grid
size, the number of snapshots, and which WELL fields it would write. info.json
says 3 snapshots and 13 variables; tell me if the dry run disagrees.
```

**Expect:** A 1600 × 2000 grid, 3 snapshots, eleven `t0_fields` scalars and a 2-component velocity, with times 0, 5e-6 and 1e-5 s taken from `info.json`. Nothing is written. A dry run still records its declared output file, which has no hash because it was never created.

### 10. Convert the trajectory

```text
Convert data/blastnet_data/lifted_hydrogen_jet/hydrogen-jet-5000 to
data/well_output/lifted_hydrogen_jet_traj_5000.hdf5 by running the converter's registered
code with its exact command.
```

**Expect:** About 500 MB, exit 0. The root attributes carry `dataset_name`, `grid_type`, `n_spatial_dims`, `n_trajectories` and `simulation_parameters`, and every array should match its source `.dat` file. Whether it matches the reference is step 14's question, not this one.

### 11. Run the readiness check on the WELL file

```text
Run the AI-readiness check on data/well_output/lifted_hydrogen_jet_traj_5000.hdf5.
```

**Expect:** The AIDRIN quality baseline refuses it: `HDF5 file has N datasets in an incompatible layout; refusing to flatten into one table`. A WELL file's datasets differ in shape by design, and the count is the converter's rather than the format's; runs have shown 15 and 22. AIDRIN exits 1, so the refusal is on the record as a failure.

### 12. Derive the table the check can read

```text
Write a code named well-sample-points that samples a WELL HDF5 file into a
point table for the readiness checks: a positional WELL file, the options
--output-file, --n-points (default 2000), --seed (default 0) and --trajectory
(default 0), writing a CSV with one row per sampled grid point per snapshot
and one column per field, with the coordinates and the snapshot time beside
them. Put it under the blastnet-to-well skill's scripts/ and register it.
Then run it on data/well_output/lifted_hydrogen_jet_traj_5000.hdf5 to
audit/readiness/well_points.csv and run the AIDRIN quality baseline and
summarize on the result.
```

**Expect:** `save_code_spec` registers `well-sample-points` and its run is an execution record. The table is 6000 rows, 2000 points across 3 snapshots, and about a megabyte; the column count follows what the agent carries beside the fields. The baseline reports completeness 1.0, duplicity 0.0 and an outlier rate near 0.16, which is the long tail of a turbulent jet rather than a defect, and `summarize` gives temperature from 400 K to about 2700 K, pressure near one atmosphere, and the oxidizer topping out at 0.233 in the co-flow.

### 13. Add the checks the baseline cannot make

```text
Run two domain readiness checks on audit/readiness/well_points.csv through the
aidrin code:
1. Physical bounds as custom outlier rules: temperature, density and pressure
   strictly positive, and every mass_fraction_* column within [0, 1].
2. A custom AIDRIN module for species closure. Scaffold it into audit/ with
   add-custom-module, then fill it in: this hydrogen-air case stores eight
   species and leaves N2 as the complement, so a valid point has
   0 <= sum(mass_fraction_*) <= 1 and an implied N2 = 1 - sum in [0, 1]. The
   metric reports the violating row count and rate and the min, max and mean
   of the implied N2; the remedy drops the violating rows.
Report both results with the per-rule counts.
```

**Expect:** The bounds rules return a valid and outlier count per rule: temperature, density and pressure positive everywhere, and a few dozen species violations out of 48,000 checks at values near negative machine epsilon, which is float32 rounding rather than a fault. The closure module reports species sums from 0.233 to 0.650 and an implied N2 from 0.350 to 0.767 with no violations. Both ends are physically recognizable, 0.767 being the nitrogen of air and 0.350 that of the diluted fuel, which is the strongest single confirmation that the species mapping survived.

### 14. Check against the holdout and iterate

Unpack the reference into the project first:

```bash
tar xzf combustion_simulation.tar.gz -C "$PROJ/data" ./holdout
```

```text
Spot-check data/well_output/lifted_hydrogen_jet_traj_5000.hdf5 against
data/holdout/well_output/lifted_hydrogen_jet_traj_5000.hdf5 with 10 random
points per dataset by running the registered check-well-output code with its exact
command. If anything differs, fix the converter in the skill, reconvert with the
converter's registered code, and check again. When the spot-check passes, run the
full comparison the same way. After the check passes, update the skill's SKILL.md so
its rules match the converter.
```

**Expect:** The first comparison fails on one or more documented pitfalls, all visible in the checker's output, and each fix is a new version of the script inside the skill with its reconversion and check on the record. The loop ends with an exact match and the skill's rules agreeing with the converter that passed. Saving the skill again can register its scripts a second time under derived names, which duplicates a code already registered properly.

### 15. Re-measure and collect the reports

```text
Re-sample the passing conversion to audit/readiness/well_points_final.csv and
re-run the quality baseline, the bounds rules and the species-closure module on
it. Then pull every readiness report for this session with readiness_reports
and give me the per-metric change from the first conversion to the passing one.
```

**Expect:** `readiness_reports` returns the report each `aidrin` run printed, held in that run's execution record, in the order they happened. What the metrics show depends on what the loop fixed, and on this trajectory they barely move either way: the outlier rate sits near 0.16 before and after. The loop's renaming of the species fields breaks a closure module written against the old names, which is the coupling worth seeing.

### 16. Record what no artifact holds

```text
Remember for this project what you found that the catalog pages get wrong, and
what you could not verify.
```

**Expect:** The prompt hands over no list, so what lands is the agent's reading of which findings are memory-shaped. The runs behind this walkthrough turned up four candidates, each a correction to an upstream document or a named gap in what was checked: a jet composition given by volume where the data carries it by mass, a grid given as uniform where the two axes differ, boundary types with no reference file to check them against, and a publication no agent can fetch. `kb_remember` writes each to `.dsagt/explicit_memories.yaml` and the memory collection.

### 17. Recall it

```text
What do you remember about this project?
```

**Expect:** Run across the session break the recall reads from disk rather than from context, which is the case worth showing. A grounded reply carries the stored facts and, from `session_memory`, details of the run that were never stored at all. A conversion rule is not among them, because the skill already holds every one.

### 18. Generate a datacard

```text
Use the datacard-generator skill to write a Level 1 datacard for the converted
WELL file to audit/. Take the values from info.json, the conversion, and the
readiness reports, and note anything unknown rather than asking. Validate the
card with the registered datacard-validate code.
```

**Expect:** The skill's `introspect.py` runs as the registered `datacard-introspect` code, so the introspection is an execution record. The card lands under `audit/` in the Genesis template, carries the readiness findings, and `datacard-validate` accepts it. A warning that the filename differs from the one it derives from the dataset name is expected.

### 19. Reconstruct the pipeline

```text
Reconstruct the conversion, validation, and readiness pipeline from the
execution records as a bash script, save it as pipeline.sh, and put the
trajectory directory in a variable at the top so it can be rerun on the other
BlastNet trajectories. Keep the final conversion, the spot check, the full
check, the point sampling, and the three readiness checks.
```

**Expect:** `reconstruct_pipeline` saves the script and returns the recorded runs in the order they ran; the agent edits it down to the conversion, both checks, the sampling and the readiness checks, with the trajectory directory in one variable at the top. The rendered script carries every recorded run, including the `--help` runs of step 8, so the agent prunes them. A dry run's declared output can make it look like the step that produced the converted file.

### 20. Train a surrogate on the converted data

```text
Register the training script at $MODEL_DIR as a code named well-train, run
--help on it to confirm its options, then train it on
data/well_output/lifted_hydrogen_jet_traj_5000.hdf5 with the outputs under
models/. Report the run's exit code, duration, and the metrics it printed,
and run the AI-readiness check on any tabular output it produced.
```

**Expect:** The training run is an execution record with the same fields as every other code, and its span appears in the trace store beside the conversion spans. The BlastNet surrogate for this walkthrough is not in the data bundle; point `MODEL_DIR` at any WELL-format training code to run the step. A released foundation model for this format reads six consecutive snapshots and a fixed set of grid sizes, and carries no representation for species, so a slice cut for it is not the slice the conversion demo ships.

### 21. Review the project artifacts

```text
Show me the contents of my project folder in a tree format, with the artifacts dsagt recorded during this session highlighted.
```

**Expect:** A listing of the project directory marking the execution records in `trace_archive/`, the reports and readiness tables in `audit/`, the knowledge collections under `kb_index/`, the explicit memories in `.dsagt/`, the registered codes and installed skills under `skills/`, and the trace store. The agent may print the tree through a command and summarize it in the reply.

## Review the traces

Exit the agent, then:

```bash
dsagt info blastnet-well-demo      # config, tracking URI, collections, codes, sessions
dsagt traces blastnet-well-demo    # opens the MLflow viewer on the project's store
```

`dsagt info` prints the tracking URI, the chunk count of every collection, the
registered codes and installed skills, and a per-session trace and token
summary, with the failed runs of step 14 under Errors. The viewer holds one
`claude_code_conversation` subtree per turn with its `llm` and `tool_<name>`
children, a `code.execute` span per recorded run, `kb.search` and `kb.embed`
spans from the knowledge-base work, and a `save_code_spec` span per
registration. Every span carries the session id.

## Post-Conditions

1. A `combustion` knowledge collection holds the two format documents, the two BlastNet catalog pages, and the papers the literature review found, and `kb_search` answers the boundary-condition, configuration, and species questions from them.
2. `well-convert` and `literature-search` are installed under `skills/` and mirrored into the agent's native skills directory.
3. `skills/blastnet-to-well/` exists with a `SKILL.md` whose mapping rules agree with the final converter, the two documents under `references/`, and the converter and point sampler under `scripts/`.
4. Code registry contains `blastnet-inspect`, the converter's code registered by `save_skill` from the skill's script, `check-well-output`, and `well-sample-points`, all under `skills/`.
5. `data/well_output/lifted_hydrogen_jet_traj_5000.hdf5` exists and the full checker run reports an exact match to the holdout reference.
6. `audit/readiness/` holds the point tables, `audit/` holds the custom closure module, and `readiness_reports` returns every readiness run of the session with its report. On the passing conversion the closure violations are zero, the implied N2 runs 0.350 to 0.767, and the species bounds report only machine-epsilon violations.
7. `trace_archive/` holds every converter, checker, sampler, and readiness run, including the failed checks that drove the fixes.
8. `.dsagt/explicit_memories.yaml` holds what step 16 found that no other artifact records, and the agent recalls it in a later session.
9. A datacard for the converted dataset exists under `audit/`, in the Genesis template, carrying the readiness findings, and `datacard-validate` accepts it.
10. `pipeline.sh`, saved by `reconstruct_pipeline`, replays the final conversion, both checks, the sampling, and the readiness checks, calling the tools directly; the trajectory directory is the only variable to edit.
11. MLflow traces (in the serverless `mlflow.db` store) capture the session; view them with `dsagt traces blastnet-well-demo`.

## Coverage

| DSAgt Capability | Steps |
|------------------|-------|
| Skill-source listing and catalog search and install (`list_skill_sources`, `search_skills`, `install_skill`) | 1 |
| Knowledge ingestion, append and collection listing (`kb_ingest`, `kb_append`, `kb_job_status`, `kb_list_collections`) | 2, 5 |
| Semantic search (`kb_search`) | 3, 4, 6 |
| Installed-skill execution grounded in the knowledge base | 4, 6 |
| Literature review whose search terms come from the knowledge base | 4, 5 |
| Skill authoring with `skill-creator` and `save_skill`, carrying its source documents as references | 7 |
| Agent-written code from documentation | 7, 12 |
| Code registration (`save_code_spec`) and registry search | 6, 8, 12 |
| Code execution with provenance through `dsagt-run` | 6, 9–15, 20 |
| AI-readiness check on a non-tabular stage, through a derived point table | 11, 12, 15 |
| Domain readiness checks: custom outlier rules and a custom AIDRIN module | 13, 15 |
| Readiness report retrieval (`readiness_reports`) | 15 |
| Check-driven iteration with failed runs on the record | 14 |
| Explicit and episodic memory, for what no artifact holds (`kb_remember`, `kb_get_memories`) | 16, 17 |
| Base-skill use (`datacard-generator`) | 18 |
| Pipeline reconstruction with a parameterized input | 19 |
| Review of the session's artifacts | 21 |
| Observability (`dsagt info`, `dsagt traces`, the MLflow span tree) | Review the traces |

## Cleanup

```bash
dsagt rm blastnet-well-demo -y
rm combustion_simulation.tar.gz
```

## Notes

- A passing readiness baseline does not mean a correct conversion. Reading a
  field in the grid files' array order transposes it, which is a permutation:
  every value is still present, at another point. A metric built on a
  distribution cannot see that, and neither can the bounds or closure checks,
  since the same misreading applies to every field alike and a point still
  carries a physically consistent set of values. Only the comparison against
  the reference finds it, by reading the value at a coordinate rather than the
  values as a population. Runs have reported an outlier rate of 0.1537 against
  0.1522 across exactly that fix.
- The grid files and the data files do not share an array order, so a reading
  of the trajectory that checks the grid files and applies their order to the
  data gets a transposed field. The physics settles it: the correct order puts
  the jet at the inlet with its stated inlet temperature and velocity and
  leaves neighbouring points a few kelvin apart, where the wrong one leaves
  them hundreds apart.
- BlastNet carries two hydrogen jet datasets that share a fuel and a mechanism.
  This conversion targets the circular jet, 2D on a 1600 x 2000 grid with
  co-flow at 1100 K. The BlastNet 2.0 paper's appendix documents the other, a
  slot burner, 3D on 2000 x 1600 x 400 with co-flow at 850 K. The bundle's
  trajectory names belong to the first, and the dry run settles it from the
  data.
- Three statements on the lifted-flame catalog page disagree with the shipped
  data: the jet composition is given by volume where the data carries it by
  mass, the grid is given as a uniform 15 um where the data is 15.625 um in x
  and 15.0 um in y, and the mechanism's reaction count does not correspond in
  any obvious way to the entries in the shipped `h2.xml`.
- [`convert_to_well_format.py`](../combustion_simulation/reference/convert_to_well_format.py)
  is the converter this workflow produced, verified against the holdout file.
  Compare the agent's converter to it after step 14, not before.
- [`reference/sample_well_points.py`](reference/sample_well_points.py) and
  [`reference/species_closure.py`](reference/species_closure.py) are the point
  sampler and the custom AIDRIN module of steps 12 and 13, as this workflow
  produced them. The closure module's tolerance is `1e-4`, above the
  accumulated rounding error of summing eight float32 columns, and it carries
  the implied-diluent range a correct conversion reports, 0.350 to 0.767.
- The species-closure invariant is specific to a case whose stored variables
  omit the diluent. A hydrogen-air trajectory stores eight species and leaves
  N2 as the complement, which is why the check reads the implied complement
  rather than testing the sum against one. The module selects its columns by
  the `mass_fraction_` prefix, so a converter that emits `Y_H2` needs the
  prefix changed with it.
- The demo bundle is the first three snapshots of the `hydrogen-jet-5000`
  trajectory (a full trajectory is ~32 GB) with the reference WELL file sliced
  to the same steps, built with
  [`make_demo_subset.py`](../combustion_simulation/scripts/make_demo_subset.py):

  ```bash
  python3 make_demo_subset.py <traj_dir> <out_dir> --steps 3 \
      --reference <reference.hdf5>
  tar czf combustion_simulation_data.tar.gz -C <out_dir> data
  ```

  The subset converts and checks exactly like the full trajectory, since a
  converter enumerates snapshots from `info.json` and every time-varying WELL
  dataset carries time on axis 1.
