# Readiness Gate

DSAgt can run [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector) as the check before and after every pipeline stage. The `aidrin` skill is installed in every project from the AIDRIN repository whether or not the gate is on; the gate makes the agent apply it at every tabular stage. The gate is an opt-in at `dsagt init`; off by default.

The pipeline-builder instructions already require a paired check around every data operation, with reports in `audit/`. Without the gate, the agent writes its own check code for each stage. With it, the check for any tabular stage is a fixed AIDRIN metric set, so every stage of a pipeline is measured the same way and the before/after delta is comparable across stages and projects.

## Enable

Answer **yes** to "Enable the AIDRIN readiness gate?" in the `dsagt init` menu. (The automation path is `dsagt init <name> --agent <agent> --readiness aidrin`, with `--readiness-executable PATH` to use an AIDRIN you already installed.)

Init then does two things:

1. Installs AIDRIN once into `~/dsagt-projects/.tools/aidrin/` (a venv built with `uv`; AIDRIN requires Python 3.10-3.12). Every project shares it. A failed install is reported and the config still records the expected path; re-run `dsagt init` to retry.
2. Appends a short block to the agent's instructions file stating that `check_[X]` for tabular stages is an AIDRIN run of the project's metric profile, naming the executable and the `aidrin` skill in `<project>/skills/aidrin/`.

The config records the choice:

```yaml
readiness:
  tool: aidrin
  executable: /Users/me/dsagt-projects/.tools/aidrin/bin/aidrin
  profile: quality
```

## Rules the agent follows

Every `aidrin` command in the project, gate run or not, goes through `dsagt-run --code aidrin` so it is recorded; the agent uses the AIDRIN CLI.

1. Before and after each operation on a tabular file (CSV, Excel, JSON, HDF5, Parquet, npz), run every metric of the project profile on that file, one `aidrin run <metric> <file>` per metric, each wrapped by `dsagt-run --code aidrin` so it is recorded. Collected reports go to `audit/step_N_pre.aidrin.json` and `audit/step_N_post.aidrin.json`.
2. The gate is fixed: gate runs skip the skill's intent-elicitation and plan-confirmation steps. The skill's full workflow is for readiness assessments the user asks for beyond the profile.
3. After the post-run, report the per-metric change to the user before proposing the next step.
4. Do not write a custom check for a metric AIDRIN provides.
5. Metrics outside the profile (fairness rates, privacy, file-reference validation) run the same way only when the user confirms the dataset has the attributes those metrics assume.
6. Stages whose input and output are not tabular keep the generic check rule.

## Profiles

| Profile | Metrics | Column arguments |
|---|---|---|
| `quality` (default) | completeness, duplicity, outliers | none |
| `supervised` | quality + class-imbalance, feature-relevance | the target column; categorical and numerical column lists for feature-relevance |

Set `readiness.profile` in `.dsagt/config.yaml` to change the profile.

## Try it

A three-stage pipeline on AIDRIN's own demo dataset — 525 sensor readings with 25 exact
duplicates, missing values in every sensor column, and temperature outliers. About ten minutes;
the only download is a 40 KB CSV.

```bash
dsagt init
```

At the menu, name the project `gate-demo`, pick your agent, leave the knowledge
collections and skill sources unchecked (the gate needs neither), and answer
**yes** to "Enable the AIDRIN readiness gate?". Init installs AIDRIN on first
use (one-time, shared across projects). Then:

```bash
mkdir -p ~/dsagt-projects/gate-demo/data
curl -sL https://raw.githubusercontent.com/idtlab/AIDRIN/develop/demos/messy_sensor_data.csv \
    -o ~/dsagt-projects/gate-demo/data/sensors.csv
dsagt start gate-demo
```

Then one prompt. Do not mention AIDRIN or checks — the point is what the agent does on its own:

```text
Build a curation pipeline for data/sensors.csv in three steps, one at a time:
1. drop exact duplicate rows -> data/dedup.csv
2. drop rows with a missing temperature -> data/complete.csv
3. drop rows whose temperature is more than 3 standard deviations from the mean -> data/clean.csv
Confirm the approach with me before each step.
```

At each stage the agent should run the AIDRIN gate on the stage input before the operation and on
the output after it, write both reports to `audit/`, and show the metric delta before proposing
the next step. Expected values on this dataset (pre column measured directly):

| Stage | Metric | pre | post |
|---|---|---|---|
| 1 dedup | duplicity | 0.0476 | 0.0 |
| 2 complete | completeness (`temperature`) | 0.8457 | 1.0 |
| 3 outliers | outliers (`temperature`) | 0.0225 | lower |

Afterwards, `ls ~/dsagt-projects/gate-demo/audit` shows the six reports and `trace_archive/`
holds one record per metric of each gate run and one per operation. Clean up with `dsagt rm gate-demo -y`.

## Demos

The [cryo-EM curation demo](use-cases/cryoem.md) runs with the gate enabled on real scientific data — the gate measures the particle-curation step unprompted. The [AIDRIN tour](use-cases/aidrin_full_tour.md) drives quality, fairness, and privacy metrics on a tabular dataset.
