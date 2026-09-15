# AI-Readiness Check

DSAgt is configured at init to run [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector) as the check before and after every tabular pipeline stage; uncheck it on the menu to turn it off. AIDRIN installs with dsagt, and every project gets the `aidrin` skill and an `aidrin` code, so each call the agent makes is an execution record in `trace_archive/` like any other code. A user who asks "is my data AI-ready?" gets the skill's own workflow.

The pipeline-builder instructions require a paired check around every data operation, with reports in `audit/`. The AI-readiness check makes that check concrete for tabular files: it is the `aidrin` skill's quality baseline (completeness, duplicity, outliers), run on a stage's input before the operation and on its output after it, so every stage of a pipeline is measured the same way and the before/after delta is comparable across stages and projects.

## The setting

`dsagt init` asks "Assess tabular data for AI-readiness before and after each data transform?", default yes. (The automation path turns it off with `dsagt init <name> --agent <agent> --no-readiness`.) The answer is the one thing the setting controls: when it is yes, the agent's instructions carry one paragraph at the per-operation check rule; when it is no, they do not. The `aidrin` code and skill are present either way.

The config records the answer:

```yaml
readiness:
  auto_assess: true
```

## What the paragraph says

For a stage whose input or output is a tabular file (CSV, Excel, JSON, HDF5, Parquet, npz), the check is the `aidrin` skill's quality baseline, run through the registered `aidrin` code before and after the operation. The agent runs the baseline directly, without the skill's intent and plan steps, which are for assessments the user asks for. It saves the results as `audit/step_N_pre.aidrin.json` and `audit/step_N_post.aidrin.json`, reports the per-metric change before proposing the next step, writes no custom check for a metric AIDRIN provides, and keeps the generic check rule for stages whose input and output are not tabular.

## Try it

A three-stage pipeline on AIDRIN's own demo dataset — 525 sensor readings with 25 exact
duplicates, missing values in every sensor column, and temperature outliers. About ten minutes;
the only download is a 40 KB CSV.

```bash
dsagt init
```

At the menu, name the project `assessment-demo`, pick your agent, leave the knowledge
collections and skill sources unchecked (the check needs neither), and keep the
AI-readiness check on. Then:

```bash
mkdir -p ~/dsagt-projects/assessment-demo/data
curl -sL https://raw.githubusercontent.com/idtlab/AIDRIN/develop/demos/messy_sensor_data.csv \
    -o ~/dsagt-projects/assessment-demo/data/sensors.csv
dsagt start assessment-demo
```

Then one prompt. Do not mention AIDRIN or checks — the point is what the agent does on its own:

```text
Build a curation pipeline for data/sensors.csv in three steps, one at a time:
1. drop exact duplicate rows -> data/dedup.csv
2. drop rows with a missing temperature -> data/complete.csv
3. drop rows whose temperature is more than 3 standard deviations from the mean -> data/clean.csv
Confirm the approach with me before each step.
```

At each stage the agent should run the AIDRIN quality baseline on the stage input before the
operation and on the output after it, write both reports to `audit/`, and show the metric delta
before proposing the next step. Expected values on this dataset (pre column measured directly):

| Stage | Metric | pre | post |
|---|---|---|---|
| 1 dedup | duplicity | 0.0476 | 0.0 |
| 2 complete | completeness (`temperature`) | 0.8457 | 1.0 |
| 3 outliers | outliers (`temperature`) | 0.0225 | lower |

Afterwards, one more prompt:

```text
Show me the execution records for this session as a table of step, command, and exit code.
```

The table lists one record per baseline run (two per stage) and one per operation, and
`audit/` holds the six reports. Clean up with `dsagt rm assessment-demo -y`.

## Demos

The [cryo-EM curation demo](use-cases/cryoem.md) runs on real scientific data — the check measures the particle-curation step unprompted. The [AIDRIN example](use-cases/aidrin-ai-readiness.md) drives quality, fairness, and privacy metrics on a tabular dataset.
