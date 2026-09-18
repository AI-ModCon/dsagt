# AI-Readiness Check

DSAgt is configured at init to run [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector) as the check before and after every tabular pipeline stage; uncheck it on the menu to turn it off. AIDRIN installs with dsagt, and every project gets the `aidrin` skill and an `aidrin` code, so each call the agent makes is an execution record in `trace_archive/` like any other code. A user who asks "is my data AI-ready?" gets the AIDRIN skill's workflow.

The DSAgt instructions request assessments for the effects of data transformations for the downstream application, with reports in `audit/`. For tabular files that check is the `aidrin` skill's quality baseline (completeness, duplicity, outliers), run on the stage's input before the operation and on its output after it. Every stage is measured the same way, so the before/after delta is comparable across stages and projects.

`dsagt init` asks "Assess tabular data for AI-readiness before and after each data transform?", default yes; when it is yes, the agent's instructions carry one paragraph at the per-operation check rule; when it is no, they do not. The `aidrin` code and skill are present either way.

The inserted paragraph:

> #### AI-readiness check
>
> For a stage whose input or output is a tabular file (CSV, Excel, JSON, HDF5,
> Parquet, npz), the check is the `aidrin` skill's quality baseline: run it on
> the file before and after the operation, through the registered `aidrin`
> code's `executable` (never bare `aidrin`). Run the baseline directly; do not
> ask the user about intent or confirm a plan for these checks (the skill's full workflow is for assessments the user
> asks for). The CLI prints its report to stdout, so name the audit file with
> `dsagt-run`'s `--stdout` option, which records it as the run's output:
> `dsagt-run --code aidrin --stdout audit/step_N_pre.aidrin.json -- aidrin
> data-quality <file> --detail` before the operation and `--stdout
> audit/step_N_post.aidrin.json` after it, then report the per-metric change
> to the user before proposing the next step. Do
> not write a custom check for a metric AIDRIN provides. A stage with tabular
> input or output gets this check; every other stage keeps the check rule
> above.

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
