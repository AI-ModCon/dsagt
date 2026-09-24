# AI-Readiness Check

DSAgt is configured at init to run [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector) as the check before and after every tabular pipeline stage; uncheck it on the menu to turn it off. AIDRIN installs with dsagt, and every project gets the `aidrin` skill and an `aidrin` code, so each call the agent makes is an execution record in `trace_archive/` like any other code. A user who asks "is my data AI-ready?" gets the AIDRIN skill's workflow.

The DSAgt instructions request assessments for the effects of data transformations for the downstream application. For tabular files that check is the `aidrin` skill's quality baseline (completeness, duplicity, outliers), run on the data as it arrives and on the output of each transformation. Every stage is measured the same way, so the change between a table and the one it was derived from is comparable across stages and projects. Each run's report is the text AIDRIN printed, which its execution record holds and `readiness_reports` returns.

`dsagt init` asks "Assess tabular data for AI-readiness before and after each data transform?", default yes; when it is yes, the agent's instructions carry one paragraph at the per-operation check rule; when it is no, they do not. The `aidrin` code and skill are present either way.

The inserted paragraph:

> #### AI-readiness check
>
> For a stage whose input or output is a tabular file (CSV, TSV, Excel, JSON,
> HDF5, Parquet, npz), the check is the `aidrin` skill's quality baseline: run
> it on the data as it arrives and on the output of each transformation,
> through the registered `aidrin` code's `executable` (never bare `aidrin`). A
> JSON, HDF5, or NumPy file may hold nested or multi-dataset structure that
> the AIDRIN baseline reads as one flat table; say so beside the numbers when
> you report them. The report AIDRIN prints is saved to the run's record and
> is retrieved with the `readiness_reports` MCP tool. Report the per-metric
> change to the user before proposing the next step. Compare a score only with
> an earlier report on the same table, or with the report of the table it was
> derived from. Do not write a custom check for a metric AIDRIN provides.
## Try it

A three-stage pipeline on AIDRIN's own demo dataset: 525 sensor readings with 25 exact
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

Then one prompt. Leave AIDRIN and checks out of it; the demo shows what the agent does unprompted:

```text
Build a curation pipeline for data/sensors.csv in three steps, one at a time:
1. drop exact duplicate rows -> data/dedup.csv
2. drop rows with a missing temperature -> data/complete.csv
3. drop rows whose temperature is more than 3 standard deviations from the mean -> data/clean.csv
Confirm the approach with me before each step.
```

At each stage the agent should run the AIDRIN quality baseline on the stage input and on the
output it produces, and show the metric change before proposing the next step. Expected values on this dataset (pre column measured directly):

| Stage | Metric | pre | post |
|---|---|---|---|
| 1 dedup | duplicity | 0.0476 | 0.0 |
| 2 complete | completeness (`temperature`) | 0.8457 | 1.0 |
| 3 outliers | outliers (`temperature`) | 0.0225 | lower |

Afterwards, one more prompt:

```text
Show me the execution records for this session as a table of step, command, and exit code.
```

The table lists one record per baseline run (two per stage) and one per operation, and each
baseline record holds the report that run printed. Clean up with `dsagt rm assessment-demo -y`.

## Demos

The [cryo-EM curation demo](use-cases/cryoem.md) runs on real scientific data; the check measures the particle-curation step unprompted. The [AIDRIN example](use-cases/aidrin-ai-readiness.md) drives quality, fairness, and privacy metrics on a tabular dataset.
