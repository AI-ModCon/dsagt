---
title: AIDRIN
domain: AI data readiness, `aidrin` metrics (quality, fairness, privacy) on UCI Adult
summary: >-
  Apply AIDRIN through DSAgt to a single tabular dataset (UCI Adult), 15
  metrics spanning data-quality, impact-on-AI, fairness-and-bias, and
  data-governance.
status: published
order: 50
---

# DSAgt Demo: AIDRIN

> **Estimated time:** ~15 minutes

This tutorial demonstrates [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector) on a single tabular dataset: metrics from all four of AIDRIN's
categories, with execution provenance.
For how the `aidrin` skill and code are set up and what rules the agent follows, see the
[AI-Readiness Check](../../docs/readiness.md) page.

The dataset is the **UCI Adult** census extract included with AIDRIN
(`examples/sample_data/csv/adult.csv`). The demonstrated AIDRIN metrics rely on: a record **ID**, quasi-identifiers (`age`, `sex`, `race`), sensitive attributes (`sex`,
`race`), and a prediction **target** (`income`).

## Applied Metrics

| Category | Metrics |
|---|---|
| data-quality | `completeness`, `duplicity`, `outliers` |
| impact-of-data-on-AI | `correlations`, `feature-relevance` |
| fairness-and-bias | `class-imbalance`, `statistical-rates`, `representation-rate` |
| data-governance | `k-anonymity`, `l-diversity`, `t-closeness`, `entropy-risk`, `single-attribute-risk`, `multiple-attribute-risk`, `differential-privacy` |

## Prerequisites

- DSAgt installed and an agent platform installed and **already
  authenticated**
- Python 3.12 or later

## Setup

```bash
dsagt init
```

At the menu, name the project `aidrin`, pick your agent, and keep the defaults. Then fetch the
sample dataset into the project and start the session:

```bash
PROJ=~/dsagt-projects/aidrin
mkdir -p "$PROJ/data"
curl -sL https://raw.githubusercontent.com/idtlab/AIDRIN/v2026.08.2/examples/sample_data/csv/adult.csv \
    -o "$PROJ/data/adult.csv"
dsagt start aidrin
```

## Execution

Paste these prompts one at a time.

### 1. Confirm the AIDRIN skill is installed

```text
Using the aidrin skill, list the readiness metrics AIDRIN provides.
```

**Verify:** the agent reads `skills/aidrin/SKILL.md` and its `reference/metrics.md` and lists the metrics by category; it may also run `aidrin list` through `dsagt-run`.

### 2. Run the metrics

```text
Using the aidrin skill, run a readiness assessment on data/adult.csv. Cover these
four categories:
(1) data-quality: completeness, duplicity, outliers;
(2) impact-of-data-on-AI: correlations on "age,education.num,sex,race", and feature-relevance with
    categorical columns "workclass,education,sex,race", numerical columns
    "age,education.num,hours.per.week", target income;
(3) fairness-and-bias: class-imbalance on income, statistical-rates on income with sensitive
    attribute sex, representation-rate on "sex,race";
(4) data-governance: k-anonymity on "age,sex,race", l-diversity on "age,sex,race" with sensitive
    column income, t-closeness on "age,sex,race" with sensitive column income, entropy-risk on
    "age,sex,race", single-attribute-risk with id-column ID and eval-columns "age,sex,race",
    multiple-attribute-risk with id-column ID and eval-columns "age,sex,race", and
    differential-privacy on "age,hours.per.week" with epsilon 1.0.
Then give me a readiness verdict organized by the four categories.
```

**Expect:** the exact commands and representative results (positional args; JSON to stdout):

**Data quality**

The skill runs the three quality metrics as one baseline call.

| Command | Result |
|---|---|
| `aidrin data-quality data/adult.csv --detail` | completeness `1.0`; duplicity `0.0`; outliers overall `≈0.050` (`hours.per.week` ≈0.277) |

**Impact on AI**

| Command | Result |
|---|---|
| `aidrin run correlations data/adult.csv "age,education.num,sex,race"` | Theil's U + Pearson matrices |
| `aidrin run feature-relevance data/adult.csv "workclass,education,sex,race" "age,education.num,hours.per.week" income` | Pearson-to-target (e.g. `education.num` ≈0.34, `age` ≈0.23) |

**Fairness & bias**

| Command | Result |
|---|---|
| `aidrin run class-imbalance data/adult.csv income` | imbalance degree `≈0.52` |
| `aidrin run statistical-rates data/adult.csv income sex` | Female `>50K` ≈11% vs Male ≈31% |
| `aidrin run representation-rate data/adult.csv "sex,race"` | Male:Female ≈2.0, White:Black ≈8.9 |

**Data governance / privacy**

| Command | Result |
|---|---|
| `aidrin run k-anonymity data/adult.csv "age,sex,race"` | `k = 1` |
| `aidrin run l-diversity data/adult.csv "age,sex,race" income` | `l = 1` |
| `aidrin run t-closeness data/adult.csv "age,sex,race" income` | `t ≈ 0.76` |
| `aidrin run entropy-risk data/adult.csv "age,sex,race"` | `≈0.06` |
| `aidrin run single-attribute-risk data/adult.csv ID "age,sex,race"` | per-attribute risk stats |
| `aidrin run multiple-attribute-risk data/adult.csv ID "age,sex,race"` | joint re-identification risk |
| `aidrin run differential-privacy data/adult.csv "age,hours.per.week" 1.0` | noised mean/variance per column; also writes `noisy/noisy_data.csv` |

The agent should produce a four-part verdict: **quality** is clean (complete, no duplicates,
moderate `hours.per.week` outliers); **impact** shows `education.num`/`age` as the strongest income
predictors; **fairness** flags a large gender gap in the target (men ~2.8× more likely `>50K`); and
**governance** flags severe re-identification risk (`k = 1`, `l = 1`) on the `age,sex,race`
quasi-identifiers; bin or suppress before sharing.

### 3. Batch several metrics from one config

```text
Write an aidrin batch config (YAML) that runs completeness, class-imbalance, statistical-rates, and
representation-rate on data/adult.csv with target income and sensitive attribute sex, then run it
with the aidrin skill. The config is one flat mapping, and the key names are in the skill's
reference/metrics.md.
```

The config for this step:

```yaml
file_path: data/adult.csv
metrics: [completeness, class-imbalance, statistical-rates, representation-rate]
target-column: income
y-true-column: income
sensitive-attribute-column: sex
columns: [sex, race]
```

### 4. Generate a datacard from the assessment

```text
Use the datacard-generator skill to write a Level 1 datacard for data/adult.csv that incorporates
the readiness findings above. Take the values from the dataset and the reports, note anything
unknown rather than asking, and write it as one file, data/genesis_datacard_adult.md.
```

The agent follows the `datacard-generator` base skill: it fills the skill's template from
the dataset and the readiness reports, writes one Genesis Datacard,
`data/genesis_datacard_adult.md`, and validates it with the registered `datacard-validate`
code. The validator warns that the filename differs from the one it derives from the dataset
name; that warning is expected, since the prompt fixes the filename.

### 5. Review the execution records

```text
Show me the execution records for this session as a table of metric, command, and exit code.
```

The agent reads the records `dsagt-run` wrote to `trace_archive/` and lists one row per
`aidrin` command: the runs from step 2 (thirteen with the quality baseline as one
`data-quality` call, fifteen when the agent runs the three quality metrics separately) and
the batch run from step 3, every exit code 0.

### 6. Review the project artifacts

```text
Show me the contents of my project folder in a tree format, with the artifacts dsagt recorded during this session highlighted. Include the registered codes and installed skills.
```

**Expect:** a listing of the whole project directory, including the registered codes and
installed skills under `skills/`, with a line on what each entry is. The listing marks the
execution records in `trace_archive/`, the reports in `audit/`, the datacard, the trace
store `mlflow.db`, and the session's other outputs.

## Post-Conditions

1. `skills/aidrin/SKILL.md` is present, with a `PROVENANCE.txt` naming the AIDRIN source.
2. `trace_archive/` holds one execution record per `aidrin` command from step 2, at least thirteen.
3. Results span the four categories, with the gender-fairness gap and the `k = 1` / `l = 1`
   re-identification risks identified.
4. One datacard for the dataset exists, `data/genesis_datacard_adult.md`.
5. The agent lists every metric call from the execution records with its command and exit code.
6. MLflow traces capture token usage, latency, and the code-execution spans.

## Coverage

| DSAgt Capability | Steps |
|------------------|-------|
| The `aidrin` base skill and code installed at init | Setup |
| Base-skill use: the `aidrin` CLI through `dsagt-run` | 1 |
| Code execution with provenance (execution records in `trace_archive/`) | 2 |
| Multi-metric orchestration | 2 |
| Multi-metric / batch execution | 3 |
| Base-skill use (`datacard-generator`) | 4 |
| Provenance review from the execution records | 5 |
| Review of the session's artifacts | 6 |
| Observability (MLflow spans in the serverless `mlflow.db` store) | all |

View the traces any time with
`dsagt traces aidrin`.

## Cleanup

```bash
dsagt rm aidrin -y
```
