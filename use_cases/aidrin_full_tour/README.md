---
title: AIDRIN Feature Tour
domain: AI data readiness — `aidrin` metrics (quality, fairness, privacy) on UCI Adult
summary: >-
  Drive AIDRIN through DSAgt on a single tabular dataset (UCI Adult) — 15
  metrics spanning data-quality, impact-on-AI, fairness-and-bias, and
  data-governance, each recorded with full provenance.
status: published
order: 50
---

# DSAgt Demo: AIDRIN Feature Tour

> **Estimated time:** ~30–40 minutes — this is a long-form tour, not a quick
> demo. Most of the time is the one-time AIDRIN build (clone + `pip install -e`
> in a Python 3.10 venv) plus the agent issuing one run per metric.

This guide drives [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector)
through DSAgt on a single tabular dataset — metrics from all four of AIDRIN's
categories, with full execution provenance. It is the companion to the
[cryo-EM curation demo](../cryoem/), where the readiness assessment applies the quality
subset to scientific data; here we use a dataset rich enough to exercise the fairness and privacy
metrics too.

The dataset is the **UCI Adult** census extract included with AIDRIN
(`examples/sample_data/csv/adult.csv`). It has everything these metrics need: a record **ID**, quasi-identifiers (`age`, `sex`, `race`), sensitive attributes (`sex`,
`race`), and a prediction **target** (`income`). The walkthrough has been run end to end with
Claude Code on Sonnet 4.5.

## Applied Metrics

| Category | Metrics |
|---|---|
| data-quality | `completeness`, `duplicity`, `outliers` |
| impact-of-data-on-AI | `correlations`, `feature-relevance` |
| fairness-and-bias | `class-imbalance`, `statistical-rates`, `representation-rate` |
| data-governance | `k-anonymity`, `l-diversity`, `t-closeness`, `entropy-risk`, `single-attribute-risk`, `multiple-attribute-risk`, `differential-privacy` |

## Prerequisites

- DSAgt installed (`uv sync --all-groups`) and an agent platform installed and **already
  authenticated** (BYOA — your agent talks to its own LLM provider; dsagt writes no
  credentials). The default local embedder needs no API key.
- `uv` and Git installed. Enabling the readiness assessment at `dsagt init` installs AIDRIN itself
  (one-time, shared across projects, Python 3.10-3.12).

## Setup

```bash
dsagt init
```

At the menu, name the project `aidrin-tour`, pick your agent, and answer **yes** to
"Enable the AIDRIN AI-readiness assessment?". Init installs AIDRIN on first use (one-time,
shared across projects, into `~/dsagt-projects/.tools/`). Then copy the sample dataset
into the project and start the session:

```bash
PROJ=~/dsagt-projects/aidrin-tour
mkdir -p "$PROJ/data"
git clone -b develop --depth 1 https://github.com/idtlab/AIDRIN.git   # for the sample dataset only
cp AIDRIN/examples/sample_data/csv/adult.csv "$PROJ/data/"
dsagt start aidrin-tour
```

Every init installs the `aidrin` skill into `$PROJ/skills/aidrin/`; the readiness assessment
records the executable in `.dsagt/config.yaml` and adds the assessment rules to the instructions.

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

**Expect** — the exact commands and representative results (positional args; JSON to stdout):

**Data quality**

| Command | Result |
|---|---|
| `aidrin run completeness data/adult.csv` | overall `1.0` |
| `aidrin run duplicity data/adult.csv` | `0.0` |
| `aidrin run outliers data/adult.csv` | overall `≈0.050` (`hours.per.week` ≈0.277) |

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
| `aidrin run differential-privacy data/adult.csv "age,hours.per.week" 1.0` | noised mean/variance per column |

The agent should produce a four-part verdict: **quality** is clean (complete, no duplicates,
moderate `hours.per.week` outliers); **impact** shows `education.num`/`age` as the strongest income
predictors; **fairness** flags a large gender gap in the target (men ~2.8× more likely `>50K`); and
**governance** flags severe re-identification risk (`k = 1`, `l = 1`) on the `age,sex,race`
quasi-identifiers — bin or suppress before sharing.

### 3. (Optional) Batch several metrics from one config

```text
Write an aidrin batch config (YAML) that runs completeness, class-imbalance, statistical-rates, and
representation-rate on data/adult.csv with target income and sensitive attribute sex, then run it
with the aidrin skill through dsagt-run.
```

The config is one flat mapping, not per-metric blocks; the `aidrin` skill's
`reference/metrics.md` documents the keys. For this step:

```yaml
file-path: data/adult.csv
file-type: csv
metrics: [completeness, class-imbalance, statistical-rates, representation-rate]
target-column: income
y-true-column: income
sensitive-attribute-column: sex
columns: [sex, race]
```

### 4. Generate a datacard from the assessment

```text
Search for a skill that can generate a datacard for data/adult.csv, then use it to write a Level 1
datacard that incorporates the readiness findings above. Take the values from the dataset and the
reports, and note anything unknown rather than asking.
```

The agent discovers the `datacard-generator` skill and writes a Genesis Datacard (e.g.
`data/genesis_datacard_*.md`) documenting the dataset and its readiness profile.

### 5. Reconstruct the pipeline

```text
Reconstruct the full readiness assessment you just ran from the execution records as a bash script.
```

## Post-Conditions

1. `skills/aidrin/SKILL.md` is present, with a `PROVENANCE.txt` naming the AIDRIN source.
2. `trace_archive/` holds one execution record per metric run from step 2.
3. Results span the four categories, with the gender-fairness gap and the `k = 1` / `l = 1`
   re-identification risks identified.
4. A datacard for the dataset exists (`data/genesis_datacard_*.md`).
5. A reconstructed pipeline script replays all metrics in order.
6. MLflow traces capture token usage, latency, and the code-execution and MCP tool spans.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| The `aidrin` base skill installed at init; the readiness assessment enabled | Setup |
| Base-skill use: the `aidrin` CLI through `dsagt-run` | 1 |
| Code execution with provenance (execution records in `trace_archive/`) | 2 |
| Multi-metric orchestration | 2 |
| Multi-metric / batch execution | 3 |
| Skill discovery and use (datacard generation) | 4 |
| Pipeline reconstruction from execution records | 5 |
| Observability (MLflow spans in the serverless `mlflow.db` store) | all |

View the traces any time with
`mlflow ui --backend-store-uri sqlite:///$PROJ/mlflow.db`.

## Cleanup

```bash
dsagt rm aidrin-tour -y
rm -rf AIDRIN
```
