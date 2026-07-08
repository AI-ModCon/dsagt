# DSAgt Demo: Full AIDRIN Feature Tour

> **Estimated time:** ~30–40 minutes — this is a long-form tour, not a quick
> demo. Most of the time is the one-time AIDRIN build (clone + `pip install -e`
> in a Python 3.10 venv) plus the agent issuing ~15 separate metric runs. The
> dataset ships with AIDRIN (no large download).

This guide drives **every** [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness Inspector)
metric through DSAgt on a single tabular dataset — exercising all 15 metrics across all four
categories, with full execution provenance. It is the companion to the
[cryo-EM readiness gate demo](../aidrin_readiness_gate/cryoem_readiness_demo.md), which applies the quality subset to
scientific data; here we use a dataset rich enough to exercise the fairness and privacy metrics too.

The dataset is the **UCI Adult** census extract bundled with AIDRIN
(`examples/sample_data/csv/adult.csv`). It has everything the full metric suite needs: a record
**ID**, quasi-identifiers (`age`, `sex`, `race`), sensitive attributes (`sex`, `race`), and a
prediction **target** (`income`).

## The 15 metrics, by category

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
- **AIDRIN** installed from its `develop` branch in its own Python 3.10 virtual environment.
- Git installed. (No large download — the dataset ships with AIDRIN.)

## Setup

```bash
git clone -b develop https://github.com/idtlab/AIDRIN.git
python3.10 -m venv aidrin-venv
source aidrin-venv/bin/activate
pip install -e ./AIDRIN
aidrin list          # sanity check: 15 metrics in 4 categories
AIDRIN_BIN="$(pwd)/aidrin-venv/bin/aidrin"; echo "$AIDRIN_BIN"
deactivate

dsagt init aidrin-tour --agent claude
PROJ=~/dsagt-projects/aidrin-tour
mkdir -p "$PROJ/data"
cp AIDRIN/examples/sample_data/csv/adult.csv "$PROJ/data/"
dsagt start aidrin-tour
```

## Execution

Paste these prompts one at a time (substitute the absolute `$AIDRIN_BIN` path).

### 1. Register the AIDRIN CLI

```text
Register a data-readiness CLI named aidrin into the code registry. The executable is at
<AIDRIN_BIN>. Run "<AIDRIN_BIN> --help" and "<AIDRIN_BIN> list" to discover its subcommands and the
15 metrics, then save a code spec named aidrin describing the run/batch/data-quality subcommands
and their positional arguments.
```

**Verify:** `Search the registry for the aidrin data-readiness code.`

### 2. Run all 15 metrics through `dsagt-run`

```text
Using the registry aidrin code, run AIDRIN's full readiness assessment on data/adult.csv, executing
every metric through dsagt-run so each is recorded. Cover all four categories:
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
through the registry aidrin code.
```

Batch config keys: `file-path`, `file-type`, `metrics`, `target-column`,
`sensitive-attribute-column`, `columns`.

### 4. Generate a datacard from the assessment

```text
Search for a skill that can generate a datacard for data/adult.csv, then use it to produce the
datacard — incorporating the readiness findings above.
```

The agent discovers the `datacard-generator` skill and writes a Genesis Datacard (e.g.
`data/genesis_datacard_*.md`) documenting the dataset and its readiness profile.

### 5. Reconstruct the pipeline

```text
Reconstruct the full readiness assessment you just ran from the execution records as a bash script.
```

## Post-Conditions

1. Code registry contains the `aidrin` spec (`codes/aidrin/SKILL.md`).
2. `trace_archive/` holds one provenance record per metric run (15 from step 2).
3. Results span all four categories, with the gender-fairness gap and the `k = 1` / `l = 1`
   re-identification risks identified.
4. A datacard for the dataset exists (`data/genesis_datacard_*.md`).
5. A reconstructed pipeline script replays all metrics in order.
6. MLflow traces capture token usage, latency, and the `dsagt-run` / MCP spans.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| External-CLI registration (`save_code_spec`) | 1 |
| Registry search | 1 (Verify) |
| Code execution with provenance (`dsagt-run` → `trace_archive/`) | 2 |
| Full-suite (15-metric) orchestration | 2 |
| Multi-metric / batch execution | 3 |
| Skill discovery and use (datacard generation) | 4 |
| Pipeline reconstruction from execution records | 5 |
| Observability (MLflow spans in the serverless `mlflow.db` store) | all |

View the traces any time with
`mlflow ui --backend-store-uri sqlite:///$PROJ/mlflow.db`.

## Cleanup

```bash
dsagt rm aidrin-tour -y
rm -rf AIDRIN aidrin-venv
```
