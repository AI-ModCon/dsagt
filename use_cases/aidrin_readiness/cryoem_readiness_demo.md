# DSAgt Demo: AIDRIN Readiness Gate on a Cryo-EM Pipeline

> **Estimated time:** ~30 minutes, dominated by a **~2 GB EMPIAR-10017
> download** (from `calla.rnet.missouri.edu`) plus the one-time AIDRIN build.
> The agent portion (register + before/after assessment + datacard) is ~10
> minutes once the data is staged.

This guide demonstrates DSAgt using [AIDRIN](https://github.com/idtlab/AIDRIN) (AI Data Readiness
Inspector) as a **readiness gate** around a cryo-EM data-curation step. The agent registers the
AIDRIN CLI, then runs the applicable readiness metrics **before and after** particle curation to
*measure* how much the pipeline improved the data — every call recorded through `dsagt-run`.

This use case is self-contained: it downloads its own cryo-EM data. It pairs naturally with the
[Cryo-EM curation demo](../cryoem/cryoem_demo.md), which builds the curation pipeline itself, but
does not require it.

## Which metrics apply to cryo-EM data

Cryo-EM particle tables are physical measurements (defocus, CTF, coordinates, class assignments) —
there are no people in them. So this demo runs AIDRIN's **data-quality** and
**impact-of-data-on-AI** metrics, plus **class-imbalance** applied to the particle class label —
the metrics that make scientific sense here:

| Category | Metrics used |
|---|---|
| data-quality | `completeness`, `duplicity`, `outliers` |
| impact-of-data-on-AI | `correlations`, `feature-relevance` |
| fairness-and-bias | `class-imbalance` (on the particle class / selection label) |

AIDRIN's **fairness rate** metrics (`statistical-rates`, `representation-rate`) and **all
data-governance / privacy** metrics (`k-anonymity`, `l-diversity`, `t-closeness`, `entropy-risk`,
`single`/`multiple-attribute-risk`, `differential-privacy`) are **deliberately excluded** — they
assume sensitive attributes or personally-identifying quasi-identifiers, which cryo-EM particle
data does not contain. The [full AIDRIN tour](aidrin_full_tour_demo.md) exercises those on a
tabular dataset where they do apply.

## Prerequisites

- DSAgt installed (`uv sync --all-groups`) and an agent platform installed and **already
  authenticated** (BYOA — dsagt writes no credentials). The default local embedder needs no
  API key.
- **AIDRIN** installed from its `develop` branch in its own Python 3.10 virtual environment
  (see Setup).
- ~2 GB disk for the EMPIAR-10017 cryo-EM data.
- Git installed.

## Setup

### 1. Install AIDRIN (develop branch)

```bash
git clone -b develop https://github.com/idtlab/AIDRIN.git
python3.10 -m venv aidrin-venv
source aidrin-venv/bin/activate
pip install -e ./AIDRIN
aidrin list          # sanity check: prints 15 metrics in 4 categories
AIDRIN_BIN="$(pwd)/aidrin-venv/bin/aidrin"; echo "$AIDRIN_BIN"
```

### 2. Download the cryo-EM data and build before/after tables

Download the EMPIAR-10017 (β-galactosidase) subset from CryoPPP — it ships ground-truth particle
tables with real CTF/defocus columns and a selected-vs-excluded curation split:

```bash
mkdir -p demo_data/cryoem && cd demo_data/cryoem
curl -L https://calla.rnet.missouri.edu/cryoppp/10017.tar.gz -o 10017.tar.gz
tar xzf 10017.tar.gz
cd ../..
```

Build a **before** table (all picked particles + a `selected` flag) and an **after** table (the
curated, selected-only particles):

```bash
python - <<'PY'
import pandas as pd
gt = "demo_data/cryoem/10017/ground_truth"
sel = pd.read_csv(f"{gt}/empiar-10017_particles_selected.csv")
exc = pd.read_csv(f"{gt}/empiar-10017_particles_excluded.csv")
sel["selected"] = 1; exc["selected"] = 0
pd.concat([sel, exc], ignore_index=True).to_csv("demo_data/cryoem/cryoem_before.csv", index=False)
sel.drop(columns=["selected"]).to_csv("demo_data/cryoem/cryoem_after.csv", index=False)
print("before:", len(sel) + len(exc), "particles   after:", len(sel), "particles")
PY
deactivate
```

### 3. Initialize and start a DSAgt project

```bash
dsagt init cryoem-readiness --agent claude
PROJ=~/dsagt-projects/cryoem-readiness
mkdir -p "$PROJ/data"
cp demo_data/cryoem/cryoem_before.csv demo_data/cryoem/cryoem_after.csv "$PROJ/data/"
dsagt start cryoem-readiness
```

## Execution

Paste these prompts into the agent one at a time (substitute the absolute `$AIDRIN_BIN` path).

### 1. Register the AIDRIN CLI

```text
Register a data-readiness CLI named aidrin into the code registry. The executable is at
<AIDRIN_BIN>. Run "<AIDRIN_BIN> --help" and "<AIDRIN_BIN> list" to discover its subcommands and the
15 metrics, then save a code spec named aidrin describing the run/batch/data-quality subcommands
and their positional arguments.
```

**Verify:** `Search the registry for the aidrin data-readiness code.` →
`~/dsagt-projects/cryoem-readiness/codes/aidrin/SKILL.md` should exist.

### 2. Readiness assessment BEFORE curation

```text
Using the registry aidrin code, assess the AI data readiness of data/cryoem_before.csv. Run these
metrics through dsagt-run: completeness; duplicity; outliers; correlations on the columns
"Defocus U,Defocus V,Defocus Angle,CTF B Factor,Origin X (Ang),Origin Y (Ang)"; class-imbalance on
the Class Number column; and feature-relevance with no categorical columns, those same numerical
columns, and target column selected. Summarize the data-quality and class-balance state.
```

### 3. Readiness assessment AFTER curation, and compare

```text
Now run the same data-quality metrics (completeness, duplicity, outliers) and class-imbalance on
Class Number for data/cryoem_after.csv, through dsagt-run. Then compare before vs after and tell me
whether curation improved AI-readiness, citing the specific scores that changed.
```

**Expect** (positional args, JSON to stdout):

| Metric | before → after | Reading |
|---|---|---|
| `completeness` (overall) | 1.0 → 1.0 | already complete |
| `duplicity` | 0.0 → 0.0 | no duplicate particles |
| `outliers` (overall) | **0.041 → 0.029** | curation removed ~30% of outliers |
| `class-imbalance` (Class Number) | **22.2 → 11.1** | class distribution markedly more balanced |

Curation produced a cleaner, more balanced particle set — a **measurable** AI-readiness gain, with
`outliers` and `class-imbalance` as the headline indicators and `completeness`/`duplicity`
confirming the data was already structurally sound.

### 4. Generate a datacard for the curated dataset

```text
Search for a skill that can generate a datacard for the curated cryo-EM dataset at
data/cryoem_after.csv, then use it to produce the datacard.
```

The agent discovers the `datacard-generator` skill and writes a Genesis Datacard (e.g.
`data/genesis_datacard_*.md`) documenting the curated dataset.

### 5. Reconstruct the pipeline

```text
Reconstruct the readiness assessment you just ran from the execution records as a bash script.
```

## Post-Conditions

1. Code registry contains the `aidrin` spec (`codes/aidrin/SKILL.md`).
2. `trace_archive/` holds one provenance record per metric run (before and after).
3. Before/after scores show curation reduced outliers (~0.041 → ~0.029) and class imbalance
   (~22.2 → ~11.1) while completeness and duplicity stayed clean.
4. A datacard for the curated dataset exists (`data/genesis_datacard_*.md`).
5. A reconstructed pipeline script replays the readiness checks in order.
6. MLflow traces capture token usage, latency, and the `dsagt-run` / MCP spans.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| External-CLI registration (`save_code_spec`) | 1 |
| Registry search | 1 (Verify) |
| Code execution with provenance (`dsagt-run` → `trace_archive/`) | 2, 3 |
| Before/after comparison driven by the agent | 3 |
| Skill discovery and use (datacard generation) | 4 |
| Pipeline reconstruction from execution records | 5 |
| Observability (MLflow spans in the serverless `mlflow.db` store) | all |

View the traces any time with
`mlflow ui --backend-store-uri sqlite:///$PROJ/mlflow.db`.

## Cleanup

```bash
dsagt rm cryoem-readiness -y
rm -rf demo_data/cryoem AIDRIN aidrin-venv
```
