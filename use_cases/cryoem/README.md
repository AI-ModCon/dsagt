---
title: Cryo-EM
domain: Structural biology — EMPIAR-10017 β-galactosidase micrographs via CryoPPP
summary: >-
  DSAgt-assisted curation of cryo-EM data from the EMPIAR public archive
  (EMPIAR-10017 β-galactosidase micrographs via CryoPPP) — register curation
  codes, ingest cryo-EM quality knowledge, and build a micrograph-preprocessing
  pipeline, with the AIDRIN readiness assessment measuring the curation step
  before/after.
status: published
order: 20
---

# DSAgt Demo: Cryo-EM Data Curation Pipeline

> **Estimated time:** ~45 minutes of session time, plus the download — this is
> the broadest demo (7 stages). It pulls a **~22 GB data download** (84
> micrographs and their particle stacks), two open-access papers, and a full
> CryoPPP repo clone, then KB-ingests the whole repo (minutes on the local
> embedder) before any pipeline work. The one-time AIDRIN install at
> `dsagt init` adds a few minutes on first use.

This guide documents a comprehensive DSAgt demonstration using cryo-electron microscopy (cryo-EM) data. It exercises knowledge ingestion, KB-guided pipeline design, code registration from third-party scripts, multi-stage pipeline execution with domain-specific evaluation, and the [readiness assessment](../../docs/readiness.md): with the assessment enabled at init, the agent runs AIDRIN readiness metrics before and after the tabular curation step on its own, so the pipeline's AI-readiness gain is *measured*. The walkthrough has been run end to end with Claude Code on Sonnet 4.5.

## Prerequisites

- DSAgt installed (`uv sync --all-groups`)
- An agent platform installed and **already authenticated** (e.g., `claude` for Claude Code)
- `uv` installed. Enabling the readiness assessment at `dsagt init` installs AIDRIN itself
  (one-time, shared across projects, Python 3.10-3.12)
- ~22 GB disk space for the cryo-EM test data
- Git installed

## Setup

### 1. Initialize a DSAgt project

```bash
dsagt init
```

At the menu, name the project `cryoem-pipeline`, pick your agent, and answer **yes** to
"Enable the AIDRIN AI-readiness assessment?". Init installs AIDRIN on first use (one-time, shared
across projects, into `~/dsagt-projects/.tools/`). The defaults are fine for the rest. Then:

```bash
PROJ=~/dsagt-projects/cryoem-pipeline
```

### 2. Download the data, papers, and CryoPPP repository into the project

The agent runs with the project directory as its working directory, so everything it
reads goes under `$PROJ`.

```bash
mkdir -p "$PROJ/data/cryoem/papers" "$PROJ/repos"
curl -L https://calla.rnet.missouri.edu/cryoppp/10017.tar.gz | tar xz -C "$PROJ/data/cryoem"
# CryoPPP paper: Dhakal et al., Scientific Data 2023 (open access)
curl -L https://www.nature.com/articles/s41597-023-02280-2.pdf -o "$PROJ/data/cryoem/papers/cryoppp_paper.pdf"
# CryoCRAB paper: Chen et al., Scientific Data 2025 (open access) — defines the 0-7 micrograph quality score
curl -L https://www.nature.com/articles/s41597-025-05179-2.pdf -o "$PROJ/data/cryoem/papers/cryocrab_paper.pdf"
git clone https://github.com/BioinfoMachineLearning/cryoppp.git "$PROJ/repos/cryoppp"
```

The EMPIAR-10017 (β-galactosidase) subset of the CryoPPP dataset is 84 micrographs. It also
includes ground-truth particle tables with real CTF/defocus columns and a selected-vs-excluded
curation split; the pipeline merges and then curates them, so every data operation runs inside
the session with provenance.

### 3. Start the session

```bash
dsagt start cryoem-pipeline
```

## Execution

Paste these prompts one at a time. Init installed the `aidrin` skill at
`skills/aidrin/SKILL.md` and the readiness assessment put its rules in the instructions file, so the agent runs the assessment around
the tabular steps without being told to; the micrograph (image) steps are not assessed.

### 1. Create a cryo-EM knowledge collection

```text
Ingest the folder repos/cryoppp/ into the knowledge base as a collection called "cryoppp".
```

Wait for the ingest job to complete, then:

```text
Append the files data/cryoem/papers/cryoppp_paper.pdf and data/cryoem/papers/cryocrab_paper.pdf
to the cryoppp collection.
```

**Verify:**

```text
List all knowledge base collections.
```

Should show `cryoppp`.

### 2. Query the knowledge base for pipeline design

```text
Search the cryoppp collection for guidance on creating an AI-ready data processing pipeline for cryo-EM micrographs. What quality parameters should I filter on, and what thresholds are recommended?
```

The agent should return chunks describing quality metrics: CTF resolution, defocus ranges, ice thickness thresholds, and motion statistics.

### 3. Register CryoPPP processing codes

```text
Look at the scripts in repos/cryoppp/ and register any data-processing or evaluation codes you find. Run --help on each script to discover its interface.
```

**Verify:**

```text
Search the registry for cryo-EM codes.
```

### 4. Create a quality scoring code

```text
Write a Python script that scores cryo-EM micrographs based on:
- CTF fit resolution (CTFMaxRes)
- Defocus range
- Ice thickness
- Motion statistics

Use the CryoCRAB 0-7 scoring scheme from the CryoCRAB paper in the cryoppp collection: each of
its seven screening parameters within the dataset's 3-sigma interval contributes one point, and
scores map to tiers low (0-2), medium (3-5), high (6-7). Score on the parameters available in our
metadata. The script should read a metadata CSV and output a scored CSV with quality_score and
quality_tier columns. Save the script under codes/<name>/scripts/ and register it as a code.
```

The agent should search the knowledge base, write the script, and register it via `save_code_spec`.

### 5. Run the pipeline

```text
Run the pipeline on the EMPIAR-10017 dataset in data/cryoem/10017/:
1. Scan the directory to understand what's there
2. Profile the micrograph metadata
3. Run the quality scoring code on the metadata
4. Merge the two ground-truth particle tables in data/cryoem/10017/ground_truth/ into
   data/cryoem/particles.csv, adding a selected flag (1 for the selected table, 0 for excluded)
5. Curate the merged table: keep only rows with selected == 1, drop the selected column, and
   write data/cryoem/particles_curated.csv
6. Summarize: how many micrographs fall into each quality tier, and did curation improve
   the particle data?
```

Steps 4 and 5 are where the assessment shows: without any AIDRIN prompt, the agent runs the readiness
metrics around each tabular operation — the merge (a no-op delta, which is itself informative)
and the curation filter, where `particles.csv` before and `particles_curated.csv` after differ.
Both reports land in `audit/`. Expected across the curation step:

| Metric | before → after | Reading |
|---|---|---|
| `completeness` (overall) | 1.0 → 1.0 | already complete |
| `duplicity` | 0.0 → 0.0 | no duplicate particles |
| `outliers` (overall) | **0.041 → 0.029** | curation removed ~30% of outliers |
| `class-imbalance` (Class Number, passthrough) | **22.2 → 11.1** | markedly more balanced |

Curation produced a cleaner, more balanced particle set — a **measurable** AI-readiness gain, with
`outliers` (and `class-imbalance`, if the agent proposes it) as the headline indicators and
`completeness`/`duplicity` confirming the data was already structurally sound.

### 6. Generate a datacard

```text
Search for a skill that can generate a datacard for the curated cryo-EM data, then use it.
```

### 7. Reconstruct the pipeline

```text
Reconstruct the pipeline from the execution records as a bash script.
```

## Post-Conditions

1. Knowledge base contains `cryoppp` collection with repo code, docs, and appended papers.
2. `skills/aidrin/` is present (installed at init); the code registry includes the CryoPPP processing codes and the quality-scoring code.
3. Quality-scored CSV exists with tier distribution; `particles.csv` (merged) and `particles_curated.csv` (curated) exist with `trace_archive/` records for both operations.
4. `audit/` holds the assessment's pre/post AIDRIN reports for the merge and curation steps, and the scores show curation reduced outliers (~0.041 → ~0.029).
5. A datacard exists for the processed dataset.
6. A reconstructed pipeline script is available.
7. Code execution records in `trace_archive/` document the full provenance chain, including one record per metric of each assessment run.
8. MLflow traces (in the serverless `mlflow.db` store) capture token usage, latency, and full request/response history.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Knowledge ingestion (folder) | 1 |
| Knowledge append (single file) | 1 |
| Semantic search | 2 |
| Code discovery via registry | 3 |
| Code registration | 3, 4 |
| KB-guided code generation | 4 |
| Code execution with provenance | 5 |
| Readiness assessment run unprompted (before/after AIDRIN on the tabular step) | 5 |
| Skill discovery and use | 6 |
| Pipeline reconstruction | 7 |

## Cleanup

```bash
dsagt rm cryoem-pipeline -y          # unregisters the project and removes its dir, data included
```
