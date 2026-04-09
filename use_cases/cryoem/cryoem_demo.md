# DSAgt Demo: Cryo-EM Data Curation Pipeline

This guide documents a comprehensive DSAgt demonstration using cryo-electron microscopy (cryo-EM) data. It exercises knowledge ingestion, KB-guided pipeline design, tool registration from third-party code, cross-collection knowledge synthesis, and multi-stage pipeline execution with domain-specific evaluation.

## Prerequisites

- DSAgt installed (`uv sync --all-groups`)
- An agent platform installed (e.g., `claude` for Claude Code)
- `test_site_config.yaml` configured with valid API keys and embedding endpoint
- ~2 GB disk space for the cryo-EM test data
- Git installed

## Setup

### 1. Download cryo-EM dataset

Download the EMPIAR-10017 (β-galactosidase) subset from the CryoPPP dataset — 84 micrographs.

```bash
mkdir -p demo_data/cryoem
cd demo_data/cryoem
curl -L https://calla.rnet.missouri.edu/cryoppp/10017.tar.gz -o 10017.tar.gz
tar xzf 10017.tar.gz
cd ../..
```

### 2. Download the CryoPPP paper

```bash
mkdir -p demo_data/cryoem/papers
curl -L https://arxiv.org/pdf/2304.02011 -o demo_data/cryoem/papers/cryoppp_paper.pdf
```

### 3. Clone the CryoPPP repository

```bash
mkdir -p demo_repos
git clone https://github.com/BioinfoMachineLearning/cryoppp.git demo_repos/cryoppp
```

### 4. Initialize a DSAgt project

```bash
dsagt init cryoem-pipeline --agent claude-code
```

Edit `~/dsagt-projects/cryoem-pipeline/dsagt_config.yaml` — set your API keys and embedding endpoint.

### 5. Start the session

```bash
dsagt start cryoem-pipeline
```

## Execution

### 1. Create a cryo-EM knowledge collection

```text
Ingest the folder demo_repos/cryoppp/ into the knowledge base as a collection called "cryoppp".
```

Wait for the ingest job to complete, then:

```text
Append the file demo_data/cryoem/papers/cryoppp_paper.pdf to the cryoppp collection.
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

### 3. Register CryoPPP processing tools

```text
Look at the scripts in demo_repos/cryoppp/ and register any data processing or evaluation tools you find. Run --help on each script to discover its interface.
```

**Verify:**

```text
Search the registry for cryo-EM tools.
```

### 4. Create a quality scoring tool

```text
Write a Python script that scores cryo-EM micrographs based on:
- CTF fit resolution (CTFMaxRes)
- Defocus range
- Ice thickness
- Motion statistics

Use the CryoCRAB 0-7 scoring scheme described in the cryoppp collection. The script should read a metadata CSV and output a scored CSV with quality_score and quality_tier (high/medium/low) columns. Save the script under tools/code/ and register it as a tool.
```

The agent should search the knowledge base, write the script, and register it via `save_tool_spec`.

### 5. Run the pipeline

```text
Run the pipeline on the EMPIAR-10017 dataset in demo_data/cryoem/10017/:
1. Scan the directory to understand what's there
2. Profile the micrograph metadata
3. Run the quality scoring tool on the metadata
4. Summarize: how many micrographs fall into each quality tier?
```

### 6. Generate a datacard

```text
Search for a skill that can generate a datacard for the processed cryo-EM data, then use it.
```

### 7. Reconstruct the pipeline

```text
Reconstruct the pipeline from the execution records as a bash script.
```

## Post-Conditions

1. Knowledge base contains `cryoppp` collection with repo code, docs, and appended paper.
2. Tool registry includes CryoPPP processing tools and the quality scoring tool.
3. Quality-scored CSV exists with tier distribution.
4. A datacard exists for the processed dataset.
5. A reconstructed pipeline script is available.
6. Tool execution records in `trace_archive/` document the full provenance chain.
7. MLflow traces capture token usage, latency, and full request/response history.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Knowledge ingestion (folder) | 1 |
| Knowledge append (single file) | 1 |
| Semantic search | 2 |
| Tool discovery via registry | 3 |
| Tool registration | 3, 4 |
| KB-guided code generation | 4 |
| Tool execution with provenance | 5 |
| Skill discovery and use | 6 |
| Pipeline reconstruction | 7 |

## Cleanup

```bash
rm -rf ~/dsagt-projects/cryoem-pipeline demo_data/cryoem demo_repos/cryoppp
```
