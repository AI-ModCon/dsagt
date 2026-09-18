---
title: Microbial Isolates
domain: Genomics, short-read QC and assembly with `fastp` + `megahit`
summary: >-
  Register short-read QC and assembly codes, follow the genomics best-practice
  documents, and build a reproducible isolate-processing pipeline against real
  sequencing reads.
status: published
order: 10
---

# DSAgt Demo: Microbial Isolate Processing

> **Estimated time:** advanced / not a 10-minute demo. The isolate reads are a
> 3.9 GB download (step 3), and `megahit` assembly runs minutes per sample across
> 11 isolates.

This guide documents a reproducible DSAgt demonstration for microbial isolate data processing using `fastp` and `megahit`.

## Prerequisites

- DSAgt installed
- An agent platform installed and **already authenticated** (e.g., `claude` for Claude Code, or
  `goose`).
- fastp and megahit, installed by [`scripts/setup_env.sh`](scripts/setup_env.sh)
  (see Setup); conda is optional, and the script fetches micromamba when neither
  conda nor micromamba is present

## Setup

### 1. Install fastp and megahit

fastp and megahit are C/C++ programs from Bioconda, not pip packages. The
walkthrough's bundle (the reads, the two documents, and the setup script; 3.9 GB,
from the DSAgt use-case data folder,
https://drive.google.com/drive/folders/1RWQAJeHaikIaD7CCf8ciJ71m55S1erp6) holds a
setup script that creates a conda environment for them from its
`environment.yml` under the shared DSAgt tools directory, using conda if present
and a downloaded micromamba otherwise:

```bash
curl -L "https://drive.usercontent.google.com/download?id=1xabVTGy6W3vx55sRBVrPLFqNc_LqG1hz&export=download&confirm=t" \
  -o microbial_isolates.tar.gz
mkdir -p isolates_bundle && tar xzf microbial_isolates.tar.gz -C isolates_bundle ./setup
bash isolates_bundle/setup/setup_env.sh
```

It prints the environment's `bin` directory
(`~/dsagt-projects/.tools/microbial_isolates/env/bin`); that path is the
`<CONDA_PREFIX>` in the prompts below.

### 2. Initialize a DSAgt project

```bash
dsagt init
```

At the menu, name the project `isolate-pipeline` and pick your agent; the defaults are fine
for the rest. Then:

```bash
PROJ=~/dsagt-projects/isolate-pipeline
```

(The default local embedder needs no key. To use a hosted embedder instead, set
`embedding.backend: api` in `$PROJ/.dsagt/config.yaml` and export `EMBEDDING_API_KEY`
in your shell; dsagt never writes it to disk.)

### 3. Collect data and reference material into the project

The agent runs with the project directory as its working directory, so everything it
reads goes under `$PROJ`. The bundle's two documents describe the processing
pipeline and the fastp and megahit parameter choices; the agent reads them
directly.

```bash
# The bundle downloaded in step 1: 11 interleaved FASTQ files with their README
# (data/microbial_isolate/) and the two documents (docs/).
tar xzf microbial_isolates.tar.gz -C "$PROJ" --exclude='./setup'
```

### 4. Start the session

```bash
dsagt start isolate-pipeline
```

The agent launches from the project directory with the MCP server connected; there are no background services to clean up.

## Execution

Use these prompts in the agent session. Replace `<CONDA_PREFIX>` with the bin directory the setup script printed (`~/dsagt-projects/.tools/microbial_isolates/env/bin`). Data and docs paths are relative to the project directory.

### 1. Register codes

```text
Let's add <CONDA_PREFIX>/fastp to the registry
Let's add <CONDA_PREFIX>/megahit to the registry
```

**Verify:**

```text
Search the registry for assembly codes.
```

### 2. Process one sample

```text
I have an isolate file at data/microbial_isolate/53162.2.609630.AAAGGCTAGA-GATTCAGTTA.filter-ISO.fastq.gz
Information about the dataset is in the README in that directory. I need to preprocess this file and assemble it.
Follow docs/genomics.md for the processing pipeline and docs/fastp_megahit_best_practices.md for parameter choices.
fastp and megahit both have data assessment capability so we don't need to create additional codes.
megahit should be run with kmax=21 and a 3 GB memory cap to avoid OOM on this laptop.
Write the trimmed reads and the fastp reports to data/processed/<sample>/ and the assembly to
data/assemblies/<sample>/.
Tell me your plan before proceeding.
```

The reads are interleaved paired-end files with adapters already removed, and the
best-practices document gives the `fastp --interleaved_in` form and `megahit -t 1 --no-hw-accel`
for this machine; a plan that follows the document uses both.

### 3. Approve the plan and process the sample

```text
Go ahead and process that sample.
```

Each fastp and megahit run is a registered-code execution, run in the foreground and recorded
in `trace_archive/` when it exits; an assembly takes two to four minutes.

### 4. Process remaining samples

```text
Let's run this same pipeline on the rest of the fastq files at data/microbial_isolate/
```

Processing the remaining ten samples takes 30 to 40 minutes on a laptop; the agent runs
them to completion before replying.

### 5. Generate datacard

```text
Use the datacard-generator skill to write a Level 1 datacard for the assembled data under
data/assemblies/. Take the values from the data and the reports, and note anything unknown
rather than asking.
```

`datacard-generator` is a base skill, installed at init and mirrored into the agent's native skills directory, so the agent invokes it without a catalog search. Level 1 means discoverability only: the card sets `supports_discoverability` and no other capability flag.

### 6. Reconstruct pipeline

```text
Reconstruct the pipeline from the execution records as a bash script.
```

The agent calls `reconstruct_pipeline` with an `output` path under the project, and the tool
renders the `trace_archive/` records as a bash script in the order they ran and saves it there.

### 7. Review the project artifacts

```text
Show me the contents of my project folder in a tree format, with the artifacts dsagt recorded during this session highlighted.
```

**Expect:** a listing of the project directory that marks the execution records in
`trace_archive/`, the reports in `audit/`, the registered codes and installed skills under
`skills/`, the trace store `mlflow.db`, the reconstructed script, and the session's outputs,
with a line on what each is. The reply may summarize a tree printed by a command.

## Post-Conditions

1. Code registry includes `fastp` and `megahit` code specs (wrapped with `dsagt-run`).
2. `data/processed/<sample>/` and `data/assemblies/<sample>/` exist for each isolate sample.
3. For each completed sample:
   - the trimmed R1 and R2 FASTQ files are under `data/processed/<sample>/`
   - the `fastp` HTML and JSON reports are beside them
   - `data/assemblies/<sample>/final.contigs.fa` exists
4. A Level 1 datacard exists for the processed dataset and validates with the registered `datacard-validate` code with no findings.
5. A reconstructed pipeline script (bash or Snakemake) is available.
6. Code execution records in `trace_archive/` document the full provenance chain.
7. MLflow traces (in the serverless `mlflow.db` store) capture token usage, latency, and full request/response history. View with `dsagt traces isolate-pipeline`.

`megahit` segfaults on Apple Silicon with more than one thread; the best-practices document says `-t 1 --no-hw-accel`, and a sample that segfaulted is rerun that way with the same `kmax=21` and memory cap.

## Coverage

| DSAgt Capability | Steps |
|------------------|-------|
| Registering external binaries as codes | 1 |
| Registry search | 1 |
| Pipeline planning from best-practice documents, confirmed with the user | 2, 3 |
| Code execution with provenance across many samples | 3, 4 |
| Base-skill use (`datacard-generator`) | 5 |
| Pipeline reconstruction | 6 |
| Review of the session's artifacts | 7 |

## Cleanup

```bash
dsagt rm isolate-pipeline -y
rm -r microbial_isolates.tar.gz isolates_bundle
rm -rf ~/dsagt-projects/.tools/microbial_isolates     # the fastp/megahit environment
```
