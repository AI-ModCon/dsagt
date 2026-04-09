# DSAgt Demo: Microbial Isolate Processing

This guide documents a reproducible DSAgt demonstration for microbial isolate data processing using `fastp` and `megahit`.

## Prerequisites

- DSAgt installed (`uv sync --all-groups`)
- An agent platform installed (e.g., `claude` for Claude Code, or `goose`)
- `test_site_config.yaml` configured with valid API keys and embedding endpoint
- Conda (for installing fastp and megahit)

## Setup

### 1. Install bioinformatics tools

fastp and megahit are C/C++ tools installed via Bioconda, not pip:

```bash
conda create -n isolate -c conda-forge -c bioconda fastp megahit -y
conda run -n isolate fastp --version
conda run -n isolate megahit --version
```

Note the conda env prefix (e.g., `~/miniconda3/envs/isolate/bin/`) — you'll reference these paths when registering the tools.

### 2. Collect data

```bash
# Copy isolate data (from NERSC or local source)
mkdir -p demo_data/microbial_isolate
scp <nersc-username>@dtn01.nersc.gov:/global/cfs/projectdirs/amsc002/base_data/example_famous_data/* demo_data/microbial_isolate/
```

### 3. Clone reference repos (for knowledge base)

```bash
mkdir -p demo_repos
git clone https://github.com/OpenGene/fastp.git demo_repos/fastp
git clone https://github.com/voutcn/megahit.git demo_repos/megahit
```

### 4. Initialize a DSAgt project

```bash
dsagt init isolate-pipeline --agent claude-code
```

Edit `~/dsagt-projects/isolate-pipeline/dsagt_config.yaml` — set your API keys and embedding endpoint.

### 5. Start the session

```bash
dsagt start isolate-pipeline
```

The agent launches from the project directory with MCP servers connected. Services clean up automatically when the agent exits.

## Execution

Use these prompts in the agent session. Replace `<CONDA_PREFIX>` with your conda env bin path (e.g., `~/miniconda3/envs/isolate/bin`) and `<DEMO_DATA>` / `<DEMO_REPOS>` with your local paths.

### 1. Build knowledge base

```text
I'd like to create a new collection in the knowledge base: microbial_isolates.
The collection will contain:
1) the code package files for fastp: <DEMO_REPOS>/fastp/
2) the code package files for megahit: <DEMO_REPOS>/megahit/
3) a short document describing a processing pipeline: use_cases/microbial_isolates/genomics.md
4) best practices for fastp and megahit: use_cases/microbial_isolates/fastp_megahit_best_practices.md
```

### 2. Register tools

```text
Let's add <CONDA_PREFIX>/fastp to the registry
Let's add <CONDA_PREFIX>/megahit to the registry
```

### 3. Process one sample

```text
I have an isolate file at <DEMO_DATA>/microbial_isolate/53162.2.609630.AAAGGCTAGA-GATTCAGTTA.filter-ISO.fastq.gz
Information about the dataset is in the README in that directory. I need to preprocess this file and assemble it.
fastp and megahit both have data assessment capability so we don't need to create additional tools.
megahit should be run with kmax=21 and memory=0.3 to avoid OOM on this laptop.
Tell me your plan before proceeding.
```

### 4. Process remaining samples

```text
Let's run this same pipeline on the rest of the fastq files at <DEMO_DATA>/microbial_isolate/
We can process them one at a time.
```

### 5. Generate datacard

```text
Search for a skill that can generate a datacard for our processed data, then use it.
```

The agent should find the bundled `datacard-generator` skill via `search_skills`.

### 6. Reconstruct pipeline

```text
Reconstruct the pipeline from the execution records as a bash script.
```

The agent calls `reconstruct_pipeline` to generate a reproducible script from the trace archive.

## Post-Conditions

1. Knowledge base contains collection `microbial_isolates` with all listed references indexed.
2. Tool registry includes `fastp` and `megahit` tool specs (wrapped with `dsagt-run`).
3. Processed output directories exist for each isolate sample.
4. For each completed sample:
   - Preprocessed FASTQ output exists
   - `fastp` HTML and JSON reports exist
   - Assembly output exists, including `final.contigs.fa`
5. A Level 1 datacard exists for the processed dataset.
6. A reconstructed pipeline script (bash or Snakemake) is available.
7. Tool execution records in `trace_archive/` document the full provenance chain.
8. MLflow traces capture token usage, latency, and full request/response history.

### Note

`megahit` may intermittently fail with segmentation faults on some files/hardware settings. If this occurs, rerun that sample with conservative settings while preserving the required `kmax=21` and laptop-safe memory cap.
