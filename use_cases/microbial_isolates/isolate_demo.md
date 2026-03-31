# DSAGT Demo: Microbial Isolate Processing

This guide documents a reproducible DSAGT demonstration for microbial isolate data processing using `fastp` and `megahit`.

## Setup

### 1. Clone and install DSAGT

```bash
git clone <your-dsagt-repo-url> dsagt
cd dsagt
uv sync --all-groups
```

### 2. Prepare Goose in the DSAGT project

```bash
cp agents/goose/.goosehints .goosehints
```

Make sure your shell has API credentials for Goose, for example:

```bash
export OPENAI_API_KEY="<your-key>"
# Optional for OpenAI-compatible non-OpenAI endpoints:
export OPENAI_HOST="<your-openai-compatible-endpoint>"
```

### 3. Collect Required Assets

From the DSAGT project root, run:

```bash
# Create demo folders
mkdir -p DSAGT/demo/repos DSAGT/demo/data/microbial_isolate

# Clone required repos into the demo folder
git clone https://github.com/OpenGene/fastp.git DSAGT/demo/repos/fastp
git clone https://github.com/voutcn/megahit.git DSAGT/demo/repos/megahit
git clone git@github.com:AI-ModCon/BaseData_Skills.git DSAGT/demo/repos/BaseData_Skills

# Copy isolate data from NERSC into the demo folder
scp <nersc-username>@dtn01.nersc.gov:/global/cfs/projectdirs/amsc002/base_data/example_famous_data/* DSAGT/demo/data/microbial_isolate/

# Install fastp + megahit in conda env via Bioconda
conda create -n isolate -c conda-forge -c bioconda fastp megahit -y
conda run -n isolate fastp --version
conda run -n isolate megahit --version
```

Expected assets:

- `DSAGT/demo/repos/fastp/`
- `DSAGT/demo/repos/megahit/`
- `DSAGT/demo/repos/BaseData_Skills/`
- `DSAGT/demo/genomics.md`
- `DSAGT/demo/fastp_megahit_best_practices.md`
- `DSAGT/demo/data/microbial_isolate/*.filter-ISO.fastq.gz`

### 4. Run demo session

From DSAGT project root:

```bash
goose session \
  --with-extension 'uv run dsagt-registry-server' \
  --with-extension 'uv run dsagt-knowledge-server'
```

## Execution

Use these prompts in Goose. Replace placeholders with your local values:

- `<PROJECT_ROOT>`: absolute path to your DSAGT checkout
- `<CONDA_ENV_PREFIX>`: conda envs directory (example: `~/miniconda3/envs`)

```text
I'd like to create a new collection in the knowledge base: microbial_isolates.
The collection will contain
1) the code package files for fastp: <PROJECT_ROOT>/DSAGT/demo/repos/fastp/
2) the code package files for megahit: <PROJECT_ROOT>/DSAGT/demo/repos/megahit/ and
3) a short document describing a processing pipeline: <PROJECT_ROOT>/DSAGT/demo/genomics.md
4) A document describing best practices for using fastp and
megahit: <PROJECT_ROOT>/DSAGT/demo/fastp_megahit_best_practices.md

Let's add <CONDA_ENV_PREFIX>/isolate/bin/fastp to the registry
Let's add <CONDA_ENV_PREFIX>/isolate/bin/megahit to the registry

Okay good I have an isolate file located at
<PROJECT_ROOT>/DSAGT/demo/data/microbial_isolate/53162.2.609630.AAAGGCTAGA-GATTCAGTTA.filter-ISO.fastq.gz
Information about the dataset is contained in the Readme in that directory. I need to preprocess this file and assemble it.
fastp and megahit both have data assessment capability so we don't need to create additional tools.
megahit should be run with kmax=21 and memory=0.3 to avoid OOM on this laptop
Tell me your plan before proceeding.

Let's run this same pipeline the rest of the fastq files at
<PROJECT_ROOT>/DSAGT/demo/data/microbial_isolate/
We can process them one at a time. We don't need to create a script for batch processing

Use the skill located at <PROJECT_ROOT>/DSAGT/demo/repos/BaseData_Skills/datacard-generator/ to create a datacard for the processed microbial isolate data
```

## Post-Conditions

1. Knowledge base contains collection `microbial_isolates` with all listed references indexed.
2. Runtime registry includes `fastp` and `megahit` tool specs.
3. Processed output directories exist for target isolate samples under:
   - `DSAGT/demo/data/processed_microbial_isolate/`
4. For each completed sample:
   - preprocessed FASTQ output exists
   - `fastp` HTML and JSON reports exist
   - assembly output exists, including `final.contigs.fa`
5. A Level 1 datacard exists for the processed dataset.

### Note

`megahit` may intermittently fail with segmentation faults on some files/hardware settings. If this occurs, rerun that sample with conservative settings while preserving the required `kmax=21` and laptop-safe memory cap.
