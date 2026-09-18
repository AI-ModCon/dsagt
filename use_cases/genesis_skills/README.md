---
title: Genesis Skills for Data Curation
domain: Skill management — the external Genesis skill catalog driving a data-curation pipeline
summary: >-
  Sync the Genesis skill catalog, install the Croissant validation skill,
  ground the curation skills in the dataset's domain documents, and produce a
  datacard for a small curated dataset.
status: published
order: 60
---

# DSAgt Demo: Genesis Skills for a Data-Curation Pipeline

> **Estimated time:** ~10 minutes (the data is tiny; the one external
> dependency is a shallow clone of the Genesis catalog from GitHub — needs
> network access to `github.com`).

An end-to-end **data-preparation** walkthrough that exercises the skill catalog
against the **Genesis** source (AI-ModCon on GitHub). The agent installs the
BASE-Data/ModCon Croissant validator from the catalog, grounds the
`datacard-generator` base skill every project carries in the dataset's domain
documents, then prepares and **datacards a finished dataset**.

The "finished product" is a small curated dataset — a CO2-methanation **catalyst
screen** (`dataset/catalyst_screening.csv`, 8 rows) — plus the domain docs that
describe how it was produced. Everything is tiny, so the whole thing runs in
seconds with no real instruments or HPC.

## Prerequisites

- DSAgt installed and an agent platform installed and
  **already authenticated**.
- Git, with network access to `github.com` (the Genesis catalog clones from
  `AI-ModCon/genesis-skills`).
- Embedding credentials are optional — `search_skills` uses semantic search
  when `EMBEDDING_*` is set and falls back to a keyword scorer otherwise.

## Setup

```bash
dsagt init
```

At the menu, name the project `genesis-skills`, pick your agent, and **uncheck `genesis`**
at the skill-sources checkbox — the walkthrough has the agent enable that catalog itself
in step 1. Then:

```bash
PROJ=~/dsagt-projects/genesis-skills
# The fixture data (catalyst_screening.csv and the domain docs) is in the
# repository, under this use case's data/ folder. The expected datacard stays
# out of the project: it is the reference you compare against afterwards.
mkdir -p "$PROJ/mock_data"
cp -r use_cases/genesis_skills/data/dataset use_cases/genesis_skills/data/domain "$PROJ/mock_data/"
# $PROJ/mock_data now holds dataset/ and domain/
dsagt start genesis-skills
```

## Execution

Paste each prompt into the agent (running inside the project), one at a time.
Confirmation checks are consolidated in **Post-Conditions** below.

### 1. Enable the Genesis source

```text
Enable the "genesis" skill source so we have the GENESIS / ModCon data-curation skills available. Then tell me how many skills it indexed.
```

**Expect:** `add_skill_source(source="genesis")` → a shallow clone from GitHub,
its skills indexed, source written to `.dsagt/config.yaml`.

### 2. Find and install the validator skill

```text
Search the catalog for a skill that validates Croissant / JSON-LD dataset metadata and install the best match into this project.
```

**Expect:** `search_skills` surfaces **`croissant-validator`** → `install_skill`.
It is installed into `<project>/skills/` and mirrored into the agent's native
skills directory at install time, with a `PROVENANCE.txt` crediting the Genesis
source. `datacard-generator` needs no install: it is a base skill, present
since init.

### 3. Generate the datacard for the finished dataset

```text
Use the datacard-generator skill to write a Level 1 datacard for mock_data/dataset/catalyst_screening.csv. Pull the field definitions, measurement methodology, provenance, and license from the data dictionary and measurement protocol under mock_data/domain/ — don't invent them, and note anything the documents leave unspecified rather than asking. Save it to audit/catalyst_screening_datacard.md.
```

**Expect:** the agent reads the installed skill's `SKILL.md` and the two domain
documents (reactor conditions **250 °C, 1 atm, H2:CO2 = 4:1, GHSV 12,000**;
license **CC-BY-4.0**), computes basic stats from the 8-row CSV, and writes
`audit/catalyst_screening_datacard.md` covering summary / provenance / schema /
methodology / stats / limitations / license.

### 4. Validate the metadata

```text
Use the croissant-validator skill to check the Croissant/JSON-LD metadata for this dataset (generate it from the datacard if needed), and report any schema errors.
```

**Expect:** the validator skill runs and reports a clean pass or names specific
schema issues. The skill installs `mlcroissant` into a small virtual environment
for its library check; a pass is one whose output shows `mlcroissant parse OK`,
since the script skips that check when the library is absent.

## Post-Conditions

Confirm from a shell (the native skills directory is `.claude/skills/` for Claude Code,
`.agents/skills/` for Codex, Goose, and opencode, `.cline/skills/` for Cline):

```bash
dsagt info genesis-skills                  # KB lists skills_catalog__ai-modcon-genesis-skills
ls "$PROJ/skills/"                         # aidrin  croissant-validator  datacard-generator  skill-creator
cat "$PROJ/skills/croissant-validator/PROVENANCE.txt"
ls "$PROJ/audit/"                          # catalyst_screening_datacard.md
diff <(grep '^#' use_cases/genesis_skills/data/expected_datacard.md) <(grep '^#' "$PROJ/audit/catalyst_screening_datacard.md")
```

1. The KB holds a `skills_catalog__ai-modcon-genesis-skills` collection
   (searchable via `search_skills`).
2. `croissant-validator` is installed into `<project>/skills/` and mirrored
   into the agent's native skills directory, with a `PROVENANCE.txt` crediting
   the Genesis source; `datacard-generator` has been there since init as a
   base skill. The next session auto-invokes them natively; this session used
   them by reading their `SKILL.md`.
3. `audit/catalyst_screening_datacard.md` was produced for the finished dataset,
   grounded in the domain documents, covering the sections in
   `use_cases/genesis_skills/data/expected_datacard.md` and carrying its values:
   reactor conditions 250 °C, 1 atm, H2:CO2 = 4:1, GHSV 12,000; license
   CC-BY-4.0; 8 rows; and the three caveats the measurement protocol states.
4. The validator's output shows `mlcroissant parse OK`.
5. MLflow traces (in the serverless `mlflow.db` store) capture the session —
   `dsagt traces genesis-skills`.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Enabling an external skill source in-session (`add_skill_source`) | 1 |
| Catalog search and install (`search_skills`, `install_skill`) | 2 |
| Native mirroring of installed skills | 2 |
| Base-skill use (`datacard-generator`) | 3 |
| Installed-skill execution grounded in the domain documents | 3, 4 |
| Review of the session's artifacts | 5 |

## Cleanup

```bash
dsagt rm genesis-skills -y
```

The shared catalog cache is stored at `~/dsagt-projects/.skill_sources/` and is
reused across projects; delete it to force a fresh clone.
