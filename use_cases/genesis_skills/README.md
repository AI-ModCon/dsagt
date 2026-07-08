# DSAgt Demo: Genesis Skills for a Data-Curation Pipeline

> **Estimated time:** ~10 minutes (all data is bundled and tiny; the one
> external dependency is a shallow clone of the Genesis catalog from OSTI
> GitLab — needs network access to `gitlab.osti.gov`).

An end-to-end **data-preparation** walkthrough that flexes the skill catalog
against the **Genesis** source (OSTI GitLab). The agent pulls in the
BASE-Data/ModCon curation skills, grounds itself in domain context loaded into
the **knowledge base**, then prepares and **datacards a finished dataset**.

The "finished product" is a small curated dataset — a CO2-methanation **catalyst
screen** (`mock_data/dataset/catalyst_screening.csv`, 8 rows) — plus the domain
docs that describe how it was produced. Everything is tiny, so the whole thing
runs in seconds with no real instruments or HPC.

## What this demonstrates

1. **Genesis skills, agent-facing.** `add_skill_source genesis` syncs the OSTI
   GitLab catalog; `search_skills` finds the ModCon curation skills
   (`generating-datacards`, `croissant-validator`); `install_skill` draws them
   into the project where the agent **natively** auto-discovers them.
2. **Domain → knowledge base.** `kb_ingest` indexes the data dictionary +
   measurement protocol into a project collection, so the agent describes
   methods and provenance accurately (via `kb_search`) instead of guessing.
3. **Datacard for a finished product.** The installed `generating-datacards`
   skill runs over the dataset — pulling field definitions, methodology, and
   license from the KB-ingested domain docs — and emits a datacard, then
   `croissant-validator` checks the Croissant metadata.

## Setup

Assumes DSAgt (`uv sync --all-groups`) and Claude Code
(`npm i -g @anthropic-ai/claude-code`) are installed, plus git **with network
access to `gitlab.osti.gov`** (the Genesis catalog clones from OSTI GitLab, not
GitHub). Embedding credentials are optional — `search_skills` / `kb_search` use
semantic search when `EMBEDDING_*` is set and fall back to a keyword scorer
otherwise (configure it for sharper relevance over the domain docs).

```bash
dsagt init genesis-skills --agent claude          # provisions the bundled codes/skills + core KB
cp -r use_cases/genesis_skills/mock_data ~/dsagt-projects/genesis-skills/mock_data
dsagt start genesis-skills
```

The Genesis catalog is not synced at `init` — the agent enables it in step 1.

## Walkthrough

Paste each prompt into Claude Code (running inside the project), one at a time.
Confirmation checks are consolidated in **Post-conditions** below.

### 1 — Enable the Genesis source

> Enable the "genesis" skill source so we have the GENESIS / ModCon data-curation skills available. Then tell me how many skills it indexed.

*Expect:* `add_skill_source(source="genesis")` → a shallow clone of OSTI GitLab,
**74** skills indexed, source written to `.dsagt/config.yaml`.

### 2 — Find and install the curation skills

> Search the catalog for two skills — one that creates a datacard / dataset documentation for a dataset, and one that validates Croissant / JSON-LD dataset metadata — and install the best match for each into this project.

*Expect:* `search_skills` surfaces **`generating-datacards`** and
**`croissant-validator`** → `install_skill` for each. The reply presents both
as installed and immediately usable (no restart caveat), each with a
`PROVENANCE.txt` crediting the Genesis source.

### 3 — Ingest the domain docs into the KB

> Ingest the domain docs under mock_data/domain/ into a new knowledge-base collection called "methanation_domain". Poll until it finishes, then tell me what's in it.

*Expect:* `kb_ingest(folder_path="mock_data/domain", collection_name="methanation_domain")`
returns a `job_id`; the agent polls `kb_job_status` to completion, then
`kb_list_collections` shows `methanation_domain` (2 docs).

### 4 — Retrieve domain grounding

> Using the knowledge base, what reactor conditions were used for the CO2 conversion measurement, and what license applies to this dataset?

*Expect:* `kb_search` over `methanation_domain` → **250 °C, 1 atm, H2:CO2 = 4:1,
GHSV 12,000**; license **CC-BY-4.0**.

### 5 — Generate the datacard for the finished dataset

> Use the generating-datacards skill to write a datacard for mock_data/dataset/catalyst_screening.csv. Pull the field definitions, measurement methodology, provenance, and license from the methanation_domain knowledge-base collection — don't invent them. Save it to audit/catalyst_screening_datacard.md. Then compare your sections against mock_data/expected_datacard.md and report anything missing.

*Expect:* the agent reads the installed skill's `SKILL.md`, queries the KB,
computes basic stats from the 8-row CSV, and writes
`audit/catalyst_screening_datacard.md` covering summary / provenance / schema /
methodology / stats / limitations / license.

### 6 — Validate the metadata

> Use the croissant-validator skill to check the Croissant/JSON-LD metadata for this dataset (generate it from the datacard if needed), and report any schema errors.

*Expect:* the validator skill runs and reports a clean pass or names specific
schema issues.

## Post-conditions

Confirm from a shell:

```bash
dsagt info genesis-skills                           # KB lists skills_catalog__genesis-genesis-skills + methanation_domain
ls ~/dsagt-projects/genesis-skills/skills/          # generating-datacards  croissant-validator
ls ~/dsagt-projects/genesis-skills/.claude/skills/  # both mirrored here at install time
cat ~/dsagt-projects/genesis-skills/skills/generating-datacards/PROVENANCE.txt
ls ~/dsagt-projects/genesis-skills/audit/           # catalyst_screening_datacard.md
```

1. The KB holds a `skills_catalog__genesis-genesis-skills` collection
   (searchable via `search_skills`) **and** a `methanation_domain` document
   collection (retrievable via `kb_search`).
2. `generating-datacards` and `croissant-validator` are installed into
   `<project>/skills/` and mirrored into `.claude/skills/`, each with a
   `PROVENANCE.txt` crediting the Genesis source. The next Claude session
   auto-invokes them natively; this session used them by reading their
   `SKILL.md`.
3. `audit/catalyst_screening_datacard.md` was produced for the finished dataset,
   grounded in the KB-ingested domain docs, covering the sections in
   `mock_data/expected_datacard.md`.
