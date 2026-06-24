# DSAgt Demo: Genesis Skills for a Data-Curation Pipeline

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

The three tiers in one sentence: **catalog = searchable but not in context;
installed = native and auto-invoked; KB = retrievable domain grounding.**

## Setup

Assumes DSAgt (`uv sync --all-groups`) and Claude Code
(`npm i -g @anthropic-ai/claude-code`) are installed, plus git **with network
access to `gitlab.osti.gov`** (the Genesis catalog clones from OSTI GitLab, not
GitHub). Embedding credentials are optional — `search_skills` / `kb_search` use
semantic search when `EMBEDDING_*` is set and fall back to a keyword scorer
otherwise (configure it for sharper relevance over the domain docs).

```bash
dsagt setup-kb                                   # bundled tools/skills + core KB
dsagt init genesis-skills --agent claude
cp -r use_cases/genesis_skills/mock_data ~/dsagt-projects/genesis-skills/mock_data
dsagt start genesis-skills                        # mirrors skill-creator into .claude/skills/ before launch
```

The Genesis catalog is **project-scoped** and not synced by `init`, so the agent
enables it in step 1 below (or run `dsagt skills add genesis-skills genesis` from
a shell first).

## Walkthrough

Paste each prompt into Claude Code (running inside the project), one at a time,
and check the expected behavior.

### 1 — Enable the Genesis source
> Enable the "genesis" skill source so we have the GENESIS / ModCon data-curation skills available. Then tell me how many skills it indexed.

*Expect:* `add_skill_source(source="genesis")` → a shallow clone of OSTI GitLab,
**74** skills indexed into `skills_catalog__genesis-genesis-skills`, source
written to `dsagt_config.yaml`. (Two upstream skills — including
`datacard-generator` — have technically-invalid YAML frontmatter; dsagt recovers
their name/description with a lenient fallback rather than dropping them, so they
*are* searchable.) Confirm from a shell:

```bash
dsagt skills list genesis-skills --catalog       # expect the skills_catalog__genesis-genesis-skills collection
```

### 2 — Find and install the datacard skill (catalog, NOT in context)
Search for the two deliverables **separately** — a single "datacard *and* Croissant"
query lets the validator outrank the generator. First the datacard generator:

> Search the catalog for a skill that creates a datacard / dataset documentation for a dataset, then install the best match into this project.

*Expect:* `search_skills` (catalog hits tagged `[catalog · install_skill to add]`,
**`generating-datacards`** ranked top) → `install_skill(skill_name="generating-datacards")`;
the reply notes it'll be native after the next start and that a `PROVENANCE.txt`
(Genesis source) was written.

### 3 — Find and install the Croissant validator
> Now search the catalog for a skill that validates Croissant / JSON-LD dataset metadata, and install the best match.

*Expect:* `search_skills` (**`croissant-validator`** ranked top) →
`install_skill(skill_name="croissant-validator")`. **Verify** both installs (each
lands with any `scripts/`/`references/`):

```bash
ls ~/dsagt-projects/genesis-skills/skills/
cat ~/dsagt-projects/genesis-skills/skills/generating-datacards/PROVENANCE.txt
```

### 4 — Ingest the domain docs into the KB
> Ingest the domain docs under mock_data/domain/ into a new knowledge-base collection called "methanation_domain". Poll until it finishes, then tell me what's in it.

*Expect:* `kb_ingest(folder_path="mock_data/domain", collection_name="methanation_domain")`
returns a `job_id`; the agent polls `kb_job_status` to completion, then
`kb_list_collections` shows `methanation_domain` (2 docs). **Verify:**

```bash
dsagt info genesis-skills                         # the new collection appears in the KB summary
```

### 5 — Retrieve domain grounding
> Using the knowledge base, what reactor conditions were used for the CO2 conversion measurement, and what license applies to this dataset?

*Expect:* `kb_search` over `methanation_domain` → **250 °C, 1 atm, H2:CO2 = 4:1,
GHSV 12,000**; license **CC-BY-4.0** (pulled from the protocol doc, not guessed).

### 6 — Generate the datacard for the finished dataset
> Use the generating-datacards skill to write a datacard for mock_data/dataset/catalyst_screening.csv. Pull the field definitions, measurement methodology, provenance, and license from the methanation_domain knowledge-base collection — don't invent them. Save it to audit/catalyst_screening_datacard.md. Then compare your sections against mock_data/expected_datacard.md and report anything missing.

*Expect:* the agent reads the installed skill's `SKILL.md`, queries the KB,
computes basic stats from the 8-row CSV, and writes
`audit/catalyst_screening_datacard.md` covering summary / provenance / schema /
methodology / stats / limitations / license.

### 7 — Validate the metadata
> Use the croissant-validator skill to check the Croissant/JSON-LD metadata for this dataset (generate it from the datacard if needed), and report any schema errors.

*Expect:* the validator skill runs and reports a clean pass or names specific
schema issues.

### 8 — Inspect the tiers (run in a shell)
```bash
dsagt skills list genesis-skills                 # installed: skill-creator + generating-datacards + croissant-validator
dsagt skills list genesis-skills --catalog       # catalog: skills_catalog__genesis-genesis-skills
dsagt info genesis-skills                         # KB shows methanation_domain
ls ~/dsagt-projects/genesis-skills/.claude/skills/
cat ~/dsagt-projects/genesis-skills/.claude/skills/.dsagt-managed.json
ls ~/dsagt-projects/genesis-skills/audit/        # catalyst_screening_datacard.md
```

Restart Claude (`dsagt start genesis-skills` again) to pick up the installed
Genesis skills as native auto-invoked skills.

## Post-Conditions

1. The KB holds a `skills_catalog__genesis-genesis-skills` collection
   (searchable via `search_skills`, absent from Claude's context) **and** a
   `methanation_domain` document collection (retrievable via `kb_search`).
2. `generating-datacards` and `croissant-validator` are installed into
   `<project>/skills/`, mirrored into `.claude/skills/`, each with a
   `PROVENANCE.txt` crediting the Genesis source.
3. `audit/catalyst_screening_datacard.md` was produced for the finished dataset,
   grounded in the KB-ingested domain docs, covering the sections in
   `mock_data/expected_datacard.md`.
4. The Croissant metadata validates (or its errors are reported).
5. `.claude/skills/.dsagt-managed.json` tracks exactly the dsagt-placed skills.

## Cleanup

```bash
dsagt stop genesis-skills
dsagt rm genesis-skills            # add -y to skip the prompt
```

The shared catalog cache lives at `~/dsagt-projects/.skill_sources/` and is
reused across projects; delete it to force a fresh clone.

## Notes

- The Genesis catalog is hosted on **OSTI GitLab** (`gitlab.osti.gov`), not
  GitHub — reached the same way as any other source (`add_skill_source` /
  `dsagt skills add … genesis`); only the host differs.
- `generating-datacards` is the frontmatter *name* of the skill whose directory
  is `datacard-generator` in the Genesis repo — `install_skill` accepts either.
  It and `croissant-validator` live under Genesis's `modcon-skills/` category
  (the BASE-Data team's own skills), so this demo is DSAgt consuming its sibling
  project's curated skills.
- `mock_data/` is intentionally tiny and illustrative. `expected_datacard.md` is
  a *shape* to check coverage against, not a byte-for-byte answer — the installed
  skill owns the authoritative template.
- Sister demo: [`isaac_skills_demo`](../isaac_skills_demo/) flexes the same
  catalog → install → native-mirror loop plus authoring a new skill with
  `skill-creator`, against the K-Dense `scientific` (GitHub) source.
