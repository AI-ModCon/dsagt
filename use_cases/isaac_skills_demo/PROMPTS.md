# Hand-pass prompt script — isaac_skills_demo

The deterministic backbone (init, catalog sync, CLI `skills` commands, the
native mirror) was already vetted by a first pass — see "First-pass results"
at the bottom. This script is for the **interactive agent pass**: paste each
prompt into Claude Code (running inside the project) and check the expected
behavior.

## Before you start

The project `isaac-skills-demo` is already set up from the first pass:
- catalog synced (146 K-Dense + 17 Anthropic skills),
- `pymatgen` already installed and mirrored into `.claude/skills/`,
- `mock_data/` copied into the project.

To start fresh instead, delete and rebuild — **including the catalog sync**
(a fresh `init` only copies the *shared* KB; the external catalog is
project-scoped, so it must be synced after init or the catalog will be empty
and prompt 2 finds nothing):

```bash
dsagt rm isaac-skills-demo -y
dsagt init isaac-skills-demo --agent claude
cp -r use_cases/isaac_skills_demo/mock_data ~/dsagt-projects/isaac-skills-demo/mock_data
dsagt skills sync isaac-skills-demo      # REQUIRED: populate the catalog (146 skills)
```

> Alternatively, run the global `dsagt setup-kb` once (it syncs the default
> catalog into the *shared* KB, which every new `init` then copies in). The
> per-project `dsagt skills sync` above is the lighter, self-contained path.

Confirm the catalog is present before launching:

```bash
dsagt skills list isaac-skills-demo --catalog   # expect a skills_catalog__* collection
```

Launch the agent:

```bash
dsagt start isaac-skills-demo
```

---

## Prompts (paste one at a time)

### 1 — Confirm native discovery of the bundled meta-skill
> What skills do you have available right now? List them and say which are dsagt-managed.

*Expect:* `skill-creator`, `datacard-generator`, and (if you didn't rebuild) `pymatgen` are visible as native skills.

### 2 — Search the catalog (NOT in context)
> Search the skill catalog for a skill that helps work with VASP, pymatgen, or DFT materials data. List what you find and which are installable from the catalog.

*Expect:* `search_skills` is called; catalog hits are tagged `[catalog · install_skill to add]`; `pymatgen` ranks at/near the top.

### 3 — Install from the catalog
> Install the most relevant materials/DFT skill you found from the catalog into this project.

*Expect:* `install_skill(skill_name="pymatgen")`; reply notes it'll be native after the next start. (Already installed if you didn't rebuild — it should say "updated".)

### 4 — Add a second catalog source via MCP
> Enable the "anthropic" skill source so we also have the official Anthropic skills available, then tell me how many skills that added to the catalog.

*Expect:* `add_skill_source(source="anthropic")` → ~17 skills indexed; the source is written into `dsagt_config.yaml`.

### 5 — List configured/synced sources
> List the skill sources currently configured and synced.

*Expect:* `list_skill_sources` → `scientific` + `anthropic` known/synced.

### 6 — Author a project skill with skill-creator
> Use the skill-creator skill to author a new project skill named "vasp-to-isaac-mock". It should: read a mock VASP calculation directory (POSCAR + INCAR + OUTCAR) under mock_data/mock_slab/, extract the final energy, atom count, and whether it's a slab relaxation (NSW > 0), and emit a small ISAAC-style JSON record. Use mock_data/expected_isaac_record.json as the shape to target. Save it with save_skill.

*Expect:* the agent reads skill-creator's template + spec, then `save_skill` writes `<project>/skills/vasp-to-isaac-mock/`.

### 7 — Run the new skill on the mock data
> Invoke the vasp-to-isaac-mock workflow on mock_data/mock_slab/ and write the result to audit/mock_slab_isaac.json. Then diff its structure against mock_data/expected_isaac_record.json and report any missing fields.

*Expect:* a produced `audit/mock_slab_isaac.json` with the key fields (final energy ≈ -132.8421 eV, 12 atoms, slab/NSW=50). Compare to the reference.

### 8 — Inspect both tiers (run in a shell, not the agent)
```bash
dsagt skills list isaac-skills-demo
dsagt skills list isaac-skills-demo --catalog
ls ~/dsagt-projects/isaac-skills-demo/.claude/skills/
cat ~/dsagt-projects/isaac-skills-demo/.claude/skills/.dsagt-managed.json
```

*Expect:* installed list includes `pymatgen` + `vasp-to-isaac-mock`; catalog lists both `skills_catalog__*` collections; the manifest tracks exactly the dsagt-placed skills (your hand-authored `.claude/skills/` entries, if any, are untouched).

---

## First-pass results (already verified, no agent needed)

| Step | Result |
|---|---|
| `dsagt init` | ✅ `.claude/skills/` mirror fired at init: `skill-creator` (+ refs) + `datacard-generator`; manifest correct; config `skills:` block present |
| `dsagt skills sync` | ✅ real clone of K-Dense, **146 skills** indexed into `skills_catalog__k-dense-ai-scientific-agent-skills` (~19s, local embeddings) |
| `dsagt skills list --catalog` | ✅ shows the catalog collection |
| `dsagt skills search "VASP pymatgen DFT materials"` | ✅ `pymatgen` top catalog hit; tiers tagged `[bundled]` / `[catalog:…]` |
| `dsagt skills add … pymatgen` | ✅ installed into `skills/pymatgen/` **with** `scripts/` + `references/` |
| start-equivalent re-mirror | ✅ `pymatgen` now in `.claude/skills/` + manifest |
| `dsagt skills add … anthropic` | ✅ second source cloned + **17 skills** indexed; source persisted to config |

**Caveat to know:** with the default **local** embedding backend (`bge-small`), absolute search scores are low (~0.03) because short queries under-score long SKILL.md texts — *ranking* is still correct (pymatgen #1). An API embedding model scores higher. Set `EMBEDDING_*` / switch `embedding.backend` to `api` for sharper relevance.

**Fix applied during the first pass:** the CLI `dsagt skills add <proj> <source>` path now also persists the source to `dsagt_config.yaml` (previously only the MCP `add_skill_source` tool did), so a later config-driven `dsagt skills sync` re-syncs it. Regression test added.
