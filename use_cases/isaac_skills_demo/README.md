---
title: Skill-Driven VASP → ISAAC Conversion
domain: Skill management — external skill catalog (K-Dense) authoring a VASP → ISAAC converter
summary: >-
  A lightweight mock of the isaac_vasp workflow where the agent itself
  discovers, syncs, installs, and authors skills (pymatgen, skill-creator) to
  convert mock VASP output into an ISAAC record, vetting the skill-management
  feature end-to-end.
status: published
order: 70
guides:
  - text: Skill-Driven VASP → ISAAC Walkthrough
    path: README.md
---

# DSAgt Demo: Skill-Driven VASP → ISAAC Conversion

> **Estimated time:** ~15 minutes — the agent flow runs in seconds on the
> bundled few-KB mock data; the one real cost is a one-time
> `pip install pymatgen` (needed for step 6) plus a shallow clone of the
> K-Dense catalog from GitHub.

A lightweight mock of the [`isaac_vasp`](../isaac_vasp/) workflow, built to **vet
the skill-management feature**. It follows the same arc as `isaac_vasp` — install
the **pymatgen** skill, author a `vasp-to-isaac` converter that parses VASP output
**with pymatgen**, and emit an ISAAC record — but the agent **discovers, syncs,
installs, and authors** those skills itself, surfaced through Claude Code's
*native* skill discovery. It uses tiny **mock VASP outputs** (`mock_data/`, a few
KB), so the whole thing runs in seconds with no DFT, no NERSC, and no 32 MB
OUTCAR files — yet the parsing is the real `pymatgen.io.vasp`, not a hand-rolled
stand-in.

## What this demonstrates

- **`list_skill_sources`** — the agent discovers what external sources it can
  pull from (curated names + arbitrary git URLs) and which are synced.
- **`add_skill_source`** — the agent syncs a source (here: K-Dense
  `k-dense-ai`, 140+ skills) into a searchable catalog that is
  **not** loaded into context; with the single `dsagt-server` it's searchable
  immediately, no restart.
- **`search_skills`** + **`install_skill`** — find a catalog skill (hits marked
  `[catalog]`) and draw it into the project + Claude's native `.claude/skills/`.
- **`skill-creator`** — the bundled meta-skill scaffolds a new `vasp-to-isaac`
  skill (from the Anthropic template) whose converter *uses the installed
  pymatgen skill* to parse the VASP output — install-then-build, not install-and-ignore.
- **Native mirror** — installed + bundled skills appear under
  `.claude/skills/<name>/` (tracked by `.dsagt-managed.json`), so Claude
  auto-invokes them with no MCP round-trip.

The two tiers in one sentence: **catalog = searchable but not in context;
installed = native and auto-invoked.**

## Setup

Assumes DSAgt (`uv sync --all-groups`) and Claude Code
(`npm i -g @anthropic-ai/claude-code`) are installed, plus git. Embedding
credentials are optional — `search_skills` uses semantic search when
`EMBEDDING_*` is set and falls back to a keyword scorer otherwise (configure it
for sharper relevance).

```bash
dsagt init isaac-skills-demo --agent claude --exclude genesis   # core KB + bundled codes/skills, but NO catalog synced
cp -r use_cases/isaac_skills_demo/mock_data ~/dsagt-projects/isaac-skills-demo/mock_data
dsagt start isaac-skills-demo                     # mirrors the bundled skill-creator into .claude/skills/ before launch
```

The project starts with **no external catalog synced** — that's deliberate (the
`--exclude genesis` drops the default catalog): the walkthrough has the agent
*discover, sync, and search* it from inside the session. The single
`dsagt-server` owns the KB, so a source the agent syncs mid-session is
**immediately searchable, no restart**.

> To instead pre-sync a source at init, pass `--include` with a source name
> (e.g. `dsagt init … --include k-dense-ai`); `init` provisions it into the KB
> and step 3 below becomes an idempotent refresh.

## Walkthrough

Paste each prompt into Claude Code (running inside the project), one at a time.
The arc: **see what you have → find more → sync a source → install the relevant
skill → author a new one → run it.**

### 1 — What do we have? (native discovery)
> Do you have a skill available for scaffolding new skills? Name it and give me a one-line summary of what it does.

*Expect:* Claude names **`skill-creator`** and summarizes it — discovered
**natively, with no MCP call** and no file digging. That's the mirror working:
`dsagt start` copied the bundled `skill-creator` into `.claude/skills/`, so Claude
sees its name + description like any native skill (and loads the full `SKILL.md`
only when the skill is invoked — progressive disclosure). `search_skills` is for
the not-yet-installed *catalog* only, so it should not fire here. You confirm
*which* skills dsagt placed from a shell in step 7 (`cat .dsagt-managed.json`) —
that manifest is dsagt's internal mirror bookkeeping, not something the agent
reads.

### 2 — Where can we find more skills?
> Where can I get more skills from? List the skill sources you can pull from and which are already synced.

*Expect:* `list_skill_sources` → the known sources (`k-dense-ai`, `anthropic`,
`antigravity`, `composio`, `genesis`) with URLs, each flagged **available, not
synced** (nothing is synced yet on an `--exclude genesis` setup).

### 3 — Sync skills from an external repo
> Sync the "k-dense-ai" source so we can search its catalog.

*Expect:* `add_skill_source(source="k-dense-ai")` → a shallow clone of K-Dense
`scientific-agent-skills`, ~140 skills indexed into
`skills_catalog__k-dense-ai-scientific-agent-skills`, source persisted to
`.dsagt/config.yaml`. Because it's one `dsagt-server`, the catalog is searchable
**immediately** — the next prompt can hit it with no restart.

### 4 — Add the relevant skill
> Search the catalog for a skill that helps parse VASP output with pymatgen, then install the most relevant one into this project.

*Expect:* `search_skills` (catalog hits tagged `[catalog · install_skill to add]`,
`pymatgen` at/near the top) → `install_skill(skill_name="pymatgen")`; the reply
notes it'll be native after the next start. The installed `pymatgen` skill carries
the reference docs (`pymatgen.io.vasp.Incar` / `Poscar` / `Outcar`) the converter
uses next. **Verify** the skill dir (with any `scripts/`/`references/`) landed:

```bash
ls ~/dsagt-projects/isaac-skills-demo/skills/
```

### 5 — Create the converter skill with skill-creator
This mirrors `isaac_vasp`'s `vasp-to-isaac` skill — the lightweight version reads
the small mock directory, but does the parsing with **real pymatgen**, the same
way the full workflow does (just without the heavy `vasprun.xml`).

> Use the skill-creator skill to author a new project skill named "vasp-to-isaac". Following the pymatgen skill you just installed, its converter should use `pymatgen.io.vasp` — `Incar.from_file` (ENCUT, NSW, ISPIN, LDAUU), `Poscar.from_file` (formula, atom counts), and `Outcar` (final energy, energy(sigma->0), total magnetization, max force) — to read a VASP slab calc directory and emit an ISAAC-style JSON record. The mock has no vasprun.xml, so take energy/forces from the OUTCAR. Target the shape in mock_data/expected_isaac_record.json. Save it with save_skill.

*Expect:* the agent reads `skill-creator`'s template + the `pymatgen` skill's IO
docs, then `save_skill` writes `<project>/skills/vasp-to-isaac/` whose script
imports `pymatgen.io.vasp` (not a hand-rolled regex parser).

### 6 — Run the converter on the mock data
> Invoke the vasp-to-isaac skill on mock_data/mock_slab/ and write the result to audit/mock_slab_isaac.json. Then diff its structure and values against mock_data/expected_isaac_record.json and report any differences.

*Expect:* pymatgen parses the mock dir and the agent writes
`audit/mock_slab_isaac.json` with the key fields **pymatgen extracted** — final
energy ≈ -132.8421 eV (`Outcar.final_energy`), 12 atoms (`Poscar`), ENCUT 520 /
NSW 50 (`Incar`), total mag ≈ 8.0123 (`Outcar.total_mag`) — matching the
reference. (`pymatgen` must be importable in the project env — see Notes.)

### 7 — Inspect the tiers (run in a shell)
```bash
dsagt info isaac-skills-demo                      # KB shows the k-dense-ai catalog collection
ls ~/dsagt-projects/isaac-skills-demo/skills/     # installed: pymatgen + vasp-to-isaac
ls ~/dsagt-projects/isaac-skills-demo/.claude/skills/
cat ~/dsagt-projects/isaac-skills-demo/.claude/skills/.dsagt-managed.json
```

The manifest lists only the skills **dsagt** placed; any skill you hand-create
under `.claude/skills/` is never touched. Restart Claude
(`dsagt start isaac-skills-demo` again) to pick up newly-mirrored skills as
native auto-invoked skills.

## Post-Conditions

1. The KB holds the `skills_catalog__k-dense-ai-scientific-agent-skills`
   collection, **synced in-session by the agent** (step 3), searchable via
   `search_skills` but absent from Claude's context.
2. The `pymatgen` catalog skill was installed into `<project>/skills/` and
   mirrored into `.claude/skills/`.
3. A new `vasp-to-isaac` skill, authored via `skill-creator` and parsing with
   **pymatgen** (`pymatgen.io.vasp`), exists and is native-discoverable.
4. `audit/mock_slab_isaac.json` was produced from the mock VASP directory by
   pymatgen and matches the ISAAC shape + values.
5. `.claude/skills/.dsagt-managed.json` tracks exactly the dsagt-placed skills.

## Cleanup

```bash
dsagt rm isaac-skills-demo            # add -y to skip the prompt
```

The shared catalog cache lives at `~/dsagt-projects/.skill_sources/` and is
reused across projects; delete it to force a fresh clone.

## Notes

- **pymatgen must be importable** in the project env to run step 6 (the converter
  uses `pymatgen.io.vasp`, exactly as `isaac_vasp` does). Install it once —
  `pip install pymatgen` (or `uv pip install pymatgen`) into the same environment
  `dsagt` runs in; the installed `pymatgen` skill's `references/` document this.
  This is the demo's one real dependency — "lightweight" is about the data, not
  avoiding pymatgen.
- `mock_data/` is intentionally tiny and **not** real DFT output, but it is *valid
  VASP format*: the INCAR/POSCAR parse cleanly, and the OUTCAR stub keeps exactly
  the lines pymatgen's `Outcar` reads (TOTEN, `energy(sigma->0)`, magnetization,
  the force block) while omitting the ~250k-line SCF/eigenvalue blocks. No
  `vasprun.xml` (it'd be large), so the converter takes energy/forces from OUTCAR.
- With the default **local** embedder (`bge-small`), absolute search scores are
  low (~0.03) because short queries under-score long SKILL.md text — *ranking* is
  still correct (pymatgen #1). Switch `embedding.backend` to `api` for sharper
  relevance. With no embedder at all, `search_skills` falls back to keyword
  scoring; `install_skill` and the native mirror are pure filesystem ops.
- Add **more** sources the same way — ask the agent to "enable the anthropic
  source" (or `antigravity`, `composio`, `genesis`, or any
  `https://github.com/owner/repo`), which fires `add_skill_source`. Each lands
  in its own `skills_catalog__*` collection.
- Sister demo: [`genesis_skills`](../genesis_skills/) flexes the same catalog →
  install → native loop plus KB domain ingest and datacard generation, against
  the Genesis (OSTI GitLab) source.
