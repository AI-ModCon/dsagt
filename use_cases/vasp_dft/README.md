---
title: VASP DFT → AI-Ready Records
domain: Materials science — VASP DFT output to AI-ready records, via catalog skills and a registered code
summary: >-
  Convert VASP DFT output into AI-ready records — the agent discovers and
  installs a pymatgen skill from a catalog, authors a converter skill for a
  slab calculation, extends it to nudged-elastic-band calculations, registers
  that converter as a code, and runs it with provenance against a reference
  record (no DFT run, no HPC).
status: published
order: 30
---

# DSAgt Demo: VASP DFT → AI-Ready Records

> **Estimated time:** ~25 minutes

**Goal:** turn VASP calculations into AI-ready records in the
[ISAAC record schema](https://github.com/ISAAC-DOE/isaac-ai-ready-record)
through both of DSAgt's extension mechanisms:

1. **Skills.** The agent discovers the external skill sources, syncs the K-Dense
   catalog, installs its `pymatgen` skill, and uses the `skill-creator` base skill
   to author a `vasp-to-isaac` skill whose converter parses VASP output with
   `pymatgen.io.vasp`. It runs that skill on a small slab calculation.
2. **Codes.** The agent extends its skill with a converter for nudged-elastic-band
   (NEB) calculations, registers that converter as a code, and runs it through
   `dsagt-run` on a five-image NEB fixture, so the execution is captured in
   `trace_archive/` and can be reconstructed. A reference record is the oracle.

Both parts use real `pymatgen.io.vasp` parsing. The slab data is a mock: valid
VASP format, with an OUTCAR whose header comes from a real run and whose body
keeps the first and last ionic steps. The NEB data is
a fixture from the pymatgen test suite. Reference outputs for both
(`expected_isaac_record.json` for the slab, `isaac_neb_record.json` for the NEB)
come with the data, so the agent's records can be checked.

Folder contents:

| Path | Role in the demo |
|------|------------------|
| [`reference/vasp_neb_to_isaac.py`](reference/vasp_neb_to_isaac.py) | a converter that produces the NEB reference record — a reference solution, not an input |
| [`reference/isaac_neb_record.json`](reference/isaac_neb_record.json) | the NEB reference record (also in the data bundle) |
| [`reference/skills/vasp-to-isaac/`](reference/skills/vasp-to-isaac/) | a broader slab/bulk converter skill for `vasprun.xml`-bearing data; what the agent-authored skill can grow into |

## Prerequisites

- DSAgt installed with the `vasp-dft` extra
  (`pip install "dsagt[vasp-dft] @ git+https://github.com/AI-ModCon/dsagt.git"`),
  which brings `pymatgen`; both converters use `pymatgen.io.vasp`.
- An agent platform installed and **already authenticated**.
- Git, for the catalog clone.

## Setup

```bash
dsagt init
```

At the menu, name the project `isaac-vasp`, pick your agent, and keep the defaults:
`genesis` is the default skill source, and the walkthrough has the agent discover, sync,
and search a second one from inside the session. Then:

```bash
PROJ=~/dsagt-projects/isaac-vasp
# The fixtures (the NEB images, the mock slab, and their reference records) are
# in the repository, under this use case's data/ folder.
cp -r use_cases/vasp_dft/data "$PROJ/data"
# $PROJ/data now holds neb/, isaac_neb_record.json, mock_slab/, expected_isaac_record.json
dsagt start isaac-vasp                        # mirrors the skill-creator base skill into the agent's native skills dir
```

## Execution

Paste each prompt into the agent, one at a time. The arc: **see what you have →
find more → sync a source → install the relevant skill → author a new one → run
it → then register a converter as a code and run it with provenance.**

### 1. Native skill discovery

```text
Do you have a skill available for scaffolding new skills? Name it and give me a one-line summary of what it does.
```

**Expect:** the agent names **`skill-creator`** and summarizes it — discovered
natively, with no MCP call. `dsagt init` installed the skill from the genesis
catalog and `dsagt start` mirrored it into the
agent's native skills directory, so the agent sees its name and description like
any native skill and loads the full `SKILL.md` only when it is invoked.
`search_skills` is for the not-yet-installed catalog only, so it should not fire here.

### 2. List the skill sources

```text
Where can I get more skills from? List the skill sources you can pull from and which are already synced.
```

**Expect:** `list_skill_sources` → the known sources (`k-dense-ai`, `anthropic`,
`antigravity`, `composio`, `genesis`) with URLs; `genesis` is synced (the default
source) and the rest are available but not synced. Synced means indexed into this
project's knowledge base, so a source another project has cloned still shows as
not synced here, and step 3 indexes it from the shared clone under
`~/dsagt-projects/.skill_sources/`.

### 3. Sync a source

```text
Sync the "k-dense-ai" source so we can search its catalog.
```

**Expect:** `add_skill_source(source="k-dense-ai")` → a shallow clone of K-Dense
`scientific-agent-skills`, its skills indexed into
`skills_catalog__k-dense-ai-scientific-agent-skills`, source persisted to
`.dsagt/config.yaml`. The catalog is searchable immediately — no restart.

### 4. Install the relevant skill

```text
Search the catalog for a skill that helps parse VASP output with pymatgen, then install the most relevant one into this project.
```

**Expect:** `search_skills` (catalog hits tagged `[catalog · install_skill to add]`,
`pymatgen` at or near the top) → `install_skill(skill_name="pymatgen")`. The installed
skill carries the reference docs (`pymatgen.io.vasp.Incar` / `Poscar` / `Outcar`)
the converter uses next. **Verify** it landed:

```bash
ls "$PROJ/skills/"
```

### 5. Author the converter skill with skill-creator

```text
Use the skill-creator skill to author a new project skill named "vasp-to-isaac". Following the pymatgen skill you just installed, its converter should use `pymatgen.io.vasp` — `Incar.from_file` (ENCUT, NSW, ISPIN, LDAUU), `Poscar.from_file` (formula, atom counts), and `Outcar` (final energy, energy(sigma->0), total magnetization, max force) — to read a VASP slab calc directory and emit an ISAAC-style JSON record. The mock has no vasprun.xml, so take energy/forces from the OUTCAR. Target the shape in data/expected_isaac_record.json. Save it with save_skill.
```

**Expect:** the agent reads `skill-creator`'s template and the `pymatgen` skill's IO
docs, then `save_skill` writes `<project>/skills/vasp-to-isaac/` whose script
imports `pymatgen.io.vasp` (not a hand-rolled regex parser).

### 6. Run the skill on the slab calculation

```text
Invoke the vasp-to-isaac skill on data/mock_slab/ and write the result to audit/mock_slab_isaac.json. Then diff its structure and values against data/expected_isaac_record.json and report any differences.
```

**Expect:** pymatgen parses the mock directory and the agent writes
`audit/mock_slab_isaac.json` with the key fields pymatgen extracted — final
energy ≈ -132.8421 eV (`Outcar.final_energy`), 12 atoms (`Poscar`), ENCUT 520 /
NSW 50 (`Incar`), total mag ≈ 8.0123 (`Outcar.total_mag`) — matching the reference.
The reference's `ionic_steps` is NSW from the INCAR: the mock OUTCAR keeps only the
first and last of the 50 steps.

### 7. Extend the skill to NEB calculations and register the converter

```text
Extend the vasp-to-isaac skill with a second converter,
scripts/vasp_neb_to_isaac.py, for nudged-elastic-band calculations. It takes a
positional NEB directory containing 00/, 01/, ... image subdirectories and an
--output path, parses each image's OUTCAR with pymatgen.io.vasp.Outcar, and
writes an ISAAC v1.05 record whose computation block records the NEB method,
the number of intermediate images, and the reaction, and whose measurement
block carries the energy series along the path. Target the shape of
data/isaac_neb_record.json. Update SKILL.md to describe both converters. Then
register the new script as a code named vasp-neb-to-isaac with the positional
neb_dir and the --output option; run it with --help first.
```

**Verify:** `Search the registry for the vasp-neb-to-isaac code.` →
`$PROJ/codes/vasp-neb-to-isaac/SKILL.md` should exist.

### 8. Run the conversion and check it

```text
Run the registered vasp-neb-to-isaac code, with its exact command, to convert data/neb/
and write the record to data/neb_record.json.
Compare it against data/isaac_neb_record.json: report differences in structure
and in the computation and measurement blocks, fix the converter, and rerun the
registered code until they agree on the method, image count, reaction, and
energy series. The reaction is Fe(lattice) → Fe(vacancy site); the OUTCARs do
not state it, so pass it to the converter.
```

**Expect:** `dsagt-run --code vasp-neb-to-isaac -- ...` runs land in
`trace_archive/`; pymatgen parses the five OUTCARs (endpoints plus three
intermediate images); the final record's `computation.transition_state` has
`method: NEB`, `images: 3`, and the Fe vacancy-migration reaction, and its
`measurement.series` carries the five-point energy path matching the reference:
endpoints at −255.980 eV and −255.981 eV, a barrier of 0.325 eV at image 2.

One pitfall to watch for: reading `Outcar.final_energy_wo_entrp` instead of
`Outcar.final_energy` (the energy(sigma→0) of the last ionic step) shifts every
energy by about 0.33 eV and the barrier to 0.309 eV. An agent may then report
structural agreement and attribute the offset to "different calculations". The
reference values are the last `energy(sigma->0)` line of each image's OUTCAR;
hold it to them.

### 9. Reconstruct the pipeline

```text
Reconstruct the pipeline from the execution records as a bash script and save it as
pipeline.sh.
```

**Expect:** the script holds every recorded run of the NEB converter, including
the attempts that failed the comparison, in the order they ran.

### 10. Review the project artifacts

```text
Show me the contents of my project folder in a tree format, with the artifacts dsagt recorded during this session highlighted.
```

**Expect:** a listing of the project directory that marks the execution records in
`trace_archive/`, the reports in `audit/`, the registered codes under `codes/`, the
installed skills under `skills/`, the trace store `mlflow.db`, and the session's outputs,
with a line on what each is.

## Post-Conditions

Confirm from a shell (the native skills directory is `.claude/skills/` for Claude Code,
`.agents/skills/` for Codex, Goose, and opencode, `.cline/skills/` for Cline):

```bash
dsagt info isaac-vasp                     # KB shows the k-dense-ai catalog collection
ls "$PROJ/skills/"                        # aidrin  datacard-generator  pymatgen  skill-creator  vasp-to-isaac
ls "$PROJ/codes/"                         # vasp-neb-to-isaac
ls "$PROJ/audit/" "$PROJ/trace_archive/"
```

1. The KB holds the `skills_catalog__k-dense-ai-scientific-agent-skills`
   collection, synced in-session by the agent (step 3), searchable via
   `search_skills` but absent from the agent's context.
2. The `pymatgen` catalog skill is installed into `<project>/skills/` and
   mirrored into the agent's native skills directory.
3. A `vasp-to-isaac` skill, authored via `skill-creator` and parsing with
   `pymatgen.io.vasp`, exists and is natively discoverable.
4. `audit/mock_slab_isaac.json` was produced from the mock slab directory and
   matches the ISAAC shape and values.
5. The `vasp-to-isaac` skill has a second script, `vasp_neb_to_isaac.py`, and
   the code registry contains the `vasp-neb-to-isaac` spec.
6. `data/neb_record.json` has the structure of the reference `data/isaac_neb_record.json`
   and agrees with it on the method, the image count, the reaction, and the energy series
   (other fields, such as notes and the method's sub-fields, may differ); `trace_archive/`
   holds every NEB conversion attempt, including any that failed the comparison.
7. `pipeline.sh` replays every recorded NEB conversion.
8. MLflow traces (in the serverless `mlflow.db` store) capture the session —
   `dsagt traces isaac-vasp`.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Native discovery of the `skill-creator` base skill | 1 |
| Skill-source listing and in-session sync (`list_skill_sources`, `add_skill_source`) | 2, 3 |
| Catalog search and install (`search_skills`, `install_skill`) | 4 |
| Skill authoring with `skill-creator` and `save_skill` | 5 |
| Installed-skill execution | 6 |
| Agent-written converter from a reference record, added to its own skill | 7 |
| Code registration (`save_code_spec`) and registry search | 7 |
| Code execution with provenance through `dsagt-run`, iterated against a reference | 8 |
| Pipeline reconstruction | 9 |
| Review of the session's artifacts | 10 |

## Cleanup

```bash
dsagt rm isaac-vasp -y
```

The shared catalog cache is stored at `~/dsagt-projects/.skill_sources/` and is
reused across projects; delete it to force a fresh clone.

## Notes

- `mock_slab/` is not real DFT output, but it is valid VASP format: the
  INCAR/POSCAR parse cleanly, and the OUTCAR's header (dimensions, plane-wave
  table) is copied from a real VASP 5 run so pymatgen's `Outcar` parses it,
  while its body keeps only the first and last ionic steps (TOTEN,
  `energy(sigma->0)`, magnetization, the force block). There is no
  `vasprun.xml`, so the converter takes energy/forces from the OUTCAR.
- The `neb/` OUTCARs are public pymatgen test fixtures.
  [`reference/vasp_neb_to_isaac.py`](reference/vasp_neb_to_isaac.py) is a
  converter that produces the reference record; compare the agent's converter
  to it after step 8, not before.
- With the default local embedder (`bge-small`), absolute `search_skills` scores
  are low because short queries under-score long SKILL.md text — the ranking is
  still correct (`pymatgen` first). Set `embedding.backend: api` for sharper
  relevance. With no embedder at all, `search_skills` falls back to keyword
  scoring; `install_skill` and the native mirror are filesystem operations.
- [`reference/skills/vasp-to-isaac/`](reference/skills/vasp-to-isaac/) is a
  broader slab/bulk converter skill that needs `vasprun.xml`-bearing slab or
  bulk data. It is a reference for what the agent-authored skill can grow into,
  not something this demo's data exercises.
- Sister demo: [`genesis_skills`](../genesis_skills/) exercises the same catalog →
  install → native loop plus KB domain ingest and datacard generation, against
  the Genesis (OSTI GitLab) source.
