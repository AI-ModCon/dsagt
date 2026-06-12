# DSAgt Demo: Skill-Driven VASP → ISAAC Conversion

A lightweight mock of the [`isaac_vasp`](../isaac_vasp/) workflow, built specifically to **vet the skill-management feature**. Instead of shipping a hand-written converter skill, this walkthrough has the agent **discover, install, and author skills** — and surfaces them through Claude Code's *native* skill discovery.

It uses tiny **mock VASP outputs** (`mock_data/`, a few KB) so the whole thing runs in seconds with no DFT, no NERSC, and no 32 MB OUTCAR files.

## What this demonstrates (the new functionality)

- **External skill catalog** — pull Agent-Skills from GitHub repos (default: K-Dense `scientific-agent-skills`, 140+ skills) into a searchable catalog that is **not** loaded into the agent's context.
- **`search_skills`** spanning installed skills *and* the catalog (catalog hits marked `[catalog]`).
- **`install_skill`** — move a chosen catalog skill into the project, then into Claude's native `.claude/skills/`.
- **`add_skill_source`** — the agent enables another source (e.g. `anthropic`) via an MCP tool call.
- **`skill-creator`** — the bundled meta-skill scaffolds a brand-new `vasp-to-isaac-mock` skill from the Anthropic template.
- **Native mirror** — installed + bundled skills appear under `.claude/skills/<name>/` (tracked by `.dsagt-managed.json`), so Claude auto-invokes them with no MCP round-trip.

The two tiers in one sentence: **catalog = searchable but not in context; installed = native and auto-invoked.**

## Prerequisites

- DSAgt installed (`uv sync --all-groups`)
- Claude Code installed (`npm i -g @anthropic-ai/claude-code`)
- Valid embedding credentials (so `search_skills` works) — `EMBEDDING_*` in your shell or project config
- Git installed (catalog sync uses a shallow clone)

## Setup

### 1. Build the core KB + default skill catalog

```bash
dsagt setup-kb
```

This indexes the bundled tools/skills **and** clones + indexes the default skill source (K-Dense scientific). To skip the catalog, pass `--no-skill-catalog`. To go faster, you can defer the catalog and add it per-project later (Step A).

### 2. Initialize a project

```bash
dsagt init isaac-skills-demo --agent claude
```

The generated `dsagt_config.yaml` already carries a `skills:` block with the `scientific` source enabled and `populate_native: true`.

### 3. Start the session

```bash
dsagt start isaac-skills-demo
```

`dsagt start` mirrors installed + bundled skills into `.claude/skills/` **before** launching Claude, so the bundled `skill-creator` is already discoverable. Copy the mock data into the project first so the agent can reach it:

```bash
cp -r use_cases/isaac_skills_demo/mock_data ~/dsagt-projects/isaac-skills-demo/mock_data
```

## Execution

### A. Browse and install a skill from the catalog

First confirm the catalog is searchable (these are NOT in Claude's context — they live in the KB):

```bash
dsagt skills list isaac-skills-demo --catalog
```

You should see a `skills_catalog__k-dense-ai-scientific-agent-skills` collection. Now have the agent search it:

```text
Search the skill catalog for a skill that helps work with VASP, pymatgen, or DFT materials data. List what you find and which are installable from the catalog.
```

The agent calls `search_skills(...)`; catalog hits are marked `[catalog · install_skill to add]`. Install one:

```text
Install the most relevant materials/DFT skill you found from the catalog into this project.
```

The agent calls `install_skill(skill_name=...)`.

**Verify:**

```bash
ls ~/dsagt-projects/isaac-skills-demo/skills/
```

The installed skill directory (with any `scripts/` and `references/`) is now under the project's `skills/`.

### B. Add a second catalog source via the agent

```text
Enable the "anthropic" skill source so we also have the official Anthropic skills available, then tell me how many skills that added to the catalog.
```

The agent calls `add_skill_source(source="anthropic")` — it clones + indexes that repo into its own catalog collection and persists the source to `dsagt_config.yaml`. Confirm:

```text
List the skill sources currently configured and synced.
```

(`list_skill_sources`.)

### C. Author the project-specific skill with `skill-creator`

The real `isaac_vasp` ships a hand-written `vasp-to-isaac` converter. Here we let the agent build a mock one using the bundled meta-skill:

```text
Use the skill-creator skill to author a new project skill named "vasp-to-isaac-mock". It should: read a mock VASP calculation directory (POSCAR + INCAR + OUTCAR) under mock_data/mock_slab/, extract the final energy, atom count, and whether it's a slab relaxation (NSW > 0), and emit a small ISAAC-style JSON record. Use mock_data/expected_isaac_record.json as the shape to target. Save it with save_skill.
```

The agent reads `skill-creator`'s template (`references/SKILL_template.md`) and spec, then writes `<project>/skills/vasp-to-isaac-mock/`.

**Verify:**

```bash
cat ~/dsagt-projects/isaac-skills-demo/skills/vasp-to-isaac-mock/SKILL.md
```

### D. Run the new skill on the mock data

```text
Invoke the vasp-to-isaac-mock workflow on mock_data/mock_slab/ and write the result to audit/mock_slab_isaac.json. Then diff its structure against mock_data/expected_isaac_record.json and report any missing fields.
```

### E. Inspect both tiers

```bash
# Installed (native-discoverable) skills — bundled + project + anything installed
dsagt skills list isaac-skills-demo

# The native mirror Claude actually reads, plus the dsagt-managed manifest
ls ~/dsagt-projects/isaac-skills-demo/.claude/skills/
cat ~/dsagt-projects/isaac-skills-demo/.claude/skills/.dsagt-managed.json
```

The manifest lists only the skills **dsagt** placed (`skill-creator`, the installed catalog skill, `vasp-to-isaac-mock`). Any skill you hand-create under `.claude/skills/` is never touched.

To pick up newly-mirrored skills as native `/commands`, restart Claude (`dsagt start isaac-skills-demo` again, then relaunch).

## Post-Conditions

1. The KB holds per-source catalog collections (`skills_catalog__*`) for `scientific` (+ `anthropic` after Step B), searchable via `search_skills` but absent from Claude's context.
2. A catalog skill was installed into `<project>/skills/` and mirrored into `.claude/skills/`.
3. A new `vasp-to-isaac-mock` skill, authored via `skill-creator`, exists and is native-discoverable.
4. `audit/mock_slab_isaac.json` was produced from the mock VASP directory and matches the ISAAC shape.
5. `.claude/skills/.dsagt-managed.json` tracks exactly the dsagt-placed skills.

## Cleanup

```bash
dsagt stop isaac-skills-demo
dsagt rm isaac-skills-demo            # add -y to skip the prompt
```

The shared catalog cache lives at `~/dsagt-projects/.skill_sources/` and is reused across projects; delete it to force a fresh clone on the next `setup-kb` / `add_skill_source`.

## Notes

- `mock_data/` is intentionally tiny and **not** real DFT output — the OUTCAR is a truncated stub. It exists only to exercise the conversion skill's parse-and-emit path.
- If your embedding backend isn't configured, `search_skills` degrades to the "requires a configured knowledge base" message; `install_skill` and the native mirror still work (they're pure filesystem operations).
- Swap in other catalogs the same way: `dsagt skills add isaac-skills-demo antigravity` (or `composio`, or any `https://github.com/owner/repo`).
