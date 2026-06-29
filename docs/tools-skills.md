# Tools and Skills

## Tools

Tools are CLI executables defined as markdown files with YAML frontmatter under `<project>/tools/`. The agent registers new tools via the MCP server's `save_tool_spec` tool.

A tool spec includes:

- A YAML frontmatter block describing the command, arguments, dependencies, and tags.
- A markdown body with usage examples and notes for the agent.

Example tool spec structure:

```markdown
---
name: csvstat
command: csvstat
dependencies: []
tags: [csv, statistics]
---

Prints descriptive statistics for all columns in a CSV file.

Usage: csvstat [options] [FILE]
```

DSAgt wraps every registered tool with `dsagt-run` for provenance capture and `uv run --with` for Python dependencies, so the agent can call any tool without managing environments manually.

### Bundled Tools

DSAgt ships a `scan_directory` tool in `src/dsagt/tools/` that is indexed into the shared Tool Specs collection by `dsagt init` (always provisioned).

## Skills

Skills are instruction-based agent workflows in `<project>/skills/`. Each skill is a directory containing a `SKILL.md` file and optional reference documents.

### Skill discovery architecture

![DSAgt skill routing](assets/skills-routing.png)

Skills live in **two tiers**, and a single MCP service — the **SkillRouter** — is the one entry point that routes every skill operation between them:

- **Catalog tier** — skills that exist in external repositories but are *not yet installed*. DSAgt federates many sources (`scientific`, `anthropic`, `antigravity`, `composio`, `genesis`, or any git URL); each is sparse-cloned and indexed into its own per-source ChromaDB collection. The agent browses this tier with `search_skills` and manages sources with `add_skill_source` / `list_skill_sources`.
- **Installed + created tier** — skills drawn into the project's Skill Directory (`<project>/skills/`), either installed from the catalog (`install_skill`) or authored in place (e.g. with the bundled `skill-creator`). At `dsagt start` these are mirrored into each agent's *native* skill directory (`.claude/`, `.agents/`, `.cline/`, `.roo/`), where the agent auto-discovers and auto-invokes them.

The diagram's three bands trace a skill's lifecycle: **Discovery** (the router) → **Registration** (the searchable catalog) → **Progressive Exposure** (the native Skill Directory the agent loads on its own).

#### Why this shape

- **Search is catalog-only.** Every supported agent (Claude, Codex, Goose, Cline, Roo) natively auto-discovers `SKILL.md` folders, so installed skills never need to be indexed or returned by a tool — the harness already loads them. That frees `search_skills` to do the one thing native discovery cannot: browse a catalog of potentially thousands of *un*installed skills without holding them all in context. Catalogs are indexed on frontmatter (name + description + tags), which keeps those summaries compact and avoids diluting the embedding with full SKILL.md bodies.
- **Keyword fallback, no embedder required.** When no embedding model is configured, the router falls back to a zero-dependency token-overlap scorer over the local clone cache, so `search_skills` still works (just less fuzzy) — no model download, no API key.
- **One router, not scattered policy.** Backend selection (semantic vs. keyword), the catalog/installed split, and source bookkeeping all live in the SkillRouter rather than being re-implemented at each MCP and CLI call site, so the behavior can't drift between them.
- **Federated and provenance-preserving.** Each source is an independent per-source collection, so re-syncing one never disturbs another; installing a catalog skill preserves its upstream `LICENSE`/`NOTICE` and stamps a `PROVENANCE.txt` into the installed directory.

### Bundled Skills

DSAgt ships a `skill-creator` skill in `src/dsagt/skills/` (for scaffolding new SKILL.md skills). Bundled and installed skills are **not** indexed for search — every supported agent natively auto-discovers `SKILL.md` folders, so `search_skills` is reserved for the *catalog* tier (skills you can install but haven't yet). Domain skills — including the MODCON `datacard-generator` — are sourced from external catalogs (enabled from the agent with `add_skill_source`, or chosen at `dsagt init`) rather than bundled, so they stay current upstream.

### Adding Skills

Place a new directory under `<project>/skills/` with a `SKILL.md` describing the workflow. The next `dsagt start` mirrors it into the agent's native skill directory (e.g. `.claude/skills/`), after which the agent auto-discovers and invokes it — no indexing step.
