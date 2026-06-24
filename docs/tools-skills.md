# Tools and Skills

## Tools

Tools are CLI executables defined as markdown files with YAML frontmatter under `<project>/tools/`. The agent registers new tools via the registry MCP server's `save_tool_spec` tool.

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

The registry server wraps every registered tool with `dsagt-run` for provenance capture and `uv run --with` for Python dependencies, so the agent can call any tool without managing environments manually.

### Bundled Tools

DSAgt ships a `scan_directory` tool in `src/dsagt/tools/` that is indexed into the global Tool Specs collection by `dsagt setup-kb`.

## Skills

Skills are instruction-based agent workflows in `<project>/skills/`. Each skill is a directory containing a `SKILL.md` file and optional reference documents. The agent discovers skills via `search_skills`.

### Bundled Skills

DSAgt ships a `skill-creator` skill in `src/dsagt/skills/` (for scaffolding new SKILL.md skills). Bundled and installed skills are **not** indexed for search — every supported agent natively auto-discovers `SKILL.md` folders, so `search_skills` is reserved for the *catalog* tier (skills you can install but haven't yet). Domain skills — including the MODCON `datacard-generator` — are sourced from external catalogs (`dsagt skills add <project> genesis`) rather than bundled, so they stay current upstream.

### Adding Skills

Place a new directory under `<project>/skills/` with a `SKILL.md` describing the workflow. The next `dsagt start` mirrors it into the agent's native skill directory (e.g. `.claude/skills/`), after which the agent auto-discovers and invokes it — no indexing step.
