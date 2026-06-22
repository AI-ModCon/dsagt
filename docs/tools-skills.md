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

DSAgt ships a `datacard-generator` skill in `src/dsagt/skills/` with reference templates for generating dataset documentation. It is indexed into the global Skills collection by `dsagt setup-kb`.

### Adding Skills

Place a new directory under `<project>/skills/` with a `SKILL.md` describing the workflow. The knowledge server indexes it automatically on next startup, or trigger a re-index via `kb_ingest`.
