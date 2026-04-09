---
name: generating-datacards
description: Generates MODCON Datacard v1 documentation for scientific datasets by introspecting a directory and filling a structured template. Use when the user asks to create a datacard, dataset card, dataset documentation, dataset metadata, document a dataset, or prepare a dataset for sharing. Supports three readiness levels: Level 1 (Discoverable), Level 2 (Interoperable & Reusable), Level 3 (Understandable & Trustworthy).
---

# Generating Datacards

Introspect a dataset directory to auto-populate a MODCON Datacard v1, then prompt the user for anything that couldn't be inferred.

## Workflow

Copy this checklist and check off steps as you go:

```
Progress:
- [ ] 1. Gather context (path + level)
- [ ] 2. Introspect dataset directory
- [ ] 3. Auto-fill the data card
- [ ] 4. Prompt for missing fields
- [ ] 5. Generate output file
- [ ] 6. Review summary
```

### 1. Gather Context

Ask the user:
- **Dataset path** — directory to document
- **Readiness level** — choose one:
  - **Level 1 — Discoverable**: identification, description, file structure, keywords
  - **Level 2 — Interoperable & Reusable**: Level 1 + contacts, access, license, provenance, authors, related resources
  - **Level 3 — Understandable & Trustworthy**: Level 2 + schema, quality, integrity, AI/ML, variable docs

### 2. Introspect the Dataset Directory

See [reference/introspection-commands.md](reference/introspection-commands.md) for exact commands to run for file structure, schema extraction, and existing metadata files.

### 3. Auto-Fill the Data Card

Read `references/datacard_template_v1.md`. Populate fields using this decision table:

| Field | Auto-fill if… | Otherwise |
|-------|--------------|-----------|
| `datacard_creation.created_date` | Always | — |
| `datacard_creation.creation_method` | Always → `"hybrid"` | — |
| `dataset_info.data_formats` | File extensions found | Prompt |
| `dataset_info.modalities` | File types recognized | Prompt |
| `dataset_info.features` | Column headers extracted | Prompt at Level 2+ |
| `dataset_info.splits` | train/test/val dirs found | Leave empty |
| `dataset_counts` | File/record counts available | Prompt |
| `dataset_storage` | `du` output available | Prompt |
| `title` / `name` | README or metadata file found | Prompt |
| Description | README found | Prompt |
| Keywords | Metadata files found | Prompt at Level 1+ |
| License | LICENSE file found | Prompt at Level 2+ |
| Authors / contributors | CITATION.cff found | Prompt at Level 2+ |
| Citation | .bib or CITATION.cff found | Prompt at Level 2+ |

Cross-check these fields for consistency between YAML frontmatter and markdown body: `title`/`name`, `authors`/`contributors`, `license`, `dataset_info.data_formats`, key dates.

### 4. Prompt for Missing Fields

Present auto-discovered values for confirmation, then ask for unfilled fields. Ask **3–5 at a time**. See [reference/field-prompts.md](reference/field-prompts.md) for the full per-level field list.

### 5. Generate the Data Card

- **Filename**: `modcon_datacard_<snake_case_name>.md`
- **Location**: save inside `<dataset_dir>/` by default; ask if the user prefers elsewhere
- Replace all `[!TODO]`, `<REPLACE:...>`, `<INSTRUCTIONS:...>`, `<metadata_key:...>` — none should appear in output
- Set `dataset_readiness.level` to chosen level
- Unprovided fields: `"N/A"` in YAML, brief "Not provided." in markdown body

### 6. Review Summary

Present:
- ✅ Auto-populated fields
- ✏️ User-provided fields
- ⬜ Empty/N/A fields (with reason)
- 💡 Suggestions for improvement

Ask if the user wants to revise any sections.

## References

- **Template**: `references/datacard_template_v1.md`
- **Datasheet companion**: `references/datasheet_template.md` — consult if user mentions "datasheet" questions
- **Introspection commands**: [reference/introspection-commands.md](reference/introspection-commands.md)
- **Per-level field prompts**: [reference/field-prompts.md](reference/field-prompts.md)
- **Lookup tables** (OSTI codes, sensitivity tiers): [reference/lookup-tables.md](reference/lookup-tables.md)