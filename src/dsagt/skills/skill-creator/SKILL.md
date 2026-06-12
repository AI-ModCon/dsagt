---
name: skill-creator
description: "Author a new agent Skill (a SKILL.md directory in the open Agent Skills format) from the Anthropic template. Use when the user wants to create a skill, scaffold a SKILL.md, package a repeatable workflow as a reusable skill, turn instructions into a skill, or capture a procedure so the agent can auto-invoke it later. Produces a valid SKILL.md (name + description frontmatter, optional scripts/ and references/) and saves it into the project's skills directory."
metadata:
  version: "1.0"
---

# Skill Creator

Scaffold a new, spec-valid agent Skill from the Anthropic template and save it so the agent can discover and auto-invoke it.

A *skill* is a directory `<name>/SKILL.md`: YAML frontmatter (`name`, `description`) plus markdown instructions the agent follows when the description matches the task. It can bundle `scripts/` and `references/`. See [references/agent_skills_spec.md](references/agent_skills_spec.md) for the full contract.

## Workflow

Copy this checklist and check off steps as you go:

```
Progress:
- [ ] 1. Gather skill intent (name, purpose, triggers)
- [ ] 2. Draft from the template
- [ ] 3. Write the body (instructions/workflow)
- [ ] 4. Add scripts/ and references/ if needed
- [ ] 5. Validate the frontmatter
- [ ] 6. Save into the project (save_skill)
- [ ] 7. Confirm + note how it activates
```

### 1. Gather Intent

Ask the user (or infer from context):
- **name** — short, lowercase, hyphenated (e.g. `convert-vasp-outputs`). This becomes the directory name and the invocable name.
- **purpose** — one sentence on what the skill does.
- **triggers** — the user requests / phrasing that should make the agent reach for this skill. These become keywords in the `description`.

### 2. Draft From the Template

Start from [references/SKILL_template.md](references/SKILL_template.md). Fill the frontmatter:
- `name`: must equal the directory name.
- `description`: pack it with *what it does AND when to use it* (trigger phrases) — this is the only thing the agent sees when deciding to invoke. Keep it ≤ 1536 characters.

### 3. Write the Body

After the frontmatter, write the instructions the agent will follow. Prefer a copyable checklist (like this one) for multi-step workflows. Reference bundled files by relative path, e.g. `[reference](references/notes.md)`, or run a bundled script with `${CLAUDE_SKILL_DIR}/scripts/foo.py` so paths resolve regardless of cwd.

### 4. Add Supporting Files (optional)

- `scripts/` — runnable helpers the body invokes.
- `references/` — long docs/templates loaded on demand (keep them OUT of SKILL.md so they cost no tokens until used).

### 5. Validate

Confirm before saving:
- Frontmatter is valid YAML between `---` fences.
- `name` is present, lowercase-hyphenated, and equals the intended directory name.
- `description` is present and ≤ 1536 characters.
- Any `[link](references/...)` and `${CLAUDE_SKILL_DIR}/scripts/...` paths exist.

### 6. Save

Save via the **`save_skill`** MCP tool (registry server) with the `spec` (frontmatter dict: `name`, `description`, optional `tags`), the `body` markdown, and any `reference_files` (a `{relative_path: contents}` map). This writes `<project>/skills/<name>/` and indexes it for `search_skills`.

### 7. Confirm

Tell the user the skill was saved and how it activates: project skills are mirrored into the platform's native skill directory (e.g. `.claude/skills/`) at the next `dsagt start`, after which the agent auto-discovers it. To use it in the current session, restart the agent.
