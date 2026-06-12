# Agent Skills — condensed contract

A *skill* packages instructions (and optionally code/docs) so an agent can
discover and follow a repeatable workflow. dsagt skills follow the open
Agent Skills standard, which is what Claude Code, Cursor, Codex, and
Antigravity all read — so one SKILL.md works across platforms.

## Directory layout

```
<skill-name>/
├── SKILL.md          # required — frontmatter + instructions
├── references/       # optional — docs/templates loaded on demand
└── scripts/          # optional — runnable helpers
```

- The **directory name is the invocable name** (e.g. `.claude/skills/deploy/` → `/deploy`). Keep it lowercase, hyphenated.

## Frontmatter

YAML between `---` fences. Common fields:

| Field | Required | Notes |
|-------|----------|-------|
| `name` | recommended | Should equal the directory name. Lowercase-hyphenated. |
| `description` | **yes (in practice)** | What it does AND when to use it (trigger phrases). The agent sees only this when deciding to invoke. **≤ 1536 characters.** |
| `tags` | no | List of keywords; dsagt uses these for `search_skills` tag filters. |
| `metadata` | no | Free-form (e.g. `version`). Ignored by the platform. |
| `license` | no | Free-form. Ignored by the platform. |

Unknown/extra frontmatter fields are **silently ignored** by Claude Code, so dsagt-specific fields are safe to include.

## How discovery works

- At session start, each installed skill's `name` + `description` are loaded into the agent's context. The full SKILL.md body loads only when the skill is invoked (lazy — zero cost until used).
- The agent auto-invokes a skill when the `description` matches the task; the user can also invoke it directly (`/skill-name`).
- A **newly created** top-level skills directory is only picked up after the agent restarts.

## Body conventions

- Lead with a copyable progress checklist for multi-step workflows.
- Keep long material in `references/` (loaded on demand) rather than inline, to save context tokens.
- Reference bundled files by relative path, or run scripts via `${CLAUDE_SKILL_DIR}/scripts/...` so paths resolve regardless of working directory.

## Two tiers in dsagt

- **Catalog** — skills indexed from external GitHub source repos, searchable via `search_skills` but not installed. Not in context.
- **Installed** — skills in `<project>/skills/` (saved via `save_skill` or installed via `install_skill`). Mirrored into the platform's native skill dir at `dsagt start`, then natively discovered.
