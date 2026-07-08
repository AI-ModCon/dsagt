# SKILL.md template

Copy the block below into `<name>/SKILL.md` and fill it in. Only the
frontmatter `name` + `description` are required; everything else is
optional. (Based on the Anthropic skill template / open Agent Skills
standard — https://github.com/anthropics/skills/tree/main/template.)

```markdown
---
name: my-skill-name
description: A clear description of WHAT this skill does and WHEN to use it — include the user phrasings/triggers that should invoke it. (≤ 1536 chars; this is the only text the agent sees when deciding to invoke.)
# optional:
# tags: [domain, keyword]
# metadata:
#   version: "1.0"
---

# My Skill Name

One or two sentences framing the task this skill handles.

## Workflow

Copy this checklist and check off steps as you go:

```
Progress:
- [ ] 1. ...
- [ ] 2. ...
```

### 1. ...

Step-by-step instructions. Reference bundled docs by relative path:
[details](references/details.md). Run bundled scripts with an absolute
skill-dir path so cwd doesn't matter:

    python ${CLAUDE_SKILL_DIR}/scripts/helper.py

## Notes / Guidelines
- ...
```

## Optional bundled files

```
my-skill-name/
├── SKILL.md          (required)
├── references/       (long docs/templates, loaded on demand)
│   └── details.md
└── scripts/          (runnable helpers the body invokes)
    └── helper.py
```
