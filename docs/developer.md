# Developer Guide

We recommend setting up a dsagt virtual environment with [uv](https://github.com/astral-sh/uv) with Python 3.12 or later; CI tests 3.12 and 3.13:

```bash
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync --all-groups          # runtime + dev + docs dependencies
source .venv/bin/activate      # so dsagt / dsagt-run / dsagt-server are on PATH
```

## Tests

```bash
python -m pytest -m "not integration" -q   # unit suite
python -m pytest tests/test_config.py -q   # a single file
python -m pytest -m integration -v         # integration (needs creds)
```

## Lint & format

CI enforces both on `src/` and `tests/` (scientific scripts under `use_cases/`
are exempt):

```bash
uv run ruff check src tests
uv run black src tests          # omit the paths to format everything you touched
```

## Docs

The site is MkDocs (Material). `mkdocs.yml` at the repo root is the site config;
`docs/` holds the pages. The `.github/workflows/docs.yml` workflow builds the
site with `--strict` on every PR and deploys it to GitHub Pages from `main`.

```bash
uv run mkdocs serve             # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict    # what CI runs
```

## Pull requests

- One concern per pull request. A small, focused pull request receives full
  and timely review; a monolithic refactor invites a cursory one and occludes important changes.
- Code an agent wrote gets a human review before it merges, the same as any
  other code.
- Work on a branch off `main`. Describe the intent, not the diff. Update
  `docs/` and `CHANGELOG.md` in the same pull request when behavior changes.
- `ruff check`, `black --check`, and the unit suite pass before review.
- This is pre-1.0 code: prefer clean removal over a compatibility shim.

## Agentic coding

Five skills are in the genesis catalog, under `skills/basedata-skills/` which steer consistent agentic code development:

| Skill | Purpose |
|---|---|
| `coding` | writing or changing code, removing code, committing |
| `documentation` | writing a docstring, comment, README, or plan; deciding where a fact is recorded |
| `cleanup` | aligning documents and memories with the code |
| `autodocs` | adding a page or a collection to this site |
| `write-like-aaron` | any prose: docs, comments, commit messages, pull-request descriptions |

Install these into the agent's skills directory (for Claude Code,
`~/.claude/skills/<name>/`) from a clone of the catalog. `CLAUDE.md` at the
repository root contains DSAgt specific instructions for agents.

## Codebase orientation

The [Architecture](architecture.md) page explains in more detail the components of DSAgt — the
capabilities, the single `dsagt-server` MCP layout, and the observability and
memory design. `CLAUDE.md` contains helpful information for the human as well as agent developer.

## Troubleshooting

**Agent command not found.** The agent CLI isn't installed or isn't on PATH —
see the [supported agents](index.md#supported-agents).

**MCP server not connecting.** Confirm the entry point resolves:

```bash
uv run which dsagt-server
```

If it's missing, reinstall:
`pip install --force-reinstall "git+https://github.com/AI-ModCon/dsagt.git"`.

## Reporting issues

Open a GitHub issue with steps to reproduce, expected and actual behavior, the
OS and Python version, and the agent platform involved. For a security issue,
follow [SECURITY.md](https://github.com/AI-ModCon/dsagt/blob/main/.github/SECURITY.md)
instead of opening a public issue.
