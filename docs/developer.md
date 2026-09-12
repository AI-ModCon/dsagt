# Developer Guide

How the DSAgt codebase is set up, how to work in it, and how a change gets
reviewed and merged.

## Setup

DSAgt develops on [uv](https://github.com/astral-sh/uv) with Python 3.12 or later; CI tests 3.12 and 3.13:

```bash
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync --all-groups          # runtime + dev + docs dependencies
source .venv/bin/activate      # so dsagt / dsagt-run / dsagt-server are on PATH
```

## Tests

Use `python -m pytest`, not bare `pytest` (the bare binary can resolve the wrong
interpreter):

```bash
python -m pytest -m "not integration" -q   # unit suite
python -m pytest tests/test_config.py -q   # a single file
python -m pytest -m integration -v         # integration (needs creds)
```

Integration tests hit real embedding/LLM providers and need `EMBEDDING_*` /
`LLM_*` credentials in the environment; they're excluded from CI and the default
local run.

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
  and timely review; a monolithic refactor receives a cursory one and hides
  the change that matters.
- Code an agent wrote gets a human review before it merges, the same as any
  other code.
- Work on a branch off `main`. Describe the intent, not the diff. Update
  `docs/` and `CHANGELOG.md` in the same pull request when behavior changes.
- `ruff check`, `black --check`, and the unit suite pass before review.
- This is pre-1.0 code: prefer clean removal over a compatibility shim.

## Agentic coding

Five skills in the genesis catalog, under `skills/basedata-skills/`, carry
the rules an agent follows in this repository:

| Skill | Load it before |
|---|---|
| `coding` | writing or changing code, removing code, committing |
| `documentation` | writing a docstring, comment, README, or plan; deciding where a fact is recorded |
| `cleanup` | aligning documents and memories with the code |
| `autodocs` | adding a page or a collection to this site |
| `write-like-aaron` | any prose: docs, comments, commit messages, pull-request descriptions |

Install them into the agent's skills directory (for Claude Code,
`~/.claude/skills/<name>/`) from a clone of the catalog. `CLAUDE.md` at the
repository root holds what is specific to DSAgt: the document map, the
commands, the glossary, and the invariants. (The skills keep their working
notes in `DESIGN.md`, `DEVELOPMENT.md`, and `history/` at the repository
root; those are ignored by git and never part of a pull request.)

## Codebase orientation

The [Architecture](architecture.md) page is the map of the system — the
capabilities, the single `dsagt-server` MCP layout, and the observability and
memory design. `CLAUDE.md` at the repo root records what is specific to this repository;
read it before a substantial change.

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
