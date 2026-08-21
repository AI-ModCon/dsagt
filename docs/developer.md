# Developer Guide

How the DSAgt codebase is set up and how to work in it. For contribution
mechanics (branch/PR flow, commit style), see
[CONTRIBUTING.md](https://github.com/AI-ModCon/dsagt/blob/main/.github/CONTRIBUTING.md).

## Setup

DSAgt develops on [uv](https://github.com/astral-sh/uv) with Python 3.12 or 3.13:

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
uv run --no-sync python -m pytest -m "not integration" -q   # unit suite (~640 tests)
uv run --no-sync python -m pytest tests/test_config.py -q   # a single file
uv run --no-sync python -m pytest -m integration -v         # integration (needs creds)
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

## Codebase orientation

The [Architecture](architecture.md) page is the map of the system — the
capabilities, the single `dsagt-server` MCP layout, and the observability and
memory design. `CLAUDE.md` at the repo root records the house coding and prose
conventions (it doubles as instructions for AI coding agents working in the
repo); read it before a substantial change.

## Troubleshooting

**Agent command not found.** The agent CLI isn't installed or isn't on PATH —
see the [supported agents](index.md#supported-agents).

**MCP server not connecting.** Confirm the entry point resolves:

```bash
uv run which dsagt-server
```

If it's missing, reinstall:
`pip install --force-reinstall "git+https://github.com/AI-ModCon/dsagt.git"`.
