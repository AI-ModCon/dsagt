# Contributing to DSAgt

Thanks for your interest in improving DSAgt. This guide covers the mechanics;
for how the codebase is organized and why, see the
[Developer Guide](https://ai-modcon.github.io/dsagt/developer/) (source:
`docs/developer.md`).

## Getting started

DSAgt develops on [`uv`](https://github.com/astral-sh/uv) with Python 3.12 or
3.13.

```bash
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync --all-groups          # runtime + dev + docs dependencies
source .venv/bin/activate      # so `dsagt` and the helpers are on PATH
```

Work on a feature branch off `main`; open a pull request when it's ready.

## Tests

Run the unit suite before opening a PR (`python -m pytest`, **not** bare
`pytest` — the bare binary can resolve the wrong interpreter):

```bash
uv run --no-sync python -m pytest -m "not integration" -q
```

Run a single file while iterating:

```bash
uv run --no-sync python -m pytest tests/test_config.py -q
```

Integration tests (`-m integration`) hit real embedding/LLM providers and need
`EMBEDDING_*` / `LLM_*` credentials in the environment; they're excluded from CI
and the default local run.

## Lint & format

CI enforces both on `src/` and `tests/` (contributor scripts under
`use_cases/` are exempt):

```bash
uv run ruff check src tests
uv run black src tests          # drop the path to format everything you touched
```

## Pull requests

- Keep PRs focused and reasonably small; describe the intent, not just the diff.
- Update `docs/` and `CHANGELOG.md` when behavior changes.
- Make sure `ruff`, `black --check`, and the non-integration tests pass.
- This is pre-1.0, dev-stage code: prefer clean removal over back-compat shims.

## Reporting issues

Open a GitHub issue with steps to reproduce, expected vs actual behavior, your
OS/Python version, and the agent platform involved. For security issues, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
