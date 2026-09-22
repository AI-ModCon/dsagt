# Contributing to DSAgt

<!-- md-shared:contributing:start -->
We recommend setting up a dsagt virtual environment with [uv](https://github.com/astral-sh/uv) with Python 3.12 or later; CI tests 3.12 and 3.13:

```bash
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync --all-groups          # runtime + dev + docs dependencies
uv sync --all-groups --all-extras   # plus the walkthrough extras (vasp-dft, combustion-simulation, tokamak-stability, plasma-turbulence)
source .venv/bin/activate      # so dsagt / dsagt-run / dsagt-server are on PATH
```

`uv.lock` is tracked. After a change to `pyproject.toml`, run `uv lock` and
commit the lock with it; CI syncs with `uv sync --locked`, which fails on a
lock that does not match.

## Tests

```bash
python -m pytest -m "not integration" -q   # unit suite
python -m pytest tests/test_config.py -q   # a single file
python -m pytest -m integration -v         # integration (needs creds)
```

A walkthrough under `use_cases/` is validated by running its Execution prompts
through a headless agent:

```bash
dsagt init <name> --agent claude|codex --location ~/dsagt-projects   # then stage data per the README
python tests/headless_usecases.py use_cases/<case> <name>            # --only N,M --from N --subst KEY=VALUE
```

The driver reads the agent from the project's config, continues one session
across the prompts, and appends each response to `<project>/headless_run.log`.
Judge the run by the README's post-conditions and the `trace_archive/` record
count, not by the log alone.

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

`CLAUDE.md` at the repository root is the contract an agent works to here:
what the project is, the commands, the glossary, and the invariants. How the
agent gets from that contract to a change is the developer's own tooling. If
you have no skills corpus of your own for coding, or want one in dsagt's
register, five skills under `basedata_aaron/` in
[AI-ModCon/dev_skills](https://github.com/AI-ModCon/dev_skills) are one
option:

| Skill | Load it when |
|---|---|
| `coding` | writing or changing code, removing code, committing |
| `documentation` | writing a docstring, comment, README, or plan; deciding where a fact is recorded |
| `cleanup` | aligning documents and memories with the code |
| `autodocs` | adding a page or a collection to this site |
| `write-like-aaron` | any prose: docs, comments, commit messages, pull-request descriptions |

Copy a skill directory into the agent's skills directory (for Claude Code,
`~/.claude/skills/<name>/` for every project, or `.claude/skills/<name>/` in
this checkout) and the agent loads it when its description matches the task.

## Where a function lands

The agent reaches dsagt two ways, and each owns one thing:

- **MCP tools** (`dsagt-server`) own dsagt's state: the registry, the
  knowledge base, memory, skills, and the execution records. A function that
  reads or writes `trace_archive/`, `kb_index/`, `.dsagt/`, or a contract's
  fingerprint is a tool.
- **`dsagt-run`** owns execution in the user's environment: anything that runs
  the user's code, data, or binaries, wrapped so the run is recorded. The
  agent invokes it from its own shell, which is the one process that has the
  user's activated environment (`PATH`, a venv or conda env, `module load`)
  on every platform; codex and cline give the MCP server only the env block
  dsagt writes.
- The **`dsagt` CLI** is for people. No script or agent invokes it.

So a built-in code that operates on the user's objects (a `Dataset`, a data
file, a binary) runs under `dsagt-run` in the user's environment and imports
nothing from the package; when it needs something dsagt holds, the agent
calls a tool for it beside the code. A function that needs only dsagt's own
state is a tool. Nothing is both.

## Codebase orientation

The [Architecture](https://ai-modcon.github.io/dsagt/architecture/) page explains in more detail the components of DSAgt — the
capabilities, the single `dsagt-server` MCP layout, and the observability and
memory design. `CLAUDE.md` contains helpful information for the human as well as agent developer.

## Troubleshooting

**Agent command not found.** The agent CLI isn't installed or isn't on PATH —
see the [supported agents](https://ai-modcon.github.io/dsagt/#supported-agents).

**MCP server not connecting.** Confirm the entry point resolves:

```bash
uv run which dsagt-server
```

If it's missing, reinstall:
`pip install --force-reinstall "git+https://github.com/AI-ModCon/dsagt.git"`.

## AI/LLM-assisted contributions

- **Remain accountable.** You are responsible for the accuracy, quality, and
  consequences of anything you submit, regardless of how it was produced. Using
  an AI tool does not transfer that responsibility to the tool.
- **Understand your work.** Review AI-generated code line by line before
  submitting it. You are responsible for its correctness, security, and scope,
  and for confirming it doesn't breach copyright.
- **Disclose it.** If AI/LLM tools were used to generate a substantial part of
  a PR, say so in the PR description.
- **Human review is mandatory.** An LLM review can supplement a human
  reviewer, but every PR needs a human reviewer who is accountable for the
  review.
- **No proprietary or personal data to AI tools.** Never send proprietary data,
  credentials, or personal information to a code generator or AI tool.

## Reporting issues

Open a GitHub issue with steps to reproduce, expected and actual behavior, the
OS and Python version, and the agent platform involved. For a security issue,
follow [SECURITY.md](https://github.com/AI-ModCon/dsagt/blob/main/.github/SECURITY.md)
instead of opening a public issue.
<!-- md-shared:contributing:end -->
