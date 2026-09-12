# CLAUDE.md

## What this is

DSAgt (DataSmith Agent) is an MCP server and a CLI that give a user's own agent platform (Claude Code, Goose, Codex, opencode, Cline) code registration, a knowledge base, skill discovery, execution provenance, memory, and trace logging for building data-curation pipelines. Two facts every change respects: the agent talks to its own LLM provider and dsagt recovers its traces from the on-disk transcript; and all self-logging goes to the serverless store `sqlite:///<pdir>/mlflow.db`, so a project is self-contained in its directory.

## Documents

- `README.md`: what it does, install, usage, rules.
- `docs/`: the MkDocs site. `docs/developer.md` is the contributor guide (setup, tests, lint, docs build, review policy, the agentic workflow).
- `CHANGELOG.md` and `agent-card.md`: the release log and the Genesis agent card.
- `use_cases/`: end-to-end walkthroughs, published to the site by `hooks/gen_use_cases.py`; reference material outside the test suite.

## Commands

```bash
uv sync --all-groups                                        # install
uv run --no-sync python -m pytest tests/test_<file>.py -q   # targeted tests
uv run --no-sync python -m pytest -m "not integration" -q   # unit suite, about 50 s
uv run ruff check src tests && uv run black src tests       # lint, format
uv run mkdocs build --strict                                # docs, what CI runs
```

## Glossary

- **project**: a directory with `.dsagt/config.yaml`, registered in `~/dsagt-projects/projects.yaml` (`session.init_project`).
- **session**: one agent launch, minted into `.dsagt/state.yaml` (`session.append_session`).
- **code**: a CLI executable registered at `<project>/codes/<name>/SKILL.md` (`registry.CodeRegistry`). "Tool" means an MCP tool.
- **skill**: an instruction workflow at `<project>/skills/<name>/` (`registry.SkillRegistry`). **Base skills** are installed at every init (`skills.BASE_SKILLS`).
- **source**, **corpus**: an external skill catalog, cloned and indexed one collection per source (`skills.SkillsCatalog`, `skills.KNOWN_SOURCES`).
- **collection**: a ChromaDB collection under `<project>/kb_index` (`knowledge.KnowledgeBase`).
- **execution record**: the JSON `dsagt-run` writes to `trace_archive/` (`provenance.run_and_record`), indexed into `code_use` by `provenance.CodeUseIndexer`.
- **explicit memory**, **episodic memory**: `memory.ExplicitMemory`, `memory.MemoryExtractor`.
- **trace**: one session's spans as plain data (`traces.Trace`). The **heartbeat** (`mcp.server._heartbeat`) runs `traces.TraceCollector`; the **deferred final turn** is the open last turn a periodic pass withholds; **catch-up** re-collects the previous session at startup (`session.catch_up_extraction`).
- **readiness gate**: the opt-in AIDRIN check around every tabular stage (`readiness.instructions_block`).
- **store**: the project's MLflow sqlite file (`observability.resolve_tracking_uri`).

## Invariants

- Provenance rides in the code spec's `executable` string (`dsagt-run --code <name> -- ...`), so a run through the agent's own shell is still recorded. The server offers no execute-by-name tool: the agent's shell is always available, and a server-side dispatch misses every run made outside it.
- The package holds no skill directories. A base skill is an entry in `skills.BASE_SKILLS` whose directory is maintained upstream (the genesis catalog's `skills/basedata-skills/`, idtlab/AIDRIN).
- `dsagt init` is the one place collections are provisioned; `dsagt-server` opens only `<project>/kb_index`.
- A tool is registered on `dsagt-server` only when its handler is complete end to end; internal scaffolding for an unfinished path stays unregistered.
- `dsagt-server` derives its project from its cwd and behaves the same from a bare launch or `dsagt start`. The MCP-config env block carries routing only; dsagt never reads or writes provider credentials.
- Agent traces come from the on-disk transcript through the heartbeat, the same way for all five agents.
- Modules on the `dsagt-run` path import no heavy dependency at module scope.

## Exceptions

- Run only the test file relevant to a change; the unit suite takes about 50 s. `test_integration.py`, `test_*_integration.py`, `test_server_startup.py`, and `test_dependency_integration.py` reach the network or spawn subprocesses.
- Use `python -m pytest`; the bare `pytest` binary on this machine resolves the wrong interpreter.
