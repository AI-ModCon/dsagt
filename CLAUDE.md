# CLAUDE.md

## What this is

DSAgt (DataSmith Agent) is an MCP server and a CLI that give a user's own agent platform (Claude Code, Goose, Codex, opencode, Cline) code registration, a knowledge base, skill discovery, execution provenance, memory, and trace logging for building data-curation pipelines. Two facts every change respects: the agent talks to its own LLM provider and dsagt recovers its traces from the on-disk transcript; and all self-logging goes to one MLflow store, the project's `sqlite:///<pdir>/mlflow.db` unless `MLFLOW_TRACKING_URI` names a shared server, so a project is self-contained in its directory by default.

## Documents

- `README.md`: what it does, install, usage, rules.
- `.github/CONTRIBUTING.md`: the contributor guide (setup, tests, lint, docs build, review policy, the agentic workflow, AI-assisted contributions); `docs/developer.md` includes it, the way the site includes the README.
- `docs/`: the MkDocs site.
- `CHANGELOG.md` and `agent-card.md`: the release log and the Genesis agent card.
- `use_cases/`: end-to-end walkthroughs, published to the site by `hooks/gen_use_cases.py`; reference material outside the test suite.

## Commands

```bash
uv sync --all-groups                                        # install (add --all-extras for the walkthrough packages)
uv run --no-sync python -m pytest tests/test_<file>.py -q   # targeted tests
uv run --no-sync python -m pytest -m "not integration" -q   # unit suite, about 50 s
uv run ruff check src tests && uv run black src tests       # lint, format
uv run mkdocs build --strict                                # docs, what CI runs
```

## Glossary

- **project**: a directory with `.dsagt/config.yaml`, registered in `~/dsagt-projects/projects.yaml` (`session.init_project`).
- **session**: one agent launch, minted into `.dsagt/state.yaml` (`session.append_session`).
- **code**: a CLI executable registered at `<project>/codes/<name>/SKILL.md` (`registry.CodeRegistry`). "Tool" means an MCP tool.
- **skill**: an instruction workflow at `<project>/skills/<name>/` (`registry.SkillRegistry`). **Base skills** (`skill-creator`, `datacard-generator`, `aidrin`) are installed at every init (`skills.base_skills`).
- **source**, **corpus**: an external skill catalog, cloned and indexed one collection per source (`skills.SkillsCatalog`, `skills.KNOWN_SOURCES`).
- **collection**: a ChromaDB collection under `<project>/kb_index` (`knowledge.KnowledgeBase`).
- **execution record**: the JSON `dsagt-run` writes to `trace_archive/` (`provenance.run_and_record`), indexed into `code_use` by `provenance.CodeUseIndexer`.
- **explicit memory**, **episodic memory**: `memory.ExplicitMemory`, `memory.MemoryExtractor`.
- **trace**: one session's spans as plain data (`traces.Trace`). The **periodic pass** (`mcp.server._periodic_pass`, every 45 seconds) runs `traces.TraceCollector`; the **deferred final turn** is the open last turn a periodic pass withholds; **catch-up** re-collects the previous session at startup (`session.catch_up_extraction`).
- **AI-readiness check**: the default-on AIDRIN quality baseline around every tabular stage, one paragraph at the per-operation check rule (`readiness.INSTRUCTIONS_PARAGRAPH`); `aidrin` is a registered code in every project (`skills.base_skills`).
- **store**: the MLflow store traces go to, `MLFLOW_TRACKING_URI` when set, else the project's sqlite file (`observability.resolve_tracking_uri`); traces log to the **experiment** `dsagt-<8 hex>` derived from the project directory, or `mlflow.experiment` from the config (`observability.experiment_name`).

## Invariants

- The agent reaches dsagt two ways, and each owns one thing. MCP tools (`dsagt-server`) own dsagt's state: the registry, the knowledge base, memory, skills, and the execution records; a function that reads or writes `trace_archive/`, `kb_index/`, `.dsagt/`, or a contract's fingerprint is a tool. `dsagt-run` owns execution in the user's environment, invoked from the agent's own shell, which is the process that has the user's activated environment on every platform (codex and cline give the server only the env block). The `dsagt` CLI is for people; no script or agent invokes it. A built-in code that operates on the user's objects runs under `dsagt-run` and imports nothing from the package; what it needs from dsagt, the agent gets from a tool beside it.
- Provenance rides in the code spec's `executable` string (`dsagt-run --code <name> -- ...`), so a run through the agent's own shell is still recorded. The server offers no execute-by-name tool: the agent's shell is always available, and a server-side dispatch misses every run made outside it.
- The package holds no skill directories. A base skill is an entry in `skills.base_skills` whose directory is maintained upstream (the genesis catalog's `skills/basedata-skills/`, idtlab/AIDRIN).
- `dsagt init` is the one place collections are provisioned; `dsagt-server` opens only `<project>/kb_index`. The shared `kb_index/codes` collection holds the bundled tools and the base-skill codes, embedded once and stamped (`setup_core_kb.CODES_STAMP_FILE`); init copies it and loads no embedding model.
- A tool is registered on `dsagt-server` only when its handler is complete end to end; internal scaffolding for an unfinished path stays unregistered.
- `dsagt-server` derives its project from its cwd and behaves the same from a bare launch or `dsagt start`. The MCP-config env block carries routing only. dsagt never reads or writes an LLM-provider credential (`ANTHROPIC_*`, `OPENAI_*`, `GOOSE_*`); its own service credentials, the trace store's `MLFLOW_TRACKING_API_KEY` or `_TOKEN` and the embedding backend's `EMBEDDING_API_KEY`, are read from the shell or `~/.config/dsagt/env` (`session.load_user_env`) and never written into a project or an agent config.
- Agent traces come from the on-disk transcript through the periodic pass, the same way for all five agents.
- Modules on the `dsagt-run` path import no heavy dependency at module scope.

## Exceptions

- Run only the test file relevant to a change; the unit suite takes about 50 s. `test_*_integration.py` and `test_server_startup.py` load the local embedder, spawn subprocesses, or install into the venv; `test_mcp_wire.py` spawns `tests/wire_server.py` and speaks JSON-RPC over stdio.
- Use `python -m pytest`; the bare `pytest` binary on this machine resolves the wrong interpreter.
