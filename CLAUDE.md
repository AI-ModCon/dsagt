# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSAGT (DataSmith Agent) is an AI-assisted data pipeline builder that exposes MCP (Model Context Protocol) servers to agent platforms (Claude Code, Goose, Roo, Cline, Codex). It helps domain scientists create reproducible, auditable data curation pipelines through iterative, knowledge-driven tool generation.

## Two run modes

1. **BYOA (Bring Your Own Agent)** — default for everyday use. `dsagt init --agent <name>` writes per-agent MCP config artifacts; `dsagt mlflow <project>` backgrounds MLflow and prints the OTel routing exports the user pastes into the shell that runs `claude` / `goose` / etc. Project / agent / session_id are read from `<project>/dsagt_config.yaml` + `.runtime` (single source of truth, no env-var duplication). `dsagt memory --project X` extracts episodic memory from accumulated traces — but only from proxy-shape traces (see #2).
2. **Proxy mode** — `dsagt start --enable-proxy <project>` interposes a LiteLLM proxy between the agent and its LLM provider. The proxy autologs every LLM call into MLflow with `mlflow.spanInputs` / `mlflow.spanOutputs` populated, which is the only trace shape `dsagt memory` knows how to extract from. Use this when you want both (a) request/response columns populated in the MLflow UI and (b) episodic memory extraction. Native agent OTel emission (Claude Code, Goose) is visible in the UI but uses a different shape (`api_response_body` log events), so memory extraction skips those traces.

## Commands

```bash
uv sync --all-groups                                                  # install
uv run --no-sync python -m pytest tests/test_<file>.py -q             # targeted tests
uv run black .                                                         # format
uv run ruff check .                                                    # lint
```

**`python -m pytest` not bare `pytest`.** The bare-binary form picks up the wrong pytest on this machine and crashes with `ModuleNotFoundError: dsagt`.

**Don't run the full suite by default.** ~50s for 547 tests is too slow for an iteration loop. Run only the test file relevant to the change. `tests/test_config.py` covers session, init, agents, BYOA hints, launch shim, and memory state. Skip `test_integration.py`, `test_*_integration.py`, `test_server_startup.py`, `test_dependency_integration.py` unless explicitly relevant — they hit network or spawn subprocesses.

## Code Organization

The codebase separates **commands** (entry points with argparse, launched as CLI tools or subprocesses) from **modules** (importable logic). Commands live in `src/dsagt/commands/`, modules live in `src/dsagt/`.

**Commands** (`src/dsagt/commands/`):
- `cli.py` — `dsagt init / mlflow / memory / info / list / mv / rm / setup-kb / smoke-test / stop / start` (user-facing CLI). `dsagt start --enable-proxy` activates proxy mode; without `--enable-proxy` it's the supervised BYOA equivalent (start MLflow + agent under one process tree).
- `proxy_server.py` — `dsagt-proxy` (LiteLLM proxy with OTel autolog). Spawned by `dsagt start --enable-proxy`.
- `run_tool.py` — `dsagt-run` (tool execution wrapper).
- `setup_core_kb.py` — core KB setup (called via `dsagt setup-kb`).
- `info.py` — `dsagt info` (project / config introspection).

(The MCP server — `dsagt-server` — lives in the `src/dsagt/mcp/` package, see below.)

**Modules** (`src/dsagt/`):
- `session.py` — Project init, agent config generation, env-var resolution, config load/validate, service start/stop, end-of-session memory extraction orchestration.
- `agents/` — Per-agent-platform setup (`base.py` ABC + `claude.py` / `goose.py` / `cline.py` / `roo.py` / `codex.py`). Each subclass owns its `write_static`, `write_dynamic`, `env_overrides`, `byoa_env_hints`, `launch_oneliner`. Shared helpers (`_mcp_env_block`, `_render_launch_shim`, `_build_mcp_servers_dict`) in `base.py`.
- `knowledge.py` — FAISS/ChromaDB document retrieval, embedding backends, per-collection routing.
- `registry.py` — `ToolRegistry` (CLI tools) + `SkillRegistry` (agent instruction skills), KB indexing.
- `provenance.py` — Tool execution records (`run_and_record`, `ToolRecordStore`), execution-record indexing into ChromaDB, pipeline reconstruction (`reconstruct_pipeline`, dependency graph).
- `observability.py` — MLflow / OTel tracing setup, `init_tracing`, span helpers.
- `memory.py` — Explicit memory (YAML), episodic-memory extraction prompt + LLM call, outlier detection, `extract_session`.
- `skills.py` — External skill catalog data plane (`SkillsCatalog`: clone/sync/index/install), the `SkillRouter` render facade, and the Genesis-derived keyword scorer (`rank_skills`).

**MCP server** (`src/dsagt/mcp/`) — the single merged `dsagt-server`. `server.py` owns `main()`, the shared-KB startup (`_build_kb_from_config`), and the dispatch shell (`build_dispatch_server`); the tool surface is split by concern across `registry_tools.py` (tool registry + execution + provenance, 8 tools), `knowledge_tools.py` (KB retrieval, 6), `memory_tools.py` (explicit memory + suggestions, 4), and `skill_tools.py` (skill search/install/sources, 5). Each `*_tools.py` exposes a `_*_tools_and_handlers()` factory (composed by `create_dsagt_server`) plus a `create_*_server` test wrapper.

Entry points are defined in `pyproject.toml` `[project.scripts]`: the CLI/proxy/run/setup-kb tools point to `dsagt.commands.*:main`, and `dsagt-server` points to `dsagt.mcp.server:main`.

**Bundled assets** (shipped as `package-data` in `pyproject.toml`):
- `src/dsagt/tools/` — built-in tool specs (markdown + YAML frontmatter) copied into new projects.
- `src/dsagt/skills/` — built-in skills (e.g., `datacard-generator`) the agent discovers via `search_skills`.
- `src/dsagt/dsagt_instructions.md` — agent-platform-agnostic system instructions injected into per-agent files at init.

**`use_cases/`** holds end-to-end domain walkthroughs (`microbial_isolates/`, `cryoem/`, `isaac_vasp/`). They are reference material for users, not part of the test suite. `isaac_vasp/` is currently in active development on this branch.

## BYOA artifacts

`dsagt init --agent X --location <path>` writes (in the project dir):
- `dsagt_config.yaml` — internal config (project name, agent, mlflow port pinned at init, embedding/knowledge/extraction settings). No user-facing fields, no credentials.
- Per-agent instructions file (e.g., `CLAUDE.md`, `.goosehints`, `AGENTS.md`).
- Per-agent MCP config artifact (`.mcp.json` for claude, `goose.yaml` for goose, `cline_mcp_settings.json` via `cline mcp add`, `.roo/mcp.json`, `.codex-data/config.toml`). All include the env block (DSAGT_PROJECT, DSAGT_PROJECT_DIR, MLFLOW_TRACKING_URI, EMBEDDING_*) so MCP-server children that don't inherit shell env still log to the right MLflow.
- `dsagt-launch.sh` — bash shim that exports all dsagt-internal env (DSAGT_*, MLFLOW_*, OTEL_*, agent-specific telemetry verbosity flags), resolves the OTel experiment-id header at run time via curl, then execs the agent. The user runs this directly to launch.

`dsagt memory --project X` tracks a high-water-mark in `<pdir>/.dsagt/extracted_at.json` so re-runs only process new traces.

## Architecture

### MCP Server

A single merged `dsagt-server` (`src/dsagt/mcp/`) exposes 23 tools across four concern modules under one `Server` + one shared `KnowledgeBase`:

1. **Registry tools** (`mcp/registry_tools.py` + `registry.py` / `provenance.py`) — tool analysis, registration, dependency installation, command/file/http execution, and pipeline reconstruction. Tools are saved as markdown specs with YAML frontmatter.
2. **Knowledge tools** (`mcp/knowledge_tools.py` + `knowledge.py`) — semantic search over document collections (FAISS + ChromaDB, optional cross-encoder reranking); long ops run as background jobs.
3. **Memory tools** (`mcp/memory_tools.py` + `memory.py`) — explicit memory + outlier suggestions (`kb_remember` / `kb_get_memories` / …).
4. **Skill tools** (`mcp/skill_tools.py` + `skills.py`) — skill search / install + external catalog sources.

### Observability

- **MLflow** — Token usage, cost, latency, full LLM-call traces via OTel. Started by `dsagt mlflow <project>` (foreground, in its own terminal). Port is pinned at init time and lives in `dsagt_config.yaml`.
- **dsagt-run** (`commands/run_tool.py` + `provenance.py`) — Wraps tool commands; captures execution layer (command, stdout/stderr, timing, file lists) into `trace_archive/`.
- **MCP-server OTel** — `dsagt-server` calls `init_tracing()` at startup; its tool spans (kb.*, registry.*) flow to MLflow alongside the agent's LLM-call spans.

### Memory System

- **Episodic memory** (`memory.py:extract_session`) — End-of-session LLM extraction of facts from MLflow traces into ChromaDB, with per-category outlier detection via embedding centroids. Triggered by `dsagt memory --project X`.
- **Explicit memory** (`memory.py:ExplicitMemory`) — User-confirmed facts in YAML, loaded into agent context at session start.

### Key Design Patterns

- **Agent-agnostic**: DSAGT is infrastructure, not an agent. Capabilities are MCP services.
- **Session isolation**: Each project gets its own directory with config, tools, skills, kb_index, trace_archive, mlflow data.
- **Tools vs Skills**: Tools are CLI executables in `<project>/tools/` (specs with parameters, wrapped by dsagt-run). Skills are agent instruction workflows in `<project>/skills/` (SKILL.md + reference docs). Both are discoverable via ChromaDB-backed semantic search.

## DSAGT Pipeline Builder Workflow

When acting as a pipeline builder (using the MCP servers), follow these constraints:

1. **Never directly access data** — all data operations go through registered tools.
2. **Tool preference hierarchy**: Registered tool → KB package tool → Custom implementation.
3. **Generate paired tools** — every data operation gets a check tool (pre/post audit) and an operation tool.
4. **Audit everything** — before/after JSON reports saved to `audit/`.
5. **One step at a time** — iterate with the user, confirming approach before execution.

## Testing Patterns

- Tests use pytest with `subprocess.run` mocking for command execution.
- MCP server tests invoke handlers directly (no stdio transport).
- Async tests for server handlers.
- Temp directories for isolation; the `_use_tmp_registry` fixture in `tests/test_config.py` patches `DEFAULT_PROJECTS_BASE` and the project registry to `tmp_path`.
- Integration tests in `test_*_integration.py` require real `EMBEDDING_*` / `LLM_*` credentials.
- A handful of tests are class-skipped under `TestProviderEnvInjection` with a long reason about "old-code-shape env_overrides" — those describe the pre-Phase-1 design where `env_overrides` did broad provider-credential translation, which is now narrower (model env-var pinning only; provider creds + base URLs come via `proxy_env_overrides` or per-agent config files). Kept around for reference; safe to delete once Phase 2 is stable on real workloads.
