# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSAGT (DataSmith Agent) is an AI-assisted data pipeline builder that exposes an MCP (Model Context Protocol) server to agent platforms (Claude Code, Goose, Roo, Cline, Codex, opencode). It helps domain scientists create reproducible, auditable data curation pipelines through iterative, knowledge-driven tool generation.

## Run model (BYOA, serverless)

DSAGT is **BYOA (Bring Your Own Agent)**: the agent talks to its own LLM provider directly — DSAGT never interposes on its network traffic (the LiteLLM proxy was removed). `dsagt init --agent <name>` writes the per-agent instructions file + MCP config and nothing to paste (no launch shim, no OTel/env instructions). Two supported, behaviorally-identical start flows:

1. **Bare launch** — start the agent from the project dir (`cd <project> && claude` / `goose` / …).
2. **`dsagt start <project>`** — spawns the agent for you in the foreground and, on exit, runs post-session `run_extraction`. Its real job is owning a reliable session-end trigger.

The MCP server is self-sufficient: it derives the project from its cwd (`dsagt_config.yaml`), with the MCP-config env block as robustness, and behaves identically regardless of how it was launched.

**Serverless store.** Self-logging (MCP-server + `dsagt-run` spans, and Claude's `mlflow autolog` Stop hook) goes to a `sqlite:///<pdir>/mlflow.db` MLflow store — **no server to run** (SQLite is MLflow's supported serverless backend; the filesystem `./mlruns` store is deprecated). The resolver (`observability.resolve_tracking_uri`) order is `MLFLOW_TRACKING_URI` env → config → that sqlite default, and never raises. The DB auto-creates/migrates on first span; DSAGT emits only traces, so no artifact dir materializes. View traces with `mlflow ui --backend-store-uri sqlite:///<pdir>/mlflow.db`. DSAGT writes **no** telemetry env on the agent (no forced OTel) — agent LLM-call history is recovered post-hoc from the on-disk transcript (Phase 2's pipeline), not by intercepting traffic.

## Commands

```bash
uv sync --all-groups                                                  # install
uv run --no-sync python -m pytest tests/test_<file>.py -q             # targeted tests
uv run black .                                                         # format
uv run ruff check .                                                    # lint
```

**`python -m pytest` not bare `pytest`.** The bare-binary form picks up the wrong pytest on this machine and crashes with `ModuleNotFoundError: dsagt`.

**Don't run the full suite by default.** ~50s for ~590 tests is too slow for an iteration loop. Run only the test file relevant to the change. `tests/test_config.py` covers session, init, agents, BYOA hints, the serverless store resolution, and the no-launch-shim/no-OTel invariants. Skip `test_integration.py`, `test_*_integration.py`, `test_server_startup.py`, `test_dependency_integration.py` unless explicitly relevant — they hit network or spawn subprocesses.

## Code Organization

The codebase separates **commands** (entry points with argparse, launched as CLI tools or subprocesses) from **modules** (importable logic). Commands live in `src/dsagt/commands/`, modules live in `src/dsagt/`.

**Commands** (`src/dsagt/commands/`):
- `cli.py` — `dsagt init / start / info / list / mv / rm / smoke-test` (user-facing CLI). `dsagt start` launches the agent in the foreground and runs post-session extraction on exit.
- `run_tool.py` — `dsagt-run` (tool execution wrapper).
- `setup_core_kb.py` — KB-asset build engine (`resolve_assets` / `ensure_assets`), called by `dsagt init` to provision the shared KB; not a CLI command of its own.
- `info.py` — `dsagt info` (project / config introspection + trace triage).

(The MCP server — `dsagt-server` — lives in the `src/dsagt/mcp/` package, see below.)

**Modules** (`src/dsagt/`):
- `session.py` — Project init, agent config generation, env-var resolution, config load/validate, session-id minting (`new_session_id`), end-of-session extraction orchestration (`run_extraction`). No service supervision — the store is serverless.
- `agents/` — Per-agent-platform setup (`base.py` ABC + `claude.py` / `goose.py` / `cline.py` / `roo.py` / `codex.py` / `opencode.py`). Each subclass owns its `write_static`, `write_dynamic`, `runtime_env` (per-project state dirs only), `byoa_env_hints`, `vscode_hint`. Shared helpers (`_mcp_env_block`, `_build_mcp_servers_dict`) in `base.py`. DSAGT sets no telemetry/OTel env and writes no launch shim.
- `knowledge.py` — ChromaDB document retrieval, embedding backends, per-collection routing (FAISS removed).
- `registry.py` — `ToolRegistry` (CLI tools) + `SkillRegistry` (agent instruction skills), KB indexing.
- `provenance.py` — Tool execution records (`run_and_record`, `ToolRecordStore`), execution-record indexing into ChromaDB, pipeline reconstruction (`reconstruct_pipeline`, dependency graph).
- `observability.py` — MLflow tracing setup over the serverless file store; `resolve_tracking_uri` (never-raise), `init_tracing`, span helpers.
- `memory.py` — Explicit memory (YAML, `ExplicitMemory`) + the episodic `MemoryExtractor` (a `traces.TraceCollector` consumer: Tier-0 mechanical chunk+tag+embed always, Tier-1 `judge.Judge` distillation when enabled, falling back to Tier-0 on judge failure) + outlier-suggestion queue (`SuggestionQueue`). Facts carry `ts_epoch` for recency-weighted retrieval. `extract_session` remains a no-op stub kept only for the deferred cross-session N+1 catch-up call site — the heartbeat consumer is the live episodic path. The LLM judge lives in `judge.py`.
- `skills.py` — External skill catalog data plane (`SkillsCatalog`: clone/sync/index/install), the `SkillRouter` render facade, and the Genesis-derived keyword scorer (`rank_skills`).
- `traces.py` — the whole trace pipeline in one module: the pure-data `Trace` (a list of span dicts + compose/query/`to_exchanges` methods; the span/message/block shapes have one home, `Trace`'s `add_*` methods), the `Reader`/`Translator` ABCs with a per-agent subclass each (Claude bespoke; codex/goose/opencode/cline share the `Translator` turn-template; claude+codex share `JsonlReader`), and `TraceCollector` — the MCP-server heartbeat that reads→translates→hands the `Trace` to its consumers (the MLflow logger, the memory distiller), each with its own ack set for idempotency. Imports nothing heavy (mlflow is lazy, consumer-side).
- `judge.py` — the episodic Tier-1 `Judge` (`Judge.create` → `LocalJudge` GGUF default / `APIJudge` stub) + the lean per-turn distill prompt/parser. Local-by-default (no API key); `distill` is grammar-constrained on `LocalJudge`, a no-op on `APIJudge`.

**MCP server** (`src/dsagt/mcp/`) — the single merged `dsagt-server`. `server.py` owns `main()`, the shared-KB startup (`_build_kb_from_config`), and the dispatch shell (`build_dispatch_server`); the tool surface is split by concern across `registry_tools.py` (tool registry + execution + provenance, 8 tools), `knowledge_tools.py` (KB retrieval, 6), `memory_tools.py` (explicit memory + suggestions, 4), and `skill_tools.py` (skill search/install/sources, 5). Each `*_tools.py` exposes a `_*_tools_and_handlers()` factory (composed by `create_dsagt_server`) plus a `create_*_server` test wrapper.

Entry points (`pyproject.toml` `[project.scripts]`): `dsagt` → `dsagt.commands.cli:main`, `dsagt-run` → `dsagt.commands.run_tool:main`, `dsagt-server` → `dsagt.mcp.server:main`.

**Bundled assets** (shipped as `package-data` in `pyproject.toml`):
- `src/dsagt/tools/` — built-in tool specs (markdown + YAML frontmatter) copied into new projects.
- `src/dsagt/skills/` — built-in skills (e.g., `skill-creator`) the agent discovers via `search_skills`.
- `src/dsagt/dsagt_instructions.md` — agent-platform-agnostic system instructions injected into per-agent files at init.

**`use_cases/`** holds end-to-end domain walkthroughs (`microbial_isolates/`, `cryoem/`, `isaac_vasp/`). They are reference material for users, not part of the test suite.

## Code style & conventions

Distilled from working on this codebase; `knowledge.py` is the reference example of the house style.

**No defensive swallowing.** Don't add guards that silently absorb empty/invalid input (`if not texts: return []`, empty-array short-circuits, disk-state "reconciliation" of can't-happen states). They convert a caller's bug into a silent success you'll never see. Empty/invalid input is out-of-contract — let it surface. Translating a *real, reachable* exception into an actionable message (e.g. a dim-mismatch hint) is different and welcome; swallowing is not.

**YAGNI / no speculative generality.** Don't add a flag or option for a path never exercised in practice. Model real variation *structurally* (a subclass / distinct type), not with a runtime toggle nothing flips — e.g. the local store is unconditionally hybrid; "dense-only" is a future store *type*, not a `hybrid=False` flag. Don't extract a base class until a second concrete impl forces the seam (you can't validate the abstraction with one impl). This is dev-stage: gut cleanly, no back-compat shims, aim net-minus LOC.

**Explicit named arguments, never `**kwargs` config-splat.** A dict of kwargs threaded through layers and `**`-splatted into a constructor hides what's actually passed. Use explicit named params; unpack any config dict at the boundary (e.g. `Embedder.create(backend, *, model=, base_url=, ...)`, callers pass `model=cfg.get("model")`).

**Put behavior where it belongs.** Factories live on the class they build (`Embedder.create` classmethod, not a free `_make_embedder`). Trivial field getters are unpythonic — expose the attribute; but a *method* is right when access does real work (lazy I/O + memoization like `_get_bm25` — the name signals cost a bare subscript would hide). Pure, stateless algorithms shared by multiple classes stay module-level functions (RRF: `_rrf_merge`/`_rrf_across`), not staticmethods nailed to one arbitrary owner.

**Comments state the real reason, at the point they explain it.** A lazy import is justified *at the import site* with its actual cause, not as an "this is absent" note in the import block citing a stale rationale. If the reason changes, fix the comment.

**Import hygiene on hot paths.** Modules on frequently-invoked paths (`dsagt-run` runs per tool call) must not transitively drag heavy modules in for *annotation-only* type hints. Use `from __future__ import annotations` + a `TYPE_CHECKING`-guarded import (verify the module doesn't introspect annotations at runtime first). Keep cold start lean; lazy-import the heavy leaf (llama_index) at its single use site.

**Naming.** Prefer concise domain names (`APIEmbedder`/`LocalEmbedder`, not `…EmbeddingClient`).

**Module docstrings (major modules).** Open with a title line + 3–5 sentences: what the module does, the capabilities it backs, and the design motivations. Follow it with an **ASCII-art UML class map** — one consistent notation throughout (`knowledge.py` uses `◇` holds · `◆` owns · `▷` inherits, every edge drawn the same way). Treat the class-map diagram as a deliverable of any **major module refactor** — add or refresh it whenever the class structure changes substantially.

## BYOA artifacts

`dsagt init --agent X --location <path>` writes (in the project dir):
- `dsagt_config.yaml` — internal config (project name, agent, embedding/knowledge/extraction/skills settings). No mlflow port (the store is the serverless `sqlite:///<pdir>/mlflow.db`), no user-facing fields, no credentials.
- Per-agent instructions file (e.g., `CLAUDE.md`, `.goosehints`, `AGENTS.md`).
- Per-agent MCP config artifact (`.mcp.json` for claude, `goose.yaml` for goose, `cline_mcp_settings.json` via `cline mcp add`, `.roo/mcp.json`, `.codex-data/config.toml`). The env block carries benign routing only (DSAGT_PROJECT, DSAGT_PROJECT_DIR, DSAGT_SESSION_ID, MLFLOW_TRACKING_URI, EMBEDDING_*) so MCP-server children of agents that don't inherit shell env (codex/cline/roo) still log to the right store. No credentials, no OTel routing, no privacy overrides.
- For claude, `.claude/settings.json` wiring the `mlflow autolog claude` Stop hook (transcript → file store at session end).

No launch shim is written and `dsagt init` prints no env/OTel instructions — the user starts the agent directly or via `dsagt start`.

## Architecture

### MCP Server

A single merged `dsagt-server` (`src/dsagt/mcp/`) exposes 23 tools across four concern modules under one `Server` + one shared `KnowledgeBase`:

1. **Registry tools** (`mcp/registry_tools.py` + `registry.py` / `provenance.py`) — tool analysis, registration, dependency installation, command/file/http execution, and pipeline reconstruction. Tools are saved as markdown specs with YAML frontmatter.
2. **Knowledge tools** (`mcp/knowledge_tools.py` + `knowledge.py`) — semantic search over document collections (ChromaDB, optional cross-encoder reranking); long ops run as background jobs.
3. **Memory tools** (`mcp/memory_tools.py` + `memory.py`) — explicit memory + outlier suggestions (`kb_remember` / `kb_get_memories` / …).
4. **Skill tools** (`mcp/skill_tools.py` + `skills.py`) — skill search / install + external catalog sources.

### Observability

- **Serverless MLflow store** — Spans land in `sqlite:///<pdir>/mlflow.db` (no server). View with `mlflow ui --backend-store-uri sqlite:///<pdir>/mlflow.db`. The tracking URI resolves via `observability.resolve_tracking_uri` (env → config → sqlite default; never raises).
- **dsagt-run** (`commands/run_tool.py` + `provenance.py`) — Wraps tool commands; captures execution layer (command, stdout/stderr, timing, file lists) into `trace_archive/`, and emits `tool.execute` spans.
- **MCP-server spans** — `dsagt-server` calls `init_tracing()` at startup; its tool spans (kb.*, registry.*) flow to the file store. Session correlation via `DSAGT_SESSION_ID` (minted by `dsagt start`).
- **Agent traces** — recovered post-hoc from the on-disk transcript, not native OTel. Today only claude's `mlflow autolog` Stop hook populates agent traces; the general transcript pipeline is Phase 2.

### Memory System

- **Explicit memory** (`memory.py:ExplicitMemory`) — User-confirmed facts in YAML, loaded into agent context at session start via the `kb_remember` / `kb_get_memories` MCP tools (the vector mirror is optional — degrades to pure-YAML if the store is down).
- **Tool-execution indexing** — `provenance.ToolUseIndexer` embeds `trace_archive/` records into the project's `tool_use` collection incrementally on the MCP-server heartbeat (idempotent via a persisted ack set), plus a startup catch-up and an on-demand tick before `reconstruct_pipeline`. No LLM, no credentials.
- **Episodic memory** — live, **opt-in** (`episodic.enabled`, via `dsagt init --episodic`). The `memory.MemoryExtractor` consumer consumes `Trace.to_exchanges()` on the heartbeat: Tier-0 mechanical (chunk+tag+embed, no LLM) always; Tier-1 distillation via `judge.LocalJudge` (Qwen2.5-1.5B GGUF, GBNF-grammar JSON) when a judge backend is configured, degrading to Tier-0 on failure. Retrieval is recency-weighted (`episodic.recency_half_life_days`). `extract_session` stays a no-op stub for the deferred N+1 catch-up only. There is no user-facing `dsagt memory` command.

### Key Design Patterns

- **Agent-agnostic**: DSAGT is infrastructure, not an agent. Capabilities are MCP services.
- **Session isolation**: Each project gets its own directory with config, tools, skills, kb_index, trace_archive, and the `mlflow.db` sqlite trace store.
- **Tools vs Skills**: Tools are CLI executables in `<project>/tools/` (specs with parameters, wrapped by dsagt-run). Skills are agent instruction workflows in `<project>/skills/` (SKILL.md + reference docs). Both are discoverable via ChromaDB-backed semantic search.

## DSAGT Pipeline Builder Workflow

When acting as a pipeline builder (using the MCP server), follow these constraints:

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
