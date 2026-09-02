# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

DSAGT (DataSmith Agent) is an AI-assisted data pipeline builder. It exposes an MCP (Model Context Protocol) server to agent platforms (Claude Code, Goose, Cline, Codex, opencode) that helps domain scientists build reproducible, auditable data-curation pipelines through iterative, knowledge-driven code generation.

## Run model (BYOA, serverless)

DSAGT is **bring-your-own-agent**: the agent talks to its own LLM provider directly — DSAGT never proxies that traffic. `dsagt init --agent <name>` writes a per-agent instructions file plus MCP config; nothing to paste, no launch shim, no OTel/env wiring. Two behaviorally-identical start flows:

1. **Bare launch** — start the agent from the project dir (`cd <project> && claude` / `goose` / …).
2. **`dsagt start <project>`** — spawns the agent in the foreground and, on exit, runs post-session `run_extraction`. Its real job is owning a reliable session-end trigger.

The MCP server is self-sufficient: it derives the project from its cwd (`.dsagt/config.yaml`), with the MCP-config env block as robustness, and behaves identically regardless of how it was launched.

**Serverless store.** All self-logging goes to a `sqlite:///<pdir>/mlflow.db` MLflow store — no server to run (SQLite is MLflow's supported serverless backend). `observability.resolve_tracking_uri` always computes that path from the project dir (no env/config override) and never raises. The DB auto-creates on first span; DSAGT emits only traces, so no artifact dir materializes. View with `mlflow ui --backend-store-uri sqlite:///<pdir>/mlflow.db`. Agent LLM-call history is recovered post-hoc from the on-disk transcript (the trace pipeline), not by intercepting traffic.

## Commands

```bash
uv sync --all-groups                                                  # install
uv run --no-sync python -m pytest tests/test_<file>.py -q             # targeted tests
uv run black .                                                         # format
uv run ruff check .                                                    # lint
```

**`python -m pytest`, not bare `pytest`.** The bare binary picks up the wrong pytest on this machine and crashes with `ModuleNotFoundError: dsagt`.

**Don't run the full suite by default** — ~50s for ~590 tests is too slow for an iteration loop. Run only the test file relevant to the change. `tests/test_config.py` covers session, init, agents, serverless-store resolution, and the no-launch-shim/no-OTel invariants. Skip `test_integration.py`, `test_*_integration.py`, `test_server_startup.py`, `test_dependency_integration.py` unless relevant — they hit the network or spawn subprocesses.

## Code Organization

The codebase separates **commands** (entry points with argparse, launched as CLI tools or subprocesses) from **modules** (importable logic). Commands live in `src/dsagt/commands/`, modules in `src/dsagt/`.

**Commands** (`src/dsagt/commands/`):
- `cli.py` — `dsagt init / start / info / traces / list / mv / rm / smoke-test`. `dsagt start` launches the agent in the foreground and runs post-session extraction on exit. `dsagt traces <project>` opens the MLflow viewer over the project's store (runs catch-up first, deep-links to the Traces tab, quiets the mlflow noise; foreground, no managed daemon).
- `run_code.py` — `dsagt-run` (code execution wrapper).
- `setup_core_kb.py` — KB-asset build engine (`resolve_assets` / `ensure_assets`), called by `dsagt init` to provision the shared KB; not a CLI command of its own.
- `info.py` — `dsagt info` (project/config introspection + trace triage).

(The MCP server — `dsagt-server` — lives in the `src/dsagt/mcp/` package, see below.)

**Modules** (`src/dsagt/`):
- `session.py` — Project init, agent config generation, env-var resolution, config load/validate, session-id minting (`append_session` / `session_tag`), and startup **catch-up** (`catch_up_extraction`): code-use indexing + a chat-trace re-collect (`_catch_up_traces`) of the *previous* session (pinned to the `trace_source` token in `state.yaml`) so turns lost to an ungraceful shutdown still reach MLflow + episodic memory — uniform across all agents.
- `agents/` — Per-agent-platform setup (`base.py` ABC + `claude.py` / `goose.py` / `cline.py` / `codex.py` / `opencode.py`). Each subclass owns its `write_static`, `write_dynamic`, `runtime_env`, `vscode_hint`. Shared helpers (`_mcp_env_block`, `_build_mcp_servers_dict`) in `base.py`. DSAGT sets no telemetry/OTel env, writes no launch shim, and never touches provider credentials — agents are expected pre-authenticated (shell env / their own auth flows) before dsagt is pointed at them; dsagt prints no credential hints and never troubleshoots auth.
- `knowledge.py` — ChromaDB document retrieval, embedding backends, per-collection routing (the reference example of the house style).
- `registry.py` — `CodeRegistry` (CLI codes) + `SkillRegistry` (agent instruction skills), KB indexing.
- `provenance.py` — Code execution records (`run_and_record`), execution-record indexing into ChromaDB (`CodeUseIndexer` → `code_use` collection), pipeline reconstruction (`reconstruct_pipeline`, dependency graph, `compute_pipeline_fingerprint`).
- `observability.py` — first-party span emission over the serverless sqlite store via MLflow's native `mlflow.start_span` (no OTel `TracerProvider`). `resolve_tracking_uri` (never-raise), `init_tracing`, `@traced`/`obs`/`child_span` + typed span helpers. Each internal trace's root is tagged `dsagt.source` with the MCP tool *category* (`memory`/`skill`/`knowledge`/`registry`, or `execution` for dsagt-run) so the MLflow UI can filter the debug view apart from agent traces. `MLflowSink` (a `traces.Trace` consumer) replays finished transcripts via `start_span_no_context`.
- `memory.py` — Explicit memory (YAML, `ExplicitMemory`) + the episodic `MemoryExtractor` (a `traces.TraceCollector` consumer that mechanically chunks+tags+embeds every turn, no LLM). Turns carry `ts_epoch` for recency-weighted retrieval. `extract_session` is a no-op stub kept only for the deferred cross-session N+1 catch-up call site.
- `skills.py` — External skill-catalog data plane (`SkillsCatalog`: clone/sync/index/install), the `SkillRouter` render facade, and the Genesis-derived keyword scorer (`rank_skills`).
- `traces.py` — the whole trace pipeline in one module: the pure-data `Trace` (span dicts + compose/query/`to_exchanges`), the `Reader`/`Translator` ABCs with a per-agent subclass each (Claude bespoke; codex/goose/opencode/cline share the `Translator` turn-template; claude+codex share `JsonlReader`), and `TraceCollector` — the MCP-server heartbeat that reads→translates→hands the `Trace` to its consumers (MLflow logger, memory indexer), each with its own ack set for idempotency. Imports nothing heavy (mlflow is lazy, consumer-side).

**MCP server** (`src/dsagt/mcp/`) — the single merged `dsagt-server`. `server.py` owns `main()`, the shared-KB startup (`_build_kb_from_config`), and the dispatch shell (`build_dispatch_server`). The 20-tool surface is split by concern: `registry_tools.py` (code registry + execution + provenance, 8), `knowledge_tools.py` (KB retrieval, 5), `memory_tools.py` (explicit memory, 2), `skill_tools.py` (skill search/install/sources, 5). Each `*_tools.py` exposes a `_*_tools_and_handlers()` factory (composed by `create_dsagt_server`) plus a `create_*_server` test wrapper.

Entry points (`pyproject.toml` `[project.scripts]`): `dsagt` → `dsagt.commands.cli:main`, `dsagt-run` → `dsagt.commands.run_code:main`, `dsagt-server` → `dsagt.mcp.server:main`.

**Built-in assets** (declared as `package-data`):
- `src/dsagt/codes/` — built-in codes as skill-standard dirs (`<name>/SKILL.md`), served from the package (never copied into projects).
- `src/dsagt/skills/` — built-in skills (e.g., `skill-creator`) the agent discovers via `search_skills`.
- `src/dsagt/dsagt_instructions.md` — agent-agnostic system instructions injected into per-agent files at init.

**`use_cases/`** holds end-to-end domain walkthroughs. They are reference material for users, not part of the test suite.

## Code style & conventions

Distilled from working on this codebase; `knowledge.py` is the reference example of the house style.

**No defensive swallowing.** Don't add guards that silently absorb empty/invalid input (`if not texts: return []`, empty-array short-circuits, disk-state "reconciliation" of can't-happen states). They convert a caller's bug into a silent success you'll never see. Empty/invalid input is out-of-contract — let it surface. Translating a *real, reachable* exception into an actionable message (e.g. a dim-mismatch hint) is different and welcome; swallowing is not.

**YAGNI / no speculative generality.** Don't add a flag or option for a path never exercised in practice. Model real variation *structurally* (a subclass / distinct type), not with a runtime toggle nothing flips — e.g. the local store is unconditionally hybrid; "dense-only" is a future store *type*, not a `hybrid=False` flag. Don't extract a base class until a second concrete impl forces the seam. This is dev-stage: gut cleanly, no back-compat shims, aim net-minus LOC.

**Explicit named arguments, never `**kwargs` config-splat.** A dict of kwargs threaded through layers and `**`-splatted into a constructor hides what's actually passed. Use explicit named params; unpack any config dict at the boundary (e.g. `Embedder.create(backend, *, model=, base_url=, ...)`, callers pass `model=cfg.get("model")`).

**Put behavior where it belongs.** Factories live on the class they build (`Embedder.create` classmethod, not a free `_make_embedder`). Trivial field getters are unpythonic — expose the attribute; but a *method* is right when access does real work (lazy I/O + memoization like `_get_bm25`). Pure, stateless algorithms shared by multiple classes stay module-level functions (RRF: `_rrf_merge`/`_rrf_across`), not staticmethods nailed to one arbitrary owner.

**Comments state the real reason, at the point they explain it.** A lazy import is justified *at the import site* with its actual cause, not as an "this is absent" note in the import block citing a stale rationale. If the reason changes, fix the comment.

**No change-narration in comments.** A comment describes what the code does and why it exists *now* — never how it used to work, what changed, or paradigms no longer in the tree. Ban breadcrumbs like "previously…", "was formerly…", "no longer uses…", "moved from…", "(not a `.get` default)", "instead of the old…". Git carries the history; the comment describes the present. State the hazard/intent directly ("session_id is null outside a minted session, and ChromaDB rejects null metadata") rather than contrasting with a prior version ("… beats coercing to `[]` like before").

**Import hygiene on hot paths.** Modules on frequently-invoked paths (`dsagt-run` runs per tool call) must not transitively drag heavy modules in for *annotation-only* type hints. Use `from __future__ import annotations` + a `TYPE_CHECKING`-guarded import (verify the module doesn't introspect annotations at runtime first). Keep cold start lean; lazy-import the heavy leaf (llama_index) at its single use site.

**Naming.** Prefer concise domain names (`APIEmbedder`/`LocalEmbedder`, not `…EmbeddingClient`).

**Module docstrings (major modules).** Open with a title line + 3–5 sentences: what the module does, the capabilities it backs, the design motivations. Follow with an **ASCII-art UML class map** — one consistent notation throughout (`knowledge.py` uses `◇` holds · `◆` owns · `▷` inherits). Treat the class-map diagram as a deliverable of any **major module refactor** — refresh it whenever the class structure changes substantially.

**Prose register (docs, comments, changelog, commit messages).** Plain, accurate, direct — no anthropomorphism, no code-jockey slang, no advertising gloss. Concretely: files/modules are *located in* / *defined in* / *stored in*, never "live in"; DSAgt *provides* / *includes* things, it does not "ship" or "provision" them; use *built-in*, not "bundled"; drop marketing gloss ("out of the box", "seamless", "with nothing to remember", "blazing"). State what a thing does, not how nice it is. Changelogs and commit messages record real behavior changes — pure renames and doc-only churn are noise, keep them out. This applies to this file too.

## BYOA artifacts

`dsagt init --agent X --location <path>` writes, in the project dir:
- `.dsagt/config.yaml` — internal config (project name, agent, embedding/knowledge/extraction/skills settings). No mlflow port (the store is the serverless `sqlite:///<pdir>/mlflow.db`), no user-facing fields, no credentials. `.dsagt/state.yaml` (session log + memory cursor) and `.dsagt/explicit_memories.yaml` live alongside it, owned by the MCP server.
- Per-agent instructions file (e.g., `CLAUDE.md`, `.goosehints`, `AGENTS.md`).
- Per-agent MCP config artifact (`.mcp.json` for claude, `goose.yaml` for goose, `cline_mcp_settings.json` via `cline mcp add`, `.codex-data/config.toml`). The env block carries benign routing only (`DSAGT_PROJECT`, `DSAGT_PROJECT_DIR`, `DSAGT_SESSION_ID`, `MLFLOW_TRACKING_URI`, `EMBEDDING_*`) so MCP-server children of agents that don't inherit shell env (codex/cline) still log to the right store. No credentials, no OTel routing.

No launch shim is written and `dsagt init` prints no env/OTel instructions — the user starts the agent directly or via `dsagt start`. DSAGT wires no MLflow autolog hook: Claude's traces (like every agent's) come from the heartbeat pipeline, not native autolog.

## Architecture

### MCP Server

A single merged `dsagt-server` (`src/dsagt/mcp/`) exposes 20 tools across four concern modules under one `Server` + one shared `KnowledgeBase`:

1. **Registry tools** (`mcp/registry_tools.py` + `registry.py` / `provenance.py`) — tool analysis, registration, dependency installation, command/file/http execution, pipeline reconstruction. Tools are saved as markdown specs with YAML frontmatter.
2. **Knowledge tools** (`mcp/knowledge_tools.py` + `knowledge.py`) — semantic search over document collections (ChromaDB, optional cross-encoder reranking); long ops run as background jobs.
3. **Memory tools** (`mcp/memory_tools.py` + `memory.py`) — explicit memory (`kb_remember` / `kb_get_memories`).
4. **Skill tools** (`mcp/skill_tools.py` + `skills.py`) — skill search/install + external catalog sources.

### Observability

- **Serverless MLflow store** — spans land in `sqlite:///<pdir>/mlflow.db` (no server). The tracking URI resolves via `observability.resolve_tracking_uri` (never raises).
- **dsagt-run** (`commands/run_code.py` + `provenance.py`) — wraps code commands; captures the execution layer (command, stdout/stderr, timing, file lists) into `trace_archive/` and emits `code.execute` spans.
- **MCP-server + tool spans (debug view)** — `dsagt-server` calls `init_tracing()` at startup; the dispatch shell opens one categorization-root span per tool call (subsystem `kb.*`/`registry.*` spans nest under it). Each root is tagged `dsagt.source` with its concern category so it filters apart as a debugging view. Session grouping via `DSAGT_SESSION_ID`.
- **Agent traces** — recovered post-hoc from the on-disk transcript by the MCP-server heartbeat's trace pipeline (`traces.py`), uniform across all five agents. No native OTel, no autolog.

### Memory System

- **Explicit memory** (`memory.py:ExplicitMemory`) — user-confirmed facts in YAML, loaded into agent context at session start via `kb_remember` / `kb_get_memories` (the vector mirror is optional — degrades to pure-YAML if the store is down).
- **Code-execution indexing** — `provenance.CodeUseIndexer` embeds `trace_archive/` records into the project's `code_use` collection incrementally on the heartbeat (idempotent via a persisted ack set), plus a startup catch-up and an on-demand tick before `reconstruct_pipeline`. No LLM.
- **Chat-trace catch-up** — the heartbeat logs the live transcript to MLflow (+ episodic memory) and a graceful shutdown flushes the deferred final turn; an ungraceful kill is backstopped at the *next* session's startup by `session._catch_up_traces`, which re-collects the previous session pinned to its recorded `trace_source` token. Idempotency rests on the collector's **session-qualified** ack keys (`<session_id>:<span_id>`).
- **Episodic memory** — live, **opt-in** (`episodic.enabled`, via `dsagt init --episodic`). The `memory.MemoryExtractor` consumer consumes `Trace.to_exchanges()` on the heartbeat and mechanically chunks+tags+embeds every turn into `session_memory` (no LLM). Retrieval is recency-weighted (`episodic.recency_half_life_days`).

### Key Design Patterns

- **Agent-agnostic**: DSAGT is infrastructure, not an agent. Capabilities are MCP services.
- **Session isolation**: each project gets its own directory with config, tools, skills, kb_index, trace_archive, and the `mlflow.db` sqlite store.
- **Codes vs Skills**: Codes are CLI executables in `<project>/codes/<name>/` (skill-standard dirs whose SKILL.md frontmatter adds executable/parameters; wrapped by dsagt-run). Skills are agent instruction workflows in `<project>/skills/` (SKILL.md + reference docs). Both share the skill envelope, so both mirror into the agent's native skills dir; both are also discoverable via ChromaDB-backed semantic search (`search_registry` / `search_skills`).

## DSAGT Pipeline Builder Workflow

When acting as a pipeline builder (using the MCP server), follow these constraints:

1. **Never directly access data** — all data operations go through registered codes.
2. **Code preference hierarchy**: registered code → KB package code → custom implementation.
3. **Generate paired tools** — every data operation gets a check tool (pre/post audit) and an operation tool.
4. **Audit everything** — before/after JSON reports saved to `audit/`.
5. **One step at a time** — iterate with the user, confirming approach before execution.

## Testing Patterns

- pytest with `subprocess.run` mocking for command execution.
- MCP server tests invoke handlers directly (no stdio transport); async tests for server handlers.
- Temp directories for isolation; the `_use_tmp_registry` fixture in `tests/test_config.py` patches `DEFAULT_PROJECTS_BASE` and the project registry to `tmp_path`.
- Integration tests in `test_*_integration.py` require real `EMBEDDING_*` / `LLM_*` credentials.
