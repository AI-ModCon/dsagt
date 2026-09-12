# DSAgt design

DSAgt is an MCP server plus a CLI that give an agent platform (Claude Code, Goose, Codex, opencode, Cline) code registration, a knowledge base, skill discovery, execution provenance, memory, and trace logging for a data-curation project. The agent talks to its own LLM provider; DSAgt reads the agent's on-disk transcript for traces and never proxies that traffic.

## Run model

- `dsagt init --agent <name>` writes `.dsagt/config.yaml`, the per-agent instructions file (`CLAUDE.md`, `.goosehints`, `AGENTS.md`, or `.clinerules/dsagt_instructions.md`, with `src/dsagt/dsagt_instructions.md` injected), and the per-agent MCP config (`.mcp.json`, `goose.yaml`, `.codex-data/config.toml`, `opencode.json`, or `cline mcp add`). The MCP-config env block carries routing only: `DSAGT_PROJECT`, `DSAGT_PROJECT_DIR`, `DSAGT_AGENT`, `MLFLOW_TRACKING_URI`, `EMBEDDING_*`.
- The agent starts bare from the project directory or through `dsagt start <project>`, which runs the agent in the foreground. Both flows behave the same because `dsagt-server` derives the project from its cwd and runs `session.catch_up_extraction` itself at startup.
- Self-logging goes to the serverless store `sqlite:///<pdir>/mlflow.db`. `observability.resolve_tracking_uri` computes that path from the project directory and never raises. `dsagt traces <project>` opens the MLflow viewer over it after a catch-up.
- Projects are registered in `~/dsagt-projects/projects.yaml`. `.dsagt/state.yaml` holds the session log, the memory cursor, and the previous session's trace-source token; `.dsagt/explicit_memories.yaml` holds explicit memory. The MCP server owns both files.

## Packages (`src/dsagt/`)

Commands are entry points with argparse; modules are importable logic.

| Path | Holds |
|---|---|
| `commands/cli.py` | `dsagt init / start / info / traces / list / mv / rm / smoke-test` |
| `commands/run_code.py` | `dsagt-run`, the execution wrapper |
| `commands/info.py` | `dsagt info`: config introspection and trace triage |
| `commands/traces.py` | `dsagt traces`: catch-up, then the MLflow viewer deep-linked to the Traces tab |
| `commands/setup_core_kb.py` | the KB asset build (`resolve_assets`, `ensure_assets`) that `dsagt init` calls |
| `session.py` | project init, config load and validation, session minting (`append_session`, `session_tag`), startup catch-up (`catch_up_extraction`) |
| `agents/` | one `AgentSetup` subclass per platform, each owning `write_static`, `write_dynamic`, `runtime_env`, `vscode_hint`; shared helpers in `base.py` |
| `knowledge.py` | `KnowledgeBase` over ChromaDB: hybrid dense + BM25 retrieval, embedding backends, per-collection routing; the reference module for house style |
| `registry.py` | `CodeRegistry` (codes) and `SkillRegistry` (installed skills) |
| `provenance.py` | `run_and_record`, `CodeUseIndexer` (execution records into `code_use`), `reconstruct_pipeline` |
| `observability.py` | the live tracer (`init_tracing`, `traced`, `obs`, `child_span`, and typed span helpers over `mlflow.start_span`) and `MLflowSink`, the trace consumer that replays a finished transcript with its original timestamps over `mlflow.start_span_no_context` |
| `memory.py` | `ExplicitMemory` (YAML with a vector mirror) and `MemoryExtractor` (the episodic trace consumer) |
| `readiness.py` | the opt-in AIDRIN gate: `ensure_aidrin`, the metric profiles, `instructions_block` |
| `skills.py` | `SkillsCatalog` (clone, sync, index, install), `SkillRouter`, `rank_skills`, `BASE_SKILLS` and `install_base_skills` |
| `traces.py` | `Trace`, one `Reader` and one `Translator` per agent, `TraceCollector` and `make_trace_collector`; imports nothing heavy at module scope |
| `mcp/` | `server.py` (`main`, shared-KB startup, `build_dispatch_server`) and one `*_tools.py` per concern |
| `codes/` | built-in codes as skill-standard dirs, indexed by `dsagt init` |
| `dsagt_instructions.md` | agent-agnostic instructions injected into the per-agent instructions file |

## MCP server

- `dsagt-server` is one process with one `KnowledgeBase`, one embedder, and one Chroma owner. It opens only `<project>/kb_index`; `dsagt init` provisions collections.
- Twenty tools in four concern modules: registry (`search_registry`, `get_registry`, `save_code_spec`, `install_dependencies`, `run_command`, `read_file`, `http_request`, `reconstruct_pipeline`), knowledge (`kb_search`, `kb_ingest`, `kb_append`, `kb_list_collections`, `kb_job_status`), memory (`kb_remember`, `kb_get_memories`), skills (`search_skills`, `install_skill`, `save_skill`, `add_skill_source`, `list_skill_sources`). Each `*_tools.py` exposes a `_*_tools_and_handlers()` factory composed by `create_dsagt_server`, plus a `create_*_server` test wrapper.
- The dispatch shell validates arguments against each tool's input schema, opens one root span per call tagged `dsagt.source` with the concern category (`memory`, `skill`, `knowledge`, `registry`; `execution` for `dsagt-run`), and runs blocking work (`run_command`, dependency installs, registry search, reconstruction, ingest) in worker threads.
- The heartbeat runs every 45 seconds in a worker thread: the trace collector, then the code-use indexer. A graceful shutdown flushes the deferred final turn.

## Trace pipeline

- Stages, all in `traces.py`: a `Reader` locates and reads the platform's session files into raw records; the matching `Translator` maps them to one `Trace`; `TraceCollector` hands completed turns to its consumers (`MLflowSink`, `MemoryExtractor`). Each consumer keeps its own ack set keyed `<session_id>:<span_id>`, so a re-pass or a catch-up never double-logs a turn. `ack_dir` defaults to `.dsagt`.
- A periodic pass emits only completed turns; the open last turn is the deferred final turn, flushed when a later prompt bounds it or at end of session.
- `MLflowSink` writes backdated spans through `start_span_no_context`, stamping `dsagt.agent` and `dsagt.trace_id`, and carries cache token counts in the token-usage attribute. mlflow is imported inside `write`.
- At startup `session.catch_up_extraction` indexes execution records and re-collects the previous session, pinned to the `trace_source` token in `state.yaml`, so turns lost to an ungraceful shutdown reach the store.

## Memory

- Explicit memory: `kb_remember` and `kb_get_memories` over YAML plus a vector mirror; superseded entries go to `explicit_memories_history.yaml`. The tools work from the YAML alone when the vector store is unavailable.
- Episodic memory (opt-in, `episodic.enabled`): `MemoryExtractor` chunks, tags, and embeds every completed turn into `session_memory`; retrieval is recency-weighted by `episodic.recency_half_life_days`.
- Code-use memory: `CodeUseIndexer` embeds `trace_archive/` records into `code_use` on the heartbeat, at startup, and before `reconstruct_pipeline`.

## Codes and skills

- A code is a CLI executable at `<project>/codes/<name>/SKILL.md` whose frontmatter adds `executable` and `parameters`. The executable string carries `dsagt-run --code <name> --`, so provenance rides in the command the agent runs, through the MCP tool or its own shell.
- A skill is an instruction workflow at `<project>/skills/<name>/`. Codes and skills share the skill envelope and mirror into the agent's native skills dir (`.claude/skills`, `.agents/skills`, `.cline/skills`) at install and at `dsagt init` / `start`.
- The package holds no skills. `BASE_SKILLS` (`skill-creator` from the genesis catalog's `skills/basedata-skills/`, `aidrin` from idtlab/AIDRIN) are re-cloned and installed at every init. The corpus (`KNOWN_SOURCES` plus any git URL) is cloned and indexed one collection per source; `search_skills` browses it and `install_skill` copies a skill in with its upstream license files and a `PROVENANCE.txt`.
- The readiness gate (`dsagt init --readiness aidrin`) installs AIDRIN once under `~/dsagt-projects/.tools/aidrin/` and appends `instructions_block` to the agent's instructions; every `aidrin` command runs through `dsagt-run --code aidrin`.

## Packaging

- Every dependency is a range with a next-major cap, so dsagt resolves beside a project that pins differently; `uv.lock` is the reproducible environment.
- Another application runs the trace pipeline through `make_trace_collector`, pointing the readers at its own session directory with `sessions_root` and keeping ack state in its own directory with `ack_dir`.
- Supported platforms are macOS arm64 and Linux x86_64 (`required-environments` in `pyproject.toml`).
