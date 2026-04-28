# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure.

## Installation

**Prerequisites:** Python 3.10–3.13, [uv](https://github.com/astral-sh/uv), and one of the supported agent platforms:
| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | `npm i -g @roo-code/cli` | `roo --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |
| [Codex](https://github.com/openai/codex) | `npm i -g @openai/codex` (or `brew install --cask codex`) | `codex --version` |

```bash
git clone https://github.com/AI-ModCon/BaseData_pipeline_agent.git
cd BaseData_pipeline_agent
uv sync                      # add --all-groups for the test suite
source .venv/bin/activate    # activate the venv so `dsagt` is on PATH

cp .env.example .env         # then edit .env with your endpoint + keys
set -a; source .env; set +a  # export vars into the shell
```

`dsagt` resolves all `${VAR}` references in `dsagt_config.yaml` from the environment, so values set in `.env` flow into every project automatically. Re-run `source .venv/bin/activate` (and re-source `.env`) in each new shell.

### Environment variables

| Var | Purpose |
|---|---|
| `LLM_PROVIDER`        | LiteLLM provider prefix (`openai`, `anthropic`, `bedrock`, ...). [Full list](https://docs.litellm.ai/docs/providers). |
| `LLM_API_KEY`         | Auth key for the agent's LLM endpoint (use any non-empty placeholder if the endpoint doesn't auth, e.g. local Ollama). |
| `LLM_BASE_URL`        | Agent's LLM endpoint URL. |
| `LLM_MODEL`           | Model name the agent talks to. |
| `EMBEDDING_API_KEY`   | Auth key for the embedding endpoint (often the same as `LLM_API_KEY`). |
| `EMBEDDING_BASE_URL`  | OpenAI-compatible embedding endpoint. |
| `EMBEDDING_MODEL`     | Embedding model name. |

### First-time knowledge base setup

`dsagt setup-kb` downloads reference repositories and papers (NeMo Curator,
AIDRIN) and embeds them into ChromaDB collections that every project shares.
With `.env` already sourced:

```bash
dsagt setup-kb
```

**Expect this to take 10–20 minutes** if the upstream embedding API is
rate-limited (common on Azure-backed endpoints). You only need to do this
once per machine; the resulting index lives under `~/.dsagt/kb_index/`
and is reused by every project. Pass `--collection nemo_curator` or
`--collection aidrin` to build only one collection at a time.

## Quick Start

```bash
# Create a project (defaults to ~/dsagt-projects/cheese-metagenome/)
dsagt init cheese-metagenome --agent claude-code

# Start services and launch the agent — picks up endpoint and keys from .env
dsagt start cheese-metagenome
```

`dsagt start` resolves the agent (CLI flag or YAML default), picks free ports, starts background services (LLM proxy, MLflow), writes the agent's runtime config files with the actual ports baked in, and launches the agent with MCP servers (Tools/Skills, Knowledge Base). When the agent exits, memory extraction runs and services are stopped automatically.

`--agent` may be supplied at init OR at first start — the flag is required exactly once across the project's lifetime. After that, the choice is recorded in `dsagt_config.yaml` and `dsagt start` runs without the flag. Passing `--agent` on later starts is a per-run override that doesn't update the YAML default — handy for trying a different agent on the same project's accumulated knowledge base, MLflow data, and registered tools.

```bash
# Agent-agnostic init: defer the choice until first start
dsagt init my-project
dsagt start my-project --agent codex      # codex becomes YAML default
dsagt start my-project                    # uses codex
dsagt start my-project --agent goose      # per-run override; YAML still says codex

# If your configured port is taken, dsagt falls back with a warning;
# pass --proxy-port / --mlflow-port to pick deliberately.
dsagt start my-project --proxy-port 4100 --mlflow-port 5101
```

The generated `dsagt_config.yaml` references `.env` via `${VAR}` placeholders — no editing required for the common case. Edit it only to override per-project (different model, port, MLflow backend, etc.).

**Verify your install end-to-end** (~1 min, makes a few real LLM calls):

```bash
dsagt smoke-test --agent claude-code   # or: goose | cline | roo | codex
```

This drives the chosen agent through a fixed 6-step script (knowledge ingest → tool registration → directory scan → CSV analysis → explicit memory → recall) and asserts the resulting trace records, KB index, and explicit-memory file are present. See [Smoke Test](#smoke-test) below for what each agent's run produces.

## Use Case Examples

End-to-end walkthroughs for representative scientific domains live in
[`use_cases/`](use_cases/). Each one covers data acquisition, tool
registration, pipeline construction, and agent-driven execution against
a real dataset.

| Use case | Domain | Guide |
|----------|--------|-------|
| Microbial isolate processing | Genomics — short-read QC and assembly with `fastp` + `megahit` | [isolate_demo.md](use_cases/microbial_isolates/isolate_demo.md) |
| Cryo-EM data curation | Structural biology — EMPIAR-10017 β-galactosidase micrographs via CryoPPP | [cryoem_demo.md](use_cases/cryoem/cryoem_demo.md) |
| ISAAC / VASP workflows | Materials science — DFT input/output handling with VASP | [use_cases/isaac_vasp/](use_cases/isaac_vasp/) |

## Configuration

Project configuration lives at `<project_dir>/dsagt_config.yaml`. `dsagt init`
generates the file with `${VAR}` references that resolve from `.env` at
load time — credentials stay in `.env` (gitignored), the YAML stays
shareable. Override any field by replacing its `${VAR}` with a literal
value if a project needs different settings than the workspace default.

```yaml
project: cheese-metagenome
agent: claude-code              # claude-code | goose | roo | cline | codex

llm:
  provider: ${LLM_PROVIDER}     # LiteLLM provider prefix (openai, anthropic, ...)
  model: ${LLM_MODEL}
  base_url: ${LLM_BASE_URL}
  api_key: ${LLM_API_KEY}

embedding:
  model: ${EMBEDDING_MODEL}
  base_url: ${EMBEDDING_BASE_URL}
  api_key: ${EMBEDDING_API_KEY}

proxy:
  port: 4000

mlflow:
  port: 5001
  backend: sqlite               # sqlite | flat-file

knowledge:
  chunk_size: 1024              # characters per text chunk during ingestion
  vector_db: chroma             # default vector store for new collections
  rerank: false                 # enable cross-encoder reranking (slower, more accurate)
```

To switch agent platforms, either edit `agent:` in the YAML, or pass
`--agent X` to `dsagt start` (per-run override; doesn't change the YAML).
The project's data layer (knowledge base, MLflow store, registered
tools, skills, audit records) is agent-agnostic, so switching agents
preserves everything you've accumulated. To inspect resolved values
and where each came from (`.env`, environment, or literal), run
`dsagt info <project>`.

## Project Directory

Default location: `~/dsagt-projects/<name>/`. Override with `--location`:

```bash
dsagt init my-project --agent claude-code                        # ~/dsagt-projects/my-project/
dsagt init my-project --agent claude-code --location /data/runs  # /data/runs/my-project/
dsagt init my-project --agent claude-code --location .           # ./my-project/
```

Projects are registered in `~/.dsagt/projects.yaml` so `dsagt start <name>` works from any directory.

### Project directory layout
```
~/dsagt-projects/cheese-metagenome/
  dsagt_config.yaml               # project configuration
  tools/                          # registered CLI tool specs (markdown + YAML frontmatter)
  tools/code/                     # agent-written tool scripts
  skills/                         # instruction-based agent skills (SKILL.md + reference docs)
  trace_archive/                  # tool execution records (JSON, from proxy + dsagt-run)
  mlflow/                         # MLflow traces, metrics, and artifacts
  kb_index/                       # knowledge base vector collections (ChromaDB/FAISS)

  # Agent-platform config (generated by dsagt start)
  # Claude Code:  CLAUDE.md, .mcp.json
  # Goose:        goose.yaml, .goosehints, .dsagt_env
  # Roo Code:     .roo/mcp.json, .roomodes, .dsagt_env
  # Cline:        cline_mcp.json, .clinerules/dsagt_instructions.md, .dsagt_env
  # Codex:        AGENTS.md, .codex-data/config.toml, .dsagt_env

  # Service logs (written while dsagt start is running)
  proxy.log                       # LiteLLM proxy — LLM request/response forwarding
  mlflow.log                      # MLflow server — trace collection and UI
  dsagt_knowledge_server.log      # Knowledge MCP server — search, ingest, memory
  dsagt_registry_server.log       # Registry MCP server — tool/skill registration
```

## Architecture

![Sketch of Architecture](latex/architecture.png)

All LLM calls route through a local LiteLLM proxy. Two callbacks capture data:

1. **OTel callback** (LiteLLM built-in) — exports spans, token counts, latency, and cost to MLflow.
2. **DSAgt callback** — creates tool execution records with three layers:
   - **Intent** (from proxy) — what the agent asked to run
   - **Execution** (from `dsagt-run` wrapper) — exact command, stdout/stderr, timing
   - **Report** (from proxy) — what the agent reported back

The same MLflow OTel collector also receives spans from the knowledge
server, the registry server, and `dsagt-run`, so KB searches, tool
executions, and registry events show up alongside LLM calls in the trace
view.

### Tools and Skills

**Tools** are CLI executables defined as markdown files with YAML frontmatter in `<project>/tools/`. New tools are registered through the registry MCP server by the agent via `save_tool_spec`. Executables are automatically wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies. The agent discovers tools via `search_registry`.

**Skills** are instruction-based agent workflows in `<project>/skills/`. Each skill is a directory containing a `SKILL.md` with a workflow definition and optional reference docs. DSAgt ships with a bundled `datacard-generator` skill. The agent discovers skills via `search_skills`.

### Knowledge Base

Semantic search over indexed document collections. Collections default to
ChromaDB with metadata filtering and incremental updates; FAISS and other
backends are available per-collection for pre-built indexes. Cross-encoder
reranking is optional (`knowledge.rerank: true` in config). The shared
core collections are built once with `dsagt setup-kb`.

### Observability

MLflow is available at `http://localhost:<mlflow_port>` during and after
sessions. The trace view shows:

- **Knowledge base operations** — `kb.search` → `kb.embed` → `kb.index_search` → `kb.rerank` span trees with per-phase timing.
- **Tool executions** — `tool.execute` spans with exit code, duration, file counts, and truncated stderr. Full payload in `trace_archive/<record_id>.json`.
- **Registry events** — `save_tool_spec`, `install_dependencies`, and `reconstruct_pipeline` spans.

Every span carries the project's `session.id` for filtering. Tool execution
records on disk provide the canonical provenance chain — the agent calls
`reconstruct_pipeline` to render the trace archive as a reproducible bash
script or Snakemake workflow.

## CLI Reference

| Command | Description |
|---------|-------------|
| `dsagt init <name> [--agent <platform>] [--location <path>]` | Create a new project; `--agent` is optional (defer to first start if omitted) |
| `dsagt start <name> [--agent <platform>] [--proxy-port N] [--mlflow-port N] [--script <file>] [--max-turns N]` | Resolve agent → pick ports (with auto-fallback) → start services → write configs → launch; clean up on exit |
| `dsagt stop <name>` | Stop project services (proxy, MLflow); clean up orphans on configured ports |
| `dsagt info <name> [--json]` | Show resolved config (with source per value) and a session/source/error trace summary |
| `dsagt mlflow <name> [--port N]` | Run MLflow in the foreground against a project's store |
| `dsagt setup-kb [--collection <name>]` | Build the core knowledge base |
| `dsagt list` | List all projects with agent, status, and path |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project and delete its directory |
| `dsagt smoke-test [--agent goose\|claude-code\|cline\|roo\|codex]` | Run the end-to-end smoke test (sources `.env`, drives the agent non-interactively, asserts artifacts) |

## Code Organization

```
src/dsagt/
  session.py       # Config, project lifecycle, services, extraction
  agents.py        # Agent config generation, environment, launch
  registry.py      # ToolRegistry + SkillRegistry, KB indexing
  knowledge.py     # KnowledgeBase, embeddings (via LiteLLM), vector indexes
  memory.py        # Explicit + episodic memory, outlier detection
  provenance.py    # Execution capture, LLM tracking, record indexing, pipeline reconstruction
  observability.py # OTel tracing helpers + sidechannel-call interception

  commands/
    cli.py               # User-facing CLI (init/start/stop/info/mlflow/list/mv/rm/smoke-test/setup-kb)
    info.py              # `dsagt info` — config-source + trace summary report
    setup_core_kb.py     # Core KB build logic (called via dsagt setup-kb)
    proxy_server.py      # LiteLLM proxy (internal, launched by dsagt start)
    run_tool.py          # Tool execution wrapper (internal, launched per tool call)
    registry_server.py   # Registry MCP server (internal, launched by dsagt start)
    knowledge_server.py  # Knowledge MCP server (internal, launched by dsagt start)
```

## Tests

```bash
# All unit tests (no API keys needed)
uv run pytest -m "not integration"

# Integration tests (require .env with valid credentials)
uv run pytest -m integration -v

# All tests
uv run pytest
```

Integration tests read endpoint and key values from `.env` at the repo root — the same file `dsagt start` uses. Copy `.env.example` to `.env` and fill in your values.

## Smoke Test

End-to-end check that exercises every layer (proxy, MCP servers, dsagt-run wrapper, memory, MLflow) through a real agent run.

```bash
dsagt smoke-test --agent claude-code   # one of: goose | claude-code | cline | roo | codex
```

The script lifecycle: clean slate (`dsagt rm` + delete dir) → `dsagt init` with the chosen agent → `dsagt start --script` against [`tests/smoke_test/script.txt`](tests/smoke_test/script.txt) (a 6-step task list) → artifact assertions. Each agent gets its own project at `smoke-test-<agent>/`, so consecutive runs across agents preserve state for cross-agent comparison via `dsagt info smoke-test-<agent>`.

Artifacts the run asserts:

| Check | What it verifies |
|---|---|
| `csvtool_filter spec written` | agent called `dsagt-registry.save_tool_spec` |
| `trace_archive has records` + `scan_directory record` | agent invoked the registered tool through the `dsagt-run` provenance wrapper, not bare `python` |
| `knowledge ingested (route + vectors)` | agent called `dsagt-knowledge.kb_ingest` and ChromaDB index was populated |
| `explicit memory recorded` | agent called `kb_remember` (file at `explicit_memories.yaml`) |
| `mlflow has traces` + `LLM dispatch parity` | proxy log requests = MLflow `litellm-*` traces (modulo sidechannel mocks) |

For a slower hands-on tour with the same script, see the manual [`WALKTHROUGH.md`](tests/smoke_test/WALKTHROUGH.md).

## Troubleshooting

### Agent command not found

If `dsagt start` reports "Command not found", the agent CLI isn't installed. Check the install table above.

### MCP servers not connecting

Check that uv can find the server commands:

```bash
uv run which dsagt-registry-server
uv run which dsagt-knowledge-server
```

If not found, reinstall: `uv sync --reinstall`

### Proxy not intercepting LLM calls

Verify the proxy is running:

```bash
dsagt list
curl http://localhost:4000/v1/messages
```

### MLflow UI empty

Verify MLflow is running:

```bash
dsagt list
curl http://localhost:5001
```

## Sidechannel model calls

At the end of a `dsagt start` session you may see a yellow warning like:

```
  ⚠ Sidechannel model calls intercepted:
      claude-haiku-4-5-20251001  (2 calls)
    Two possible causes:
      (1) agent sidechannel (e.g. title generator) — safe to ignore
      (2) typo in dsagt_config.yaml llm.model — these replies are canned, not real
```

**What happened.** The agent you ran sent requests for a model that isn't the one configured in `dsagt_config.yaml`. The proxy has a wildcard route that catches any unrecognized model name and returns a canned mock reply ("session") instead of forwarding upstream. No tokens spent, no error in MLflow.

**Why agents do this.** Every major agent platform hardcodes a small/fast model for internal features that aren't part of the main conversation:

| Agent | Sidechannel model | Used for |
|---|---|---|
| goose | `gpt-4o-mini` | Session-name generation (the label you see in the session list) |
| claude-code | `claude-haiku-4-5-20251001` | Conversation title generation |
| others | varies | — |

These names are baked into the agent and ignore `GOOSE_MODEL` / `ANTHROPIC_MODEL`. If your upstream gateway doesn't carry the exact bare name (most lab gateways alias with suffixes like `-v1-project`), the request would 400 without the wildcard mock — the warning is just telling you the mock fired.

**When to worry.** If the model name in the warning matches what you *thought* you'd configured as your primary model, you have a typo in `dsagt_config.yaml` — your agent is getting canned replies, not real completions. Fix the `llm.model` field and rerun.

**Why a single wildcard instead of per-agent mocks.** Agent vendors rename sidechannel models every release. Maintaining an explicit list would accumulate dead code as those names drift. The wildcard catches today's names and tomorrow's without intervention — LiteLLM prefers exact matches over wildcards, so your configured primary model always routes normally.
