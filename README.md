# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

![DSAgt architecture](latex/architecture.png)

DSAgt connects an MCP-compatible AI coding agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure. DSAgt provides data-pipeline scaffolding around a user's existing agent CLI or VS Code extension (Claude Code, Goose, Codex, …);

## Installation

**Prerequisites:** Python 3.10–3.13, [uv](https://github.com/astral-sh/uv), and one of the supported agent platforms below — already installed and authenticated against whatever LLM provider you intend to use.

| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Codex](https://github.com/openai/codex) | `npm i -g @openai/codex` (or `brew install --cask codex`) | `codex --version` |
| [opencode](https://github.com/sst/opencode) | See [opencode docs](https://opencode.ai/docs/) | `opencode --version` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | `npm i -g @roo-code/cli` | `roo --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |

```bash
git clone https://github.com/AI-ModCon/BaseData_pipeline_agent.git
cd BaseData_pipeline_agent
uv sync                      # add --all-groups for the test suite
source .venv/bin/activate    # so `dsagt` is on PATH
```

## Quick Start

Explore DSAgt knowledge ingest, tool registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](tests/smoke_test/). Uses `claude`; substitute another agent (`goose` / `codex` / `opencode`) if you prefer — the prompts are agent-agnostic.

```bash
# 1. From the dsagt repo root, point a shell var at the fixture data and create the project.
export SMOKE_DIR="$(pwd)/tests/smoke_test"
dsagt init quickstart --agent claude
```

`dsagt init` prints (a) the provider env vars **you** need set in your shell — DSAgt does not manage agent credentials — (b) optional OTel export vars to point the agent's native tracing at this project's MLflow, and (c) the exact one-liner to launch your agent with DSAgt tools enabled.

```bash
# 2. In one terminal, start MLflow for this project:
dsagt mlflow quickstart

# 3. In another terminal, launch claude inside the project directory:
cd ~/dsagt-projects/quickstart && claude
```

Inside the agent, paste these prompts one at a time (substitute the absolute path you exported as `$SMOKE_DIR` — the chat doesn't expand env vars):

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > I have a CSV utility called `csvtool`. Its reference is at `$SMOKE_DIR/knowledge/api_reference.md` — register the `filter` subcommand. Use an underscore in the name.
3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`, then summarize `samples.csv` — columns, row count, quality issues.
4. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns. Then tell me what you remember about the samples dataset.

Exit the agent (`Ctrl+C` or `/exit`), then distill the session into episodic memory:

```bash
# 4. After your session, distill traces into episodic memory:
dsagt memory --project quickstart
```

What this exercised:

| Prompt | Layer |
|---|---|
| 1 | Knowledge MCP server (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | Registry MCP server (`save_tool_spec`) — writes `tools/csvtool_filter.md` |
| 3 | `dsagt-run` provenance wrapper — records exec layer to `trace_archive/` |
| 4 | Explicit memory (`kb_remember` → `explicit_memories.yaml`) + KB recall |

Verify the artifacts and view traces in the MLflow UI (URL printed by `dsagt mlflow`):

```bash
dsagt info quickstart
ls ~/dsagt-projects/quickstart/{tools,trace_archive}
cat ~/dsagt-projects/quickstart/explicit_memories.yaml
```

The same flow runs non-interactively via `dsagt smoke-test --agent claude` (or `goose` / `codex` / `opencode`), which asserts each artifact is present.

### First-time knowledge base setup

`dsagt setup-kb` builds the shared ChromaDB collections under `~/.dsagt/kb_index/` that every project on this machine reuses. Three of the six collections shown in the [architecture diagram](#architecture) are populated here — the other three are per-project and fill in automatically during use (see [Knowledge Base](#knowledge-base) below):

- **Tool Specs** — DSAgt's bundled tool specs from `src/dsagt/tools/`, tagged with `source: bundled` so the agent finds them via `search_registry` from the very first session.
- **Skills** — DSAgt's bundled skill workflows from `src/dsagt/skills/` (e.g. `datacard-generator`), discovered via `search_skills`.
- **Domain Knowledge** — Reference corpora (NVIDIA NeMo Curator, AI Data Readiness Inspector) downloaded and embedded so the agent has data-curation domain knowledge out of the box.

The Tool Specs and Skills collections are wipe-and-rebuild on every run, so re-run `setup-kb` after upgrading DSAgt to pick up new bundled assets.

```bash
dsagt setup-kb                       # all collections (local embedder, no creds)
dsagt setup-kb --collection nemo_curator
dsagt setup-kb --embedding-backend api --embedding-base-url ... --embedding-api-key ...
```

The default embedder is a local sentence-transformers model (~130 MB of weights downloaded on first run, CPU-side, no API key). Pass `--embedding-backend api` to route through a hosted embedder via LiteLLM (15–30 minutes typical for the reference corpora, depending on rate limits).

## Use Case Examples

End-to-end walkthroughs for representative scientific domains live in [`use_cases/`](use_cases/). Each one covers data acquisition, tool registration, pipeline construction, and agent-driven execution against a real dataset.

| Use case | Domain | Guide |
|----------|--------|-------|
| Microbial isolate processing | Genomics — short-read QC and assembly with `fastp` + `megahit` | [isolate_demo.md](use_cases/microbial_isolates/isolate_demo.md) |
| Cryo-EM data curation | Structural biology — EMPIAR-10017 β-galactosidase micrographs via CryoPPP | [cryoem_demo.md](use_cases/cryoem/cryoem_demo.md) |
| ISAAC / VASP workflows | Materials science — DFT input/output handling with VASP | [use_cases/isaac_vasp/](use_cases/isaac_vasp/) |

## Project Directory

Default location: `~/dsagt-projects/<name>/`. Override with `--location`:

```bash
dsagt init my-project --agent claude --location /data/runs   # /data/runs/my-project/
dsagt init my-project --agent claude --location .            # ./my-project/
```

Projects are registered in `~/.dsagt/projects.yaml` so `dsagt mlflow <name>` and `dsagt info <name>` work from any directory. The data layer (knowledge base, MLflow store, registered tools, skills, audit records) is agent-agnostic, so re-running `dsagt init <same-name> --agent <other>` switches platforms while preserving everything you've accumulated.

```
~/dsagt-projects/cheese-metagenome/
  dsagt_config.yaml             # project configuration
  tools/                        # registered CLI tool specs (markdown + YAML frontmatter)
  tools/code/                   # agent-written tool scripts
  skills/                       # agent skills (SKILL.md + reference docs)
  trace_archive/                # tool execution records (JSON, from dsagt-run)
  mlflow/                       # MLflow traces, metrics, artifacts
  kb_index/                     # knowledge base vector collections
  explicit_memories.yaml        # user-confirmed facts loaded at session start

  # Per-agent runtime config (one of, generated by dsagt init):
  #   claude:   CLAUDE.md, .mcp.json
  #   goose:    goose.yaml, .goosehints
  #   codex:    AGENTS.md, .codex-data/config.toml
  #   opencode: AGENTS.md, opencode.json
  #   roo:      .roomodes, .roo/mcp.json
  #   cline:    .clinerules/, cline_mcp_settings.json (managed via cline mcp add)
```

### MCP Servers

- **Registry** (`dsagt-registry-server`) — Tool registration and dependency installation. Tools are markdown files with YAML frontmatter under `<project>/tools/`. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies. The agent discovers tools via `search_registry`.
- **Knowledge** (`dsagt-knowledge-server`) — Semantic search over indexed document collections (FAISS / ChromaDB). Background jobs handle long ingest operations. The agent searches via `kb_search`, ingests via `kb_ingest`, and saves user-confirmed facts via `kb_remember`.

### Tools and Skills

**Tools** are CLI executables defined as markdown files with YAML frontmatter in `<project>/tools/`. The agent registers new tools via the registry MCP server's `save_tool_spec`.

**Skills** are instruction-based agent workflows in `<project>/skills/`. Each skill is a directory containing a `SKILL.md` and optional reference docs. DSAgt ships with a bundled `datacard-generator` skill. The agent discovers skills via `search_skills`.

### Knowledge Base

Six independently-partitioned ChromaDB collections hold everything the agent searches semantically. The first three are global (under `~/.dsagt/kb_index/`, populated by `dsagt setup-kb`); the last three are per-project (under `<project>/kb_index/`, populated automatically during use):

| Collection | Source | Populated by |
|---|---|---|
| **Tool Specs** | Bundled CLI tool specs in `src/dsagt/tools/` | `dsagt setup-kb` |
| **Skills** | Bundled skill workflows in `src/dsagt/skills/` | `dsagt setup-kb` |
| **Domain Knowledge** | NeMo Curator + AIDRIN reference corpora; user-ingested docs | `dsagt setup-kb` + agent's `kb_ingest` |
| **Explicit Memory** | User-confirmed facts | Agent's `kb_remember` (also written to `<project>/explicit_memories.yaml`); the agent fetches via `kb_get_memories` on demand — typically when you ask it to recall — not auto-loaded at session start |
| **Episodic Memory** | Distilled facts from MLflow traces | `dsagt memory --project <name>` (per-category outlier detection via embedding centroids) |
| **Tool Use Records** | `dsagt-run` execution traces | `dsagt-run` wrapper writes JSON to `<project>/trace_archive/`; indexed into ChromaDB by `dsagt memory` |

The default embedding backend is local (sentence-transformers, CPU-side, no API needed). Switch to `embedding.backend: api` in `dsagt_config.yaml` to route through a hosted embedder via LiteLLM. Cross-encoder reranking is optional (`knowledge.rerank: true`).

The agent searches via `kb_search` (knowledge MCP server) and writes via `kb_ingest` / `kb_remember`. Tool Specs and Skills are queried through specialized routes (`search_registry`, `search_skills`) over the same backend.

### Observability

MLflow runs at `http://localhost:<mlflow_port>` (pinned at init time, listed by `dsagt info`). The trace view shows:

- **Knowledge base operations** — `kb.search` / `kb.embed` / `kb.index_search` / `kb.rerank` span trees with per-phase timing.
- **Tool executions** — `tool.execute` spans with exit code, duration, file counts, truncated stderr. Full payload in `trace_archive/<record_id>.json`.
- **Registry events** — `save_tool_spec`, `install_dependencies`, `reconstruct_pipeline` spans.
- **Native agent OTel** *(optional)* — when you export `MLFLOW_TRACKING_URI` and `OTEL_EXPORTER_OTLP_ENDPOINT` (printed by `dsagt init`), the agent's own LLM-call traces land in the same MLflow store. Trace coverage varies by agent: claude / goose emit full payloads, codex emits token counts + tool names, opencode emits nothing natively.

Every span carries the project's `session.id` for filtering. Tool execution records on disk provide the canonical provenance chain — the agent calls `reconstruct_pipeline` to render the trace archive as a reproducible bash script or Snakemake workflow.

## CLI Reference

| Command | Description |
|---------|-------------|
| `dsagt init <name> --agent <platform> [--location <path>] [--mlflow-port N]` | Create a project; write per-agent MCP config; print launch one-liner |
| `dsagt mlflow <name>` | Run MLflow in the foreground against a project's store (port pinned at init time) |
| `dsagt memory --project <name>` | Distill new traces from this project's MLflow into episodic memory |
| `dsagt info <name> [--json]` | Resolved config (with source per value) and a session/error summary |
| `dsagt setup-kb [--collection <name>]` | Build the shared core knowledge base collections |
| `dsagt list` | List all projects with agent, status, and path |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project (and optionally delete its directory) |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode]` | End-to-end install verification |

For tests, proxy mode, troubleshooting, and other developer-facing material, see [developer.md](developer.md).
