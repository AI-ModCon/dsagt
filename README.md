# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure.

## Installation

**Prerequisites:** Python 3.10–3.13, [uv](https://github.com/astral-sh/uv), and one of the supported agent platforms.

```bash
git clone https://github.com/AI-ModCon/BaseData_pipeline_agent.git
cd BaseData_pipeline_agent
uv sync          # add --group dev if you plan to run the test suite
```

### First-time knowledge base setup

`dsagt setup-kb` downloads reference repositories and papers (NeMo Curator,
AIDRIN) and embeds them into ChromaDB collections that every project shares.
The embedder needs an OpenAI-compatible API (or a local sentence-transformers
model). Export environment variables once, then run the setup command:

```bash
export LLM_API_KEY=sk-...                          # your embedding API key
export OPENAI_BASE_URL=https://api.example.com/v1  # OpenAI-compatible endpoint
export EMBEDDING_MODEL=text-embedding-3-small      # embedding model name

dsagt setup-kb
```

Alternatively, pass everything on the command line:

```bash
dsagt setup-kb \
  --embedding-base-url https://api.example.com \
  --embedding-api-key sk-... \
  --embedding-model text-embedding-3-small
```

**Expect this to take 10–20 minutes** if the upstream embedding API is
rate-limited (common on Azure-backed endpoints). You only need to do this
once per machine; the resulting index lives under `~/.dsagt/kb_index/`
and is reused by every project. Pass `--collection nemo_curator` or
`--collection aidrin` to build only one collection at a time.

### Install an agent platform

| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | `npm i -g @roo-code/cli` | `roo --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |

## Quick Start

```bash
# Create a project (defaults to ~/dsagt-projects/cheese-metagenome/)
dsagt init cheese-metagenome --agent claude-code

# Edit the config — set API keys, model, embedding endpoint
vim ~/dsagt-projects/cheese-metagenome/dsagt_config.yaml

# Start services and launch the agent
dsagt start cheese-metagenome
```

`dsagt start` generates agent-specific config files, starts background services (LLM proxy, MLflow), and launches the agent with MCP servers (Tools/Skills, Knowledge Base). When the agent exits, memory extraction runs and services are stopped automatically.

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
generates sensible defaults; edit to fill in your API credentials and
endpoint:

```yaml
project: cheese-metagenome
agent: claude-code              # claude-code | goose | roo | cline

llm:
  model: claude-sonnet-4-20250514
  base_url: https://api.example.com    # LLM endpoint (used for memory extraction)
  api_key: ${LLM_API_KEY}              # resolved from env var, or paste a literal key

embedding:
  model: text-embedding-3-small
  base_url: https://api.example.com/v1   # OpenAI-compatible embedding endpoint
  api_key: ${LLM_API_KEY}

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

To switch agent platforms, change `agent:` and run `dsagt start` again.

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
| `dsagt init <name> --agent <platform>` | Create a new project |
| `dsagt init <name> --agent <platform> --location <path>` | Create at a custom location |
| `dsagt start <name>` | Start services, launch agent, clean up on exit |
| `dsagt setup-kb [--collection <name>]` | Build the core knowledge base |
| `dsagt list` | List all projects with agent, status, and path |
| `dsagt mv <name> <new-location>` | Move a project to a new location |

## Code Organization

```
src/dsagt/
  session.py       # Config, project lifecycle, services, extraction
  agents.py        # Agent config generation, environment, launch
  registry.py      # ToolRegistry + SkillRegistry, KB indexing
  knowledge.py     # KnowledgeBase, embeddings (via LiteLLM), vector indexes
  memory.py        # Explicit + episodic memory, outlier detection
  provenance.py    # Execution capture, LLM tracking, record indexing, pipeline reconstruction
  observability.py # OTel tracing helpers

  commands/
    cli.py               # User-facing CLI (dsagt init/start/setup-kb/list/mv)
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

# Integration tests (requires tests/test_site_config.yaml with valid credentials)
uv run pytest -m integration -v

# All tests
uv run pytest
```

Integration tests validate API endpoints (embedding, LLM) and MCP server subprocess behavior. Copy `tests/test_site_config.yaml.example` to `tests/test_site_config.yaml` and fill in your institution's values.

## Smoke Test

A guided walkthrough that validates all core functionality is available at [`tests/smoke_test/WALKTHROUGH.md`](tests/smoke_test/WALKTHROUGH.md). It covers knowledge ingestion, tool registration, execution provenance, memory, observability, and cleanup.

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
