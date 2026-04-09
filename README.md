# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure.

## Installation

**Prerequisites:** Python 3.10–3.13, [uv](https://github.com/astral-sh/uv), and one of the supported agent platforms.

```bash
git clone https://github.com/AI-ModCon/BaseData_pipeline_agent.git
cd BaseData_pipeline_agent
uv sync --all-groups
```

### First-time knowledge base setup

`dsagt-setup-kb` downloads reference repositories and papers (NeMo Curator,
AIDRIN) and embeds them into ChromaDB collections that every project shares.
The embedder needs an OpenAI-compatible API (or a local sentence-transformers
model). Easiest path is to export environment variables once, then run the
setup command:

```bash
export LLM_API_KEY=sk-...                          # your embedding API key
export OPENAI_BASE_URL=https://api.example.com/v1  # OpenAI-compatible endpoint
export EMBEDDING_MODEL=text-embedding-3-small      # embedding model name

uv run dsagt-setup-kb   # builds the core AI-ready-data knowledge base
```

Alternatively, pass everything on the command line:

```bash
uv run dsagt-setup-kb \
  --embedding-base-url https://api.example.com \
  --embedding-api-key sk-... \
  --embedding-model text-embedding-3-small
```

**Expect this to take 15–30 minutes** over an API-backed embedder — the core
KB contains the full NeMo Curator and AIDRIN source trees plus several arXiv
papers, and embedding is rate-limited by the upstream service. You only
need to do this once per machine; the resulting index lives under
`~/.dsagt/kb_index/` and is reused by every project. Pass
`--collection nemo_curator` or `--collection aidrin` to build only one
collection at a time.

Install preferred agent platform:

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

`dsagt start` generates agent-specific config files into the project directory, starts background services (LLM proxy, MLflow), and launches the agent with MCP servers (Tools/Skills, Knowledge Base). When the agent exits, memory extraction runs and services are stopped automatically.

## Use Case Examples

End-to-end walkthroughs for representative scientific domains live in
[`use_cases/`](use_cases/). Each one is a self-contained guide covering data
acquisition, tool registration, pipeline construction, and agent-driven
execution against a real dataset.

| Use case | Domain | Guide |
|----------|--------|-------|
| Microbial isolate processing | Genomics — short-read QC and assembly with `fastp` + `megahit` | [use_cases/microbial_isolates/isolate_demo.md](use_cases/microbial_isolates/isolate_demo.md) |
| Cryo-EM data curation | Structural biology — EMPIAR-10017 β-galactosidase micrographs via CryoPPP | [use_cases/cryoem/cryoem_demo.md](use_cases/cryoem/cryoem_demo.md) |
| ISAAC / VASP workflows | Materials science — DFT input/output handling with VASP | [use_cases/isaac_vasp/](use_cases/isaac_vasp/) |

These exercises are also the best way to validate a fresh install beyond the
basic smoke test — they cover knowledge ingestion, paired check/operation
tool generation, multi-stage pipeline execution, and memory extraction on
non-trivial domain data.

## Configuration

Project configuration with API keys, endpoints, and settings is located at `<project_dir>/dsagt_config.yaml`.  

```yaml
project: cheese-metagenome
agent: claude-code              # claude-code | goose | roo | cline

llm:
  model: claude-sonnet-4-20250514
  api_key: your-llm-api-key     # or ${LLM_API_KEY} to read from env

embedding:
  model: text-embedding-3-small
  base_url: https://api.example.com/v1   # OpenAI-compatible embedding endpoint
  api_key: your-embedding-api-key

proxy:
  port: 4000

mlflow:
  port: 5001
  backend: sqlite               # sqlite | flat-file
```

To switch agent platforms, change `agent:` and run `dsagt start` again.

## Project Directory

The default location for all project resources, logs, and artifacts is `~/dsagt-projects/<name>/`. A custom location is specified with the `--location` flag:

```bash
dsagt init my-project --agent claude-code                        # ~/dsagt-projects/my-project/
dsagt init my-project --agent claude-code --location /data/runs  # /data/runs/my-project/
dsagt init my-project --agent claude-code --location .           # ./my-project/
```

Projects are registered in `~/.dsagt/projects.yaml` so `dsagt start <name>` works correctly regardless current working directory in the shell.

### Example project directory
```
~/dsagt-projects/cheese-metagenome/
  dsagt_config.yaml     # project configuration
  tools/                # registered CLI tool specs
  tools/code/           # agent-written tool scripts
  skills/               # instruction-based agent skills (SKILL.md + reference docs)
  trace_archive/        # tool execution records (from proxy + dsagt-run)
  mlflow/               # MLflow data (traces, metrics)
  kb_index/             # knowledge base collections
  .mcp.json             # generated agent MCP config
  proxy.log             # proxy output (when running)
  mlflow.log            # MLflow output (when running)
```

## Architecture

![Sketch of Architecture](latex/architecture.png)


All LLM calls route through a local LiteLLM proxy. Two callbacks capture data:

1. **OTel callback** (LiteLLM built-in) — exports spans, token counts, latency, and cost to MLflow.
2. **DSAgt callback** — creates tool execution records with three layers:
   - **Intent** (from proxy) — what the agent asked to run
   - **Execution** (from `dsagt-run` wrapper) — exact command, stdout/stderr, timing
   - **Report** (from proxy) — what the agent reported back

### Tools and Skills

**Tools** are CLI executables defined as markdown files with YAML frontmatter in `<project>/tools/`. New tools are registered through the registry MCP server by the agent via `save_tool_spec`. Executables are automatically wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies. The agent discovers tools via `search_registry`, which exposes agent options for direct tool lookup or semantic search backed by ChromaDB.

**Skills** are instruction-based agent workflows in `<project>/skills/`. Each skill is a directory containing a `SKILL.md` with a workflow definition and optional reference docs. DSAgt ships with a bundled `datacard-generator` skill. The agent discovers skills via `search_skills` mitigating concerns with skills context explosion for complicated data processing pipelines.

### Knowledge Base

Semantic search over indexed document collections. Collections default to
ChromaDB (HNSW with metadata filtering and incremental updates); FAISS and
other backends are available per-collection via `CollectionRoute` for
user-supplied pre-built indexes. Cross-encoder reranking is optional.
Domain documentation, package references, and standards are ingested
through the knowledge MCP server. The shared core collections are built
once with `dsagt-setup-kb` (see [First-time knowledge base setup](#first-time-knowledge-base-setup)).

### Observability

MLflow is available at `http://localhost:<mlflow_port>` during and after sessions, providing token usage and cost per LLM call, latency and model information, and full request/response traces.

Tool execution records on disk provide the provenance chain for pipeline reconstruction. The agent can call `reconstruct_pipeline` to generate a reproducible bash script or Snakemake workflow from the trace archive.

## CLI Reference

### Project Management

| Command | Description |
|---------|-------------|
| `dsagt init <name> --agent <platform>` | Create a new project at `~/dsagt-projects/<name>/` |
| `dsagt init <name> --agent <platform> --location <path>` | Create at `<path>/<name>/` |
| `dsagt start <name>` | Generate configs, start services, launch agent, clean up on exit |
| `dsagt stop <name>` | Stop orphaned services (after a crash) |
| `dsagt status <name>` | Show project status and running services |
| `dsagt list` | List all registered projects and their locations |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt extract <name>` | Manually extract memories from the session log |

## Code Organization

```
src/dsagt/
  session.py       # Config, registry, project lifecycle, services, extraction
  agents.py        # Agent config generation, environment, launch
  registry.py      # ToolRegistry + SkillRegistry, KB indexing
  knowledge.py     # KnowledgeBase, embeddings, vector indexes
  memory.py        # Explicit + episodic memory, outlier detection
  provenance.py    # Execution capture, LLM tracking, record indexing, pipeline reconstruction

  commands/        # CLI entry points (each has a main() function)
    cli.py               # dsagt init/start/stop/status/list/mv/extract
    proxy_server.py      # dsagt-proxy
    run_tool.py          # dsagt-run
    registry_server.py   # dsagt-registry-server (MCP)
    knowledge_server.py  # dsagt-knowledge-server (MCP)
    setup_core_kb.py     # dsagt-setup-kb
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

If `dsagt start` reports "Command not found", the agent CLI isn't installed. Check the install table above or the `agents.py` docstring for platform-specific instructions.

### MCP servers not connecting

Check that uv can find the server commands:

```bash
uv run which dsagt-registry-server
uv run which dsagt-knowledge-server
```

If not found, reinstall: `uv sync --reinstall`

### Proxy not intercepting LLM calls

Verify the proxy is running and the agent has the right base URL:

```bash
dsagt status <project>
curl http://localhost:4000/v1/messages
```

### MLflow UI empty

Verify that MLflow is running:

```bash
dsagt status <project>
curl http://localhost:5001
```
