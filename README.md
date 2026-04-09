# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure. Projects are configured via a single YAML file and managed through the `dsagt` CLI.

## Installation

**Prerequisites:** Python 3.10–3.13, [uv](https://github.com/astral-sh/uv), and one of the supported agent platforms.

```bash
git clone <repository-url>
cd dsagt
uv sync --all-groups
```

Install your agent platform of choice:

| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | `npm i -g @roo-code/cli` | `roo --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |

See the `agents.py` docstring for platform-specific details on generated files and proxy routing.

## Quick Start

```bash
# Create a project (defaults to ~/dsagt-projects/cheese-metagenome/)
dsagt init cheese-metagenome --agent claude-code

# Edit the config — set API keys, model, embedding endpoint
vim ~/dsagt-projects/cheese-metagenome/dsagt_config.yaml

# Start services and launch the agent
dsagt start cheese-metagenome
```

`dsagt start` generates agent-specific config files into the project directory, starts background services (LLM proxy and MLflow), and launches the agent. When the agent exits, memory extraction runs and services are stopped automatically.

## Configuration

Project configuration lives in `<project_dir>/dsagt_config.yaml`. This is the single source of truth — all API keys, endpoints, and settings flow from here to every subprocess and MCP server. No shell profile environment variable exports needed.

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

Environment variable references (`${VAR_NAME}`) are resolved at startup if you prefer that pattern, but putting keys directly in the config works too. To switch agent platforms, change `agent:` and run `dsagt start` again.

## Project Directory

Each project is a self-contained directory. The default location is `~/dsagt-projects/<name>/`, or specify a custom location with `--location`:

```bash
dsagt init my-project --agent claude-code                        # ~/dsagt-projects/my-project/
dsagt init my-project --agent claude-code --location /data/runs  # /data/runs/my-project/
dsagt init my-project --agent claude-code --location .           # ./my-project/
```

Projects are registered in `~/.dsagt/projects.yaml` so `dsagt start <name>` works from any directory.

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

## How It Works

### Architecture

```
┌──────────┐    ┌─────────────┐    ┌───────────────┐
│  Agent   │───▶│ LiteLLM     │───▶│ LLM Provider  │
│ (Claude, │    │ Proxy       │    └───────────────┘
│  Goose,  │    │  ├ OTel ──────────▶ MLflow
│  Roo,    │    │  └ DSAgt ─────────▶ Tool Records
│  Cline)  │    └─────────────┘
└────┬─────┘
     │
     │ MCP
     ▼
┌──────────┐    ┌──────────────┐
│ Registry │    │ Knowledge    │
│ Server   │    │ Server       │
└──────────┘    └──────────────┘
```

All LLM calls route through a local LiteLLM proxy. Two callbacks capture data:

1. **OTel callback** (LiteLLM built-in) — exports spans, token counts, latency, and cost to MLflow.
2. **DSAgt callback** — creates tool execution records with three layers:
   - **Intent** (from proxy) — what the agent asked to run
   - **Execution** (from `dsagt-run` wrapper) — exact command, stdout/stderr, timing
   - **Report** (from proxy) — what the agent reported back

### Tools and Skills

**Tools** are CLI executables defined as markdown files with YAML frontmatter in `<project>/tools/`. New tools are registered through the registry MCP server via `save_tool_spec`. Executables are automatically wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies. The agent discovers tools via `search_registry`, which uses semantic search backed by ChromaDB.

**Skills** are instruction-based agent workflows in `<project>/skills/`. Each skill is a directory containing a `SKILL.md` with a workflow definition and optional reference docs. DSAgt ships with a bundled `datacard-generator` skill. The agent discovers skills via `search_skills`.

### Knowledge Base

Semantic search over indexed document collections (FAISS + optional cross-encoder reranking). Domain documentation, package references, and standards are ingested through the knowledge MCP server. To set up the core knowledge base collections:

```bash
dsagt-setup-kb
```

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

### Standalone Tools

| Command | Description |
|---------|-------------|
| `dsagt-run --tool <name> -- <command>` | Wrap a tool for execution provenance |
| `dsagt-proxy` | Start the LiteLLM proxy standalone |
| `dsagt-setup-kb` | Build core knowledge base collections |

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

Integration tests validate real API endpoints (embedding, LLM) and MCP server subprocess behavior. Copy `tests/test_site_config.yaml.example` to `tests/test_site_config.yaml` and fill in your institution's values.

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
