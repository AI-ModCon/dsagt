# DSAGT

**D**ata **S**cience **A**gent **T**oolkit — AI-assisted data pipeline builder.

DSAGT connects an MCP-compatible AI agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure. Projects are configured via a single YAML file and managed through the `dsagt` CLI.

## Installation

**Prerequisites:** Python 3.10–3.13, [uv](https://github.com/astral-sh/uv), an LLM API key (e.g. Anthropic, OpenAI), and an MCP-compatible agent ([Claude Code](https://github.com/anthropics/claude-code), [Goose](https://github.com/block/goose), [Roo Code](https://marketplace.visualstudio.com/items?itemName=RooVeterinaryInc.roo-cline), or [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev)).

```bash
git clone <repository-url>
cd dsagt
uv sync --all-groups
```

## Repository Layout

```
├── src/dsagt/
│   ├── __init__.py              # Package init, exports ToolRegistry
│   ├── mcp_utils.py             # Shared MCP server helpers
│   ├── registry.py              # Tool skill file parsing and execution
│   ├── registry_server.py       # MCP server: tool registration
│   ├── knowledge.py             # Document retrieval (FAISS/Chroma)
│   ├── knowledge_server.py      # MCP server: knowledge base
│   ├── pipeline_server.py       # MCP server: tool execution
│   ├── config.py                # Project config loading and validation
│   ├── session.py               # Project init, config gen, services
│   ├── cli.py                   # dsagt init/start/stop/status
│   ├── run.py                   # dsagt-run execution wrapper
│   ├── proxy_callback.py        # LiteLLM callback: tool records
│   ├── proxy.py                 # dsagt-proxy: LiteLLM + OTel
│   ├── skills/                  # Bundled default skill files
│   └── tools/
│       └── scan_directory.py    # Directory scanner tool
├── agents/                      # Per-platform configs and READMEs
├── tests/                       # Unit, integration, and smoke tests
├── scripts/
│   └── setup_core_kb.py         # Build core KB collections
├── use_cases/                   # Domain-specific demo packages
├── pyproject.toml               # Package metadata and entry points
└── README.md
```

## Quick Start

```bash
# Create a project
dsagt init cheese-metagenome --agent claude-code

# Edit the config (set API keys, adjust model, etc.)
vim runtime/cheese-metagenome/dsagt_config.yaml

# Start services and launch the agent
dsagt start cheese-metagenome
```

`dsagt start` generates agent-specific config files, starts background services (LLM proxy and MLflow), and launches the agent with the correct environment variables. When the agent session ends, background services continue running for MLflow access. Stop them with:

```bash
dsagt stop cheese-metagenome
```

## Configuration

Project configuration lives in `runtime/<project>/dsagt_config.yaml`.

```yaml
project: cheese-metagenome
agent: claude-code              # claude-code | goose | roo | cline

llm:
  model: claude-sonnet-4-20250514
  api_key: ${LLM_API_KEY}

embedding:
  model: nomic-embed-text
  api_key: ${LLM_API_KEY}

proxy:
  port: 4000

mlflow:
  port: 5001
  backend: sqlite                 # sqlite | flat-file
```

Environment variable references (`${VAR_NAME}`) are resolved at startup. Set API keys in your shell profile:

```bash
export LLM_API_KEY="your-api-key"
```

To switch agent platforms, change `agent:` and run `dsagt start` again. The CLI generates the correct platform-specific config files from the same `dsagt_config.yaml`.

## Runtime Project Directory

Each project is a self-contained directory under `runtime/`:

```
runtime/cheese-metagenome/
  dsagt_config.yaml     # project configuration
  trace_archive/        # tool execution records (from proxy + dsagt-run)
  mlflow/               # MLflow data (traces, metrics)
  skills/               # registered tool definitions
  kb_index/             # knowledge base collections
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
│  etc.)   │    │  └ DSAGT ─────────▶ Tool Records
└────┬─────┘    └─────────────┘
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
2. **DSAGT callback** — creates tool execution records with three layers:
   - **Intent** (from proxy) — what the agent asked to run
   - **Execution** (from `dsagt-run` wrapper) — exact command, stdout/stderr, timing
   - **Report** (from proxy) — what the agent reported back

### Tool Registration

Tools are defined as skill files (markdown with YAML frontmatter) in `runtime/<project>/skills/`. New tools are registered through the registry MCP server: the agent analyzes a CLI tool or its documentation, proposes a tool spec, and saves it via `save_tool_spec`. Registered tools are available immediately.

### Knowledge Base

Semantic search over indexed document collections (FAISS + optional cross-encoder reranking). Domain documentation, package references, and standards are ingested through the knowledge MCP server.

### Observability

MLflow is available at `http://localhost:<mlflow_port>` during and after sessions, providing token usage and cost per LLM call, latency and model information, and full request/response traces.

Tool execution records on disk provide the provenance chain for pipeline reconstruction.

## Agent Platforms

DSAGT supports four agent platforms. `dsagt start` generates platform-specific configuration automatically. See the agent-specific guides for platform details:

- [`agents/claude-code/README.md`](agents/claude-code/README.md) — CLI and VS Code
- [`agents/goose/README.md`](agents/goose/README.md) — CLI and VS Code
- [`agents/roo/README.md`](agents/roo/README.md) — VS Code only
- [`agents/cline/README.md`](agents/cline/README.md) — VS Code only

For CLI agents (Claude Code, Goose), `dsagt start` launches the agent directly. For VS Code agents (Roo, Cline), it starts the background services and prints instructions to open VS Code.

## CLI Reference

```bash
dsagt init <project> --agent <platform>   # Create a new project
dsagt start <project>                      # Start services and launch agent
dsagt stop <project>                       # Stop background services
dsagt status <project>                     # Show project status
```

Additional tools available as CLI entry points:

```bash
dsagt-run --tool <name> -- <command>       # Wrap a tool for execution provenance
dsagt-proxy                                 # Start the LiteLLM proxy standalone
```

## Tests

```bash
uv run pytest
uv run pytest tests/test_config.py -v      # Config and session tests
uv run pytest tests/test_run.py -v         # dsagt-run wrapper tests
uv run pytest tests/test_proxy_callback.py -v  # Proxy callback tests
```

## Troubleshooting

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

Verify that the proxy's OTel endpoint points at the MLflow port and that MLflow is running:

```bash
dsagt status <project>
curl http://localhost:5001
```
