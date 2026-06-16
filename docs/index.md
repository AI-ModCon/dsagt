# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI coding agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure. It provides data-pipeline scaffolding around your existing agent CLI or VS Code extension (Claude Code, Goose, Codex, and others).

## Supported Agents

| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Codex](https://github.com/openai/codex) | `npm i -g @openai/codex` | `codex --version` |
| [opencode](https://github.com/sst/opencode) | See [opencode docs](https://opencode.ai/docs/) | `opencode --version` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | `npm i -g @roo-code/cli` | `roo --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |

## Prerequisites

- Python 3.12–3.13
- [uv](https://github.com/astral-sh/uv)
- One of the supported agent platforms above, installed and authenticated against your LLM provider

## Installation

```bash
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync
source .venv/bin/activate
```

## Key Capabilities

| Layer | What it does |
|-------|-------------|
| **Tool Registry** | Register CLI tools as markdown specs; the agent discovers and runs them via `search_registry` |
| **Knowledge Base** | Semantic search over indexed document collections (ChromaDB + FAISS) |
| **Provenance** | `dsagt-run` wrapper records every tool execution to `trace_archive/` and MLflow |
| **Explicit Memory** | User-confirmed facts persisted to YAML and the knowledge base |
| **Episodic Memory** | Session distillation via outlier detection over MLflow traces |
| **Observability** | Full OTLP tracing to a local MLflow instance |

See the [Quick Start](quickstart.md) to try all of these in a single session.
