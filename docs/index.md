# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI coding agent to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure. It provides data-pipeline scaffolding around your existing agent CLI or VS Code extension (Claude Code, Goose, Codex, and others).

## Supported Agents

<!-- Shared with README.md — edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:agents:start -->"
   end="<!-- md-shared:agents:end -->"
%}

## Prerequisites

- Python 3.12 or 3.13
- One of the supported agent platforms above, installed and authenticated against your LLM provider
- [uv](https://github.com/astral-sh/uv) — only for the development install

## Installation

### For use (no development)

<!-- Shared with README.md — edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:install:start -->"
   end="<!-- md-shared:install:end -->"
%}

### For development

Clone the repo and use `uv` (editable install; add `--all-groups` for the test suite):

```bash
pip install https://github.com/AI-ModCon/dsagt/archive/refs/tags/0.1.0.zip
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
