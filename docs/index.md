# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

DSAgt connects an MCP-compatible AI coding agent to code registration, a semantic knowledge base, execution provenance, and observability infrastructure. It provides data-pipeline scaffolding around your existing agent CLI or VS Code extension (Claude Code, Goose, Codex, and others).

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
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt && uv sync --all-groups
source .venv/bin/activate
```

## Key Capabilities

| Capability | What it does |
|-------|-------------|
| **Code Registry** | Register CLI codes as markdown specs; the agent discovers and runs them via `search_registry` |
| **Knowledge Base** | Hybrid semantic + keyword (BM25) search over indexed ChromaDB collections |
| **Skills Discovery** | Search the external skill corpus and install workflow skills on demand via `search_skills` / `install_skill`, without flooding the agent's context |
| **Provenance** | `dsagt-run` wrapper records every code execution to `trace_archive/`; `reconstruct_pipeline` renders it as a runnable script |
| **Explicit Memory** | User-confirmed facts persisted to YAML and the knowledge base |
| **Episodic Memory** | Opt-in: the MCP server mechanically chunks and embeds each session turn into a searchable `session_memory` collection (recency-weighted retrieval) |
| **Observability** | Serverless MLflow tracing (a per-project SQLite file) — DSAgt's own spans plus agent traces recovered from the on-disk transcript |

See the [Quick Start](quickstart.md) to try all of these in a single session.
