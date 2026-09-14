# DSAgt

<!-- Shared with README.md. Edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:intro:start -->"
   end="<!-- md-shared:intro:end -->"
%}

## Installation

<!-- Shared with README.md. Edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:install:start -->"
   end="<!-- md-shared:install:end -->"
%}

For a development install from a clone, see the [Developer Guide](developer.md).

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
