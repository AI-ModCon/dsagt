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

## Capabilities

| Capability | Description |
|-------|-------------|
| **Code Registry** | Register CLI codes as markdown specs; the agent discovers them via `search_registry` and runs them from its shell |
| **Knowledge Base** | Hybrid semantic + keyword (BM25) search over indexed ChromaDB collections |
| **Skills Discovery** | Search the external skill corpus and install workflow skills on demand via `search_skills` / `install_skill`; an uninstalled skill takes no space in the agent's context |
| **Provenance** | `dsagt-run` wrapper records every code execution to `trace_archive/`; `reconstruct_pipeline` renders it as a runnable script |
| **Explicit Memory** | User-confirmed facts persisted to YAML and the knowledge base |
| **Episodic Memory** | Opt-in: the MCP server chunks and embeds each session turn into a searchable `session_memory` collection (recency-weighted retrieval) |
| **Observability** | Serverless MLflow tracing (a per-project SQLite file): DSAgt's own spans plus agent traces recovered from the on-disk transcript |

The [Quick Start](quickstart.md) exercises all of these in a single session.
