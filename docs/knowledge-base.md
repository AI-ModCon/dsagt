# Knowledge Base

DSAgt maintains independently-partitioned ChromaDB collections, with six core DSAgt collections set up and used through `dsagt init` and the agent's MCP tools:

## Collections

| Collection | Source | Populated by |
|---|---|---|
| **Code Specs** | Bundled CLI code specs | `dsagt init` (always set up) |
| **Skills Catalog** | Installable skills from external repos (one per source) | `dsagt init` (chosen sources) + `add_skill_source` |
| **Domain Knowledge** | NeMo Curator + AIDRIN reference collections; user-ingested docs | `dsagt init` (chosen collections) + agent's `kb_ingest` |
| **Explicit Memory** | User-confirmed facts | Agent's `kb_remember` (also written to `<project>/.dsagt/explicit_memories.yaml`) |
| **Episodic Memory** (`session_memory`) | Chunked session turns | Captured during the session (opt-in; see below) |
| **Code Execution Records** (`code_use`) | `dsagt-run` execution traces | `dsagt-run` writes JSON to `<project>/trace_archive/`; indexed for search during the session |

## Explicit Memory

Explicit memories are facts the user confirms during a session. The agent saves them via `kb_remember`, which writes to both the ChromaDB collection and `<project>/.dsagt/explicit_memories.yaml`. The agent fetches them via `kb_get_memories` on demand (typically when you ask it to recall something) — they are not auto-loaded at session start. If the vector store is unavailable, explicit memory degrades to pure-YAML.

## Episodic Memory

When enabled, DSAgt reads the agent's transcript as the session runs and captures each completed turn into the `session_memory` collection — a fast, local chunk-and-embed pass that reuses the same embedder as the rest of the knowledge base.

Retrieval over `session_memory` is filtered to session, and then regex over key query terms prior to a **recency-weighted** ranked semantic-vector retrieval: a newer turn edges out a stale one as a bounded boost, so a corrected fact wins by recency while a strongly-relevant old turn is never buried.

## Search

The agent searches all collections via `kb_search` and writes via `kb_ingest` / `kb_remember`. Registered codes have their own `search_registry` route over the same backend. Skills are discovered separately — installed ones natively by the agent, installable ones via `search_skills` over the external catalog (see [Codes & Skills](codes-skills.md)).

Hybrid search (semantic embeddings + keyword BM25) is on by default per collection. Optional reranking sharpens the top results. The default embedder is a local sentence-transformers model (~130 MB, CPU-side, no API key).

## Setup

`dsagt init` sets up the KB. `--include` / `--exclude` (asset names, or `all`) select which collections to include; the bundled Code Specs collection is always included. Re-running `dsagt init` on an existing project reconfigures it in place.
