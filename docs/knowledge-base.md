# Knowledge Base

DSAgt maintains six independently-partitioned ChromaDB collections. The first three are global (under `~/.dsagt/kb_index/`, populated by `dsagt setup-kb`); the last three are per-project (under `<project>/kb_index/`, populated automatically during use).

## Collections

| Collection | Source | Populated by |
|---|---|---|
| **Tool Specs** | Bundled CLI tool specs in `src/dsagt/tools/` | `dsagt setup-kb` |
| **Skills** | Bundled skill workflows in `src/dsagt/skills/` | `dsagt setup-kb` |
| **Domain Knowledge** | NeMo Curator + AIDRIN reference corpora; user-ingested docs | `dsagt setup-kb` + agent's `kb_ingest` |
| **Explicit Memory** | User-confirmed facts | Agent's `kb_remember` (also written to `<project>/explicit_memories.yaml`) |
| **Episodic Memory** | Distilled facts from MLflow traces | `dsagt memory --project <name>` |
| **Tool Use Records** | `dsagt-run` execution traces | `dsagt-run` wrapper writes JSON to `<project>/trace_archive/`; indexed by `dsagt memory` |

## Explicit Memory

Explicit memories are facts the user confirms during a session. The agent saves them via `kb_remember`, which writes to both the ChromaDB collection and `<project>/explicit_memories.yaml`. The agent fetches them via `kb_get_memories` on demand (typically when you ask it to recall something) — they are not auto-loaded at session start.

## Episodic Memory

`dsagt memory --project <name>` distills new traces from the project's MLflow store into episodic memory using per-category outlier detection over embedding centroids. Run this after each session to accumulate cross-session memory.

## Search

The agent searches all collections via `kb_search` (knowledge MCP server) and writes via `kb_ingest` / `kb_remember`. Tool Specs and Skills are queried through specialized routes (`search_registry`, `search_skills`) over the same backend.

Hybrid search (dense embeddings + sparse BM25 via Reciprocal Rank Fusion) is on by default per collection route. Cross-encoder reranking is optional.

## Setup

```bash
dsagt setup-kb                       # all global collections (local embedder)
dsagt setup-kb --collection nemo_curator
dsagt setup-kb --embedding-backend api \
    --embedding-base-url <url> \
    --embedding-api-key <key>
```

The Tool Specs and Skills collections are wiped and rebuilt on every `setup-kb` run — re-run after upgrading DSAgt to pick up new bundled assets.
