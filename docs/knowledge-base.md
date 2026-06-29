# Knowledge Base

DSAgt maintains six independently-partitioned ChromaDB collections. The first three are machine-wide (built once under `~/dsagt-projects/kb_index/` and copied into each project); the last three are per-project (under `<project>/kb_index/`, populated automatically during use). All are provisioned/used through `dsagt init` and the agent's MCP tools — there is no separate setup command.

## Collections

| Collection | Source | Populated by |
|---|---|---|
| **Tool Specs** | Bundled CLI tool specs in `src/dsagt/tools/` | `dsagt init` (always provisioned) |
| **Skills Catalog** | Installable skills from external repos (one `skills_catalog__<slug>` collection per source), frontmatter-indexed | `dsagt init` (chosen sources) + `add_skill_source` |
| **Domain Knowledge** | NeMo Curator + AIDRIN reference corpora; user-ingested docs | `dsagt init` (chosen collections) + agent's `kb_ingest` |
| **Explicit Memory** | User-confirmed facts | Agent's `kb_remember` (also written to `<project>/.dsagt/explicit_memories.yaml`) |
| **Episodic Memory** (`session_memory`) | Distilled session facts | The MCP server's in-session heartbeat (opt-in; see below) |
| **Tool Use Records** (`tool_use`) | `dsagt-run` execution traces | `dsagt-run` writes JSON to `<project>/trace_archive/`; the heartbeat embeds them incrementally (idempotent) |

## Explicit Memory

Explicit memories are facts the user confirms during a session. The agent saves them via `kb_remember`, which writes to both the ChromaDB collection and `<project>/.dsagt/explicit_memories.yaml`. The agent fetches them via `kb_get_memories` on demand (typically when you ask it to recall something) — they are not auto-loaded at session start. If the vector store is unavailable, explicit memory degrades to pure-YAML.

## Episodic Memory

Episodic memory is **opt-in** (`dsagt init --episodic`, off by default). When enabled, the MCP server's in-session heartbeat reads the agent's transcript and distills each completed turn into a few tagged, ≤1-sentence facts in the `session_memory` collection. Two tiers:

- **Tier-1 (default when enabled)** — a small **local** LLM judge (`Qwen2.5-1.5B`, grammar-constrained JSON) classifies each fact against a closed tag taxonomy (stock "AI-data-ready" tags plus any project `--domain-tags`) and condenses it. Local-by-default: no API key, no cost. The GGUF model (~1 GB) downloads on first use; inference is CPU-side.
- **Tier-0 (fallback)** — mechanical chunk + keyword-tag + embed, no LLM; used automatically if the judge fails, so a turn is never lost.

Retrieval over `session_memory` is **recency-weighted** (`episodic.recency_half_life_days`, default 14): a newer fact edges out a stale one as a bounded boost, so a corrected fact wins without contradiction detection while durable old facts keep their relevance. Optional per-category outlier detection (`episodic.outlier_sensitivity`) can queue novel facts for review.

## Search

The agent searches all collections via `kb_search` and writes via `kb_ingest` / `kb_remember`. Registered tools have their own `search_registry` route over the same backend. Skills are discovered separately — installed ones natively by the agent, installable ones via `search_skills` over the external catalog (see [Tools & Skills](tools-skills.md)).

Hybrid search (dense embeddings + sparse BM25 via Reciprocal Rank Fusion) is on by default per collection. Cross-encoder reranking is optional. The default embedder is a local sentence-transformers model (~130 MB, CPU-side, no API key).

## Provisioning

`dsagt init` provisions the KB. The shared, machine-wide collections are built once (the first project on a machine pays the cost) under `~/dsagt-projects/kb_index/`, then copied into each project. `--include` / `--exclude` (asset names, or `all`) select which collections to provision; the bundled Tool Specs collection is always included. Re-running `dsagt init` on an existing project reconfigures it in place.
