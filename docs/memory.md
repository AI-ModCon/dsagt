# Memory

DSAgt gives the agent two kinds of persistent memory backed by the project's vector store: **explicit memory** — facts the user confirms — and opt-in **episodic memory** — an automatic record of session turns. Both are retrievable by the agent with `kb_search` / `kb_get_memories` MCP tools.

## Explicit memory

Explicit memories are facts the user confirms during a session. The agent saves them via `kb_remember`, which writes to both the ChromaDB collection and `<project>/.dsagt/explicit_memories.yaml`. It fetches them via `kb_get_memories` on demand — typically when you ask it to recall something — so they are not auto-loaded at session start. 

Explicit memory is always on. It is the reliable path: when you say "remember that…", the agent records the fact so a future session can retrieve it.

## Episodic memory

Episodic memory is **opt-in**. When enabled, DSAgt reads the agent's transcript as the session runs and captures each completed turn into the `session_memory` collection — a fast, local chunk-and-embed pass that reuses the same embedder as the rest of the knowledge base.

Retrieval over `session_memory` filters first to a session, then by regex over the query's key terms, before a final **recency-weighted** semantic ranking: a newer turn edges out a stale one as a bounded boost, so a corrected fact wins by recency while a strongly-relevant old turn is never buried.

## Try it

Explicit memory needs no setup:

```bash
dsagt init            # name it `demo`; answer "yes" to "Enable episodic memory?" to also capture turns
dsagt start demo
```

Then, in the agent:

1. > Remember that `samples.csv` has null values in the status and timestamp columns.
2. > *(later, or in a new session)* What do you remember about the samples dataset?

Confirm it persisted to disk:

```bash
cat ~/dsagt-projects/demo/.dsagt/explicit_memories.yaml
```

## In practice

See the [Use Cases](use-cases/index.md), where confirmed facts about a dataset — its quirks, thresholds, and decisions — carry forward across a multi-step curation session.
