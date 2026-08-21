# Memory

DSAgt gives the agent two kinds of persistent memory backed by the project's vector store: **explicit memory** — facts the user confirms — and opt-in **episodic memory** — an automatic record of session turns. Both are retrievable by the agent with `kb_search` / `kb_get_memories` MCP tools.

![DSAgt memory](assets/memory.png)

## Explicit memory

Explicit memories are facts the user confirms during a session. The agent saves them via `kb_remember`, which writes to both the ChromaDB collection and `<project>/.dsagt/explicit_memories.yaml`. It fetches them via `kb_get_memories` on demand — typically when you ask it to recall something — so they are not auto-loaded at session start. 

## Episodic memory

When episodic memory is enabled, DSAgt reads the agent's transcript as the session runs and captures each completed turn into the `session_memory` collection — a fast, local chunk-and-embed pass that reuses the same embedder as the rest of the [knowledge base](knowledge-base.md).

Retrieval over `session_memory` filters first to a session, then by regex over the query's key terms, before a final **recency-weighted** semantic ranking: a newer turn edges out a stale one as a bounded boost, so a corrected fact wins by recency while a strongly-relevant old turn is never buried.

## Comparison with platform-native memory

Agent platforms provide their own memory: instruction files such as `CLAUDE.md` and `.goosehints`, and on some platforms notes the agent saves for itself. That memory is agent-curated (the model decides what is worth saving), stored as prose files loaded whole into context, and tied to one platform's format. DSAgt memory differs on each point:

- **Mechanical capture.** Episodic memory records every completed turn from the transcript; nothing depends on the model choosing to save it.
- **Retrieval on demand.** Memories are recalled by search with recency weighting, not loaded whole into context, so the record can grow without consuming the context window.
- **One store across agents.** The same collections and YAML files serve all five supported platforms, so memory persists across a switch of agent.
- **Auditable facts.** Explicit memories are user-confirmed and durable, with superseded entries kept in a history file.

## Try it

```bash
dsagt init            # name it `demo`; answer "yes" to "Enable episodic memory?" to also capture turns
dsagt start demo
```

Then, in the agent — **explicit** memory (you confirm a fact to store):

1. > Remember that `samples.csv` has null values in the status and timestamp columns.
2. > *(later, or in a new session)* What do you remember about the samples dataset?

Confirm it persisted to disk:

```bash
cat ~/dsagt-projects/demo/.dsagt/explicit_memories.yaml
```

And **episodic** memory (captured automatically — no `remember` step):

1. > For this dataset, let's treat any column with over 5% nulls as unusable.
2. > *(later, or in a new session)* What null threshold did we agree on for unusable columns?

The agent recalls the decision from `session_memory` even though you never explicitly stored it. Confirm the collection materialized:


```bash
ls ~/dsagt-projects/demo/kb_index/session_memory/
```