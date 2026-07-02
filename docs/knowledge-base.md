# Knowledge Base

The knowledge base is DSAgt's catalog of **domain knowledge** — reference corpora and your own documents — that the agent searches to ground its work on scientific data-processing and AI-readiness evaluation. Rather than let the agent guess at a library's API or a standard's requirements, it retrieves the relevant passages first, then acts.

## Domain-knowledge collections

| Collection | Source | Populated by |
|---|---|---|
| **Reference corpora** | NeMo Curator + AIDRIN (data-curation and AI-data-readiness references) | `dsagt init` (chosen collections) |
| **Your documents** | Papers, standards, protocols, schemas you ingest | Agent's `kb_ingest` |

The agent works these with three tools: `kb_ingest` (index a file or directory into a named collection — long ingests run in the background), `kb_search` (retrieve across one or more collections), and `kb_list_collections` (see what's indexed).

## Why hybrid vector search

Retrieval is **hybrid** — dense semantic embeddings fused with sparse BM25 keyword matching — and it's on by default per collection:

- **Semantic embeddings** catch paraphrase and synonymy: a query about "missing values" finds a passage on "null rates" even with no shared words.
- **BM25 keyword matching** catches the exact terms embeddings tend to under-rank — identifiers, gene names, parameter flags, standard names — where a literal match matters.
- **Per-collection partitioning** scopes a search to a domain, so a materials-science query isn't diluted by genomics references.
- **Optional cross-encoder reranking** re-scores the top candidates for precision when it's worth the extra pass.

The default embedder is a local sentence-transformers model (~130 MB, CPU-side, no API key).
## One store, many concerns

The same vector store backs more than domain knowledge. DSAgt's [memory](memory.md) (explicit + episodic), [skills discovery](skills.md) (the installable-skill catalog), and [code execution tracking](provenance.md) (the `code_use` records) each live in it as their own partitioned collections, sharing one embedder and one ChromaDB. This page covers domain-knowledge cataloging; those pages carry the depth on their respective collections.

## Setup

`dsagt init` sets up the knowledge base from your choices in the interactive menu. Re-running `dsagt init` on an existing project reconfigures it in place.

## Try it

```bash
dsagt init            # name it `demo`, and enable the AIDRIN collection in the prompts
dsagt start demo
```

Then, in the agent:

1. > Ingest the docs in `knowledge/` into a collection named `domain`.
2. > Search the `domain` and `aidrin` collections for how to assess data completeness.

## In practice

See the [Use Cases](use-cases/index.md), where domain references and ingested protocols guide the agent through curating a real scientific dataset.
