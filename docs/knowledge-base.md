# Knowledge Base

The knowledge base is DSAgt's catalog of domain knowledge, reference corpora and your own documents, that the agent searches to ground its work on scientific data processing and AI-readiness evaluation.

![DSAgt knowledge base](assets/knowledge-base.png)

## Domain-knowledge collections

| Collection | Source | Populated by |
|---|---|---|
| **Reference corpus** | NeMo Curator (data-curation references) | `dsagt init` (chosen collections) |
| **Your documents** | Papers, standards, protocols, schemas you ingest | Agent's `kb_ingest` |

The agent has six knowledge base tools: `kb_ingest` (index a folder into a named collection; the ingest runs in the background), `kb_append` (add documents to an existing collection), `kb_job_status` (poll a background job), `kb_search` (retrieve across one or more collections, with a `where` metadata filter and regex or substring filters over the chunk text), `kb_list_collections` (every collection with its purpose, the metadata keys a `where` filter takes, and its chunk count), and `kb_delete_collection` (remove a collection and its index; the four collections dsagt owns are refused).

The three tools that name a collection take it as `collection`, and `kb_ingest` requires it: a collection is made only where one was asked for by name. `kb_ingest` and `kb_append` also take the collection's `description`, which is what `kb_list_collections` serves and what a later session reads to choose one collection over another; a folder with a `DESCRIPTION.md` of its own supplies it when the caller gives none.

## Hybrid vector search

Retrieval is hybrid, dense semantic embeddings fused with sparse BM25 keyword matching, per collection by default:

- **Semantic embeddings** match paraphrase and synonymy: a query about "missing values" finds a passage on "null rates" with no shared words.
- **BM25 keyword matching** matches the exact terms embeddings under-rank (identifiers, gene names, parameter flags, standard names), where a literal match matters.
- **Per-collection partitioning** scopes a search to a domain, so a materials-science query is ranked against materials references only.

The default embedder is `BAAI/bge-small-en-v1.5` run locally on onnxruntime from the ONNX export the model's repository publishes (133 MB, downloaded once).

## Shared vector store

The same vector store also holds DSAgt's [memory](memory.md) (explicit and episodic), the [skill corpus](skills.md), and the [execution records](provenance.md) (the `code_use` collection). Each is a separate collection in that store, sharing one embedder and one ChromaDB; the linked pages describe each collection.

## Setup

`dsagt init` sets up the knowledge base from your choices in the interactive menu.

## Try it

```bash
dsagt init            # name it `demo`, and check `nemo_curator` at the knowledge-collections menu
dsagt start demo
```

Then, in the agent, substituting `<your-docs-folder>` with any folder of your
own documents (papers, protocols, schemas):

1. > Ingest the docs in `<your-docs-folder>` into a collection named `domain`.
2. > Search the `domain` and `nemo_curator` collections for how to assess data quality.

