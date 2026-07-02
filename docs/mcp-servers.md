# MCP Server

DSAgt exposes its capabilities through a single MCP server, **`dsagt-server`**, configured in the per-agent runtime file (`.mcp.json` for Claude Code, `goose.yaml` for Goose, etc.) and launched automatically when the agent starts. It bundles four capabilities — a code registry, a knowledge base, explicit memory, and skill discovery — behind one process with one shared embedder and one ChromaDB.

> Earlier versions ran two separate servers (`dsagt-registry-server` + `dsagt-knowledge-server`), merged in 0.2.0. Re-run `dsagt start <project>` on an existing project to regenerate its config against the single server (for cline, delete `<project>/.cline-data` first).

## Registry tools

Code registration, dependency installation, and code discovery.

| Tool | Description |
|------|-------------|
| `search_registry` | Semantic search over registered code specs |
| `save_code_spec` | Register a new CLI code as a markdown file with YAML frontmatter |
| `install_dependencies` | Install code dependencies via `uv run --with` |
| `reconstruct_pipeline` | Render the trace archive as a bash script or Snakemake workflow |

Codes are markdown files with YAML frontmatter under `<project>/codes/`. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies.

## Knowledge tools

Semantic search and ingestion over indexed document collections.

| Tool | Description |
|------|-------------|
| `kb_search` | Search across one or more knowledge collections |
| `kb_ingest` | Index a file or directory into a named collection (runs in background for large corpora) |
| `kb_remember` | Save a user-confirmed fact to explicit memory |
| `kb_get_memories` | Retrieve explicit memories for the current project |
| `search_skills` | Discover installable skills in the external catalog (installed skills are auto-discovered natively) |

### Backend

The default embedding backend is local (`sentence-transformers`, CPU-only, no API key needed). Switch to `embedding.backend: api` in `.dsagt/config.yaml` to route through a hosted, OpenAI-compatible `/v1/embeddings` endpoint (set `embedding.base_url` and export `EMBEDDING_API_KEY`). Cross-encoder reranking is available via `knowledge.rerank: true`.

Hybrid search (dense + sparse BM25, fused by Reciprocal Rank Fusion) is always on per collection — it is a property of the store, not a per-call flag.
