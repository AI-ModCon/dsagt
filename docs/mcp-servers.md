# MCP Server

DSAgt exposes its capabilities through a single MCP server, **`dsagt-server`**, configured in the per-agent runtime file (`.mcp.json` for Claude Code, `goose.yaml` for Goose, etc.) and launched automatically when the agent starts. It bundles two concern areas — a tool registry and a knowledge base — behind one process with one shared embedder and one ChromaDB owner.

> Earlier versions ran two separate servers (`dsagt-registry-server` + `dsagt-knowledge-server`), merged in 0.3.0. Re-run `dsagt start <project>` on an existing project to regenerate its config against the single server (for cline, delete `<project>/.cline-data` first).

## Registry tools

Tool registration, dependency installation, and tool discovery.

| Tool | Description |
|------|-------------|
| `search_registry` | Semantic search over registered tool specs |
| `save_tool_spec` | Register a new CLI tool as a markdown file with YAML frontmatter |
| `install_dependencies` | Install tool dependencies via `uv run --with` |
| `reconstruct_pipeline` | Render the trace archive as a bash script or Snakemake workflow |

Tools are markdown files with YAML frontmatter under `<project>/tools/`. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies.

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

The default embedding backend is local (`sentence-transformers`, CPU-only, no API key needed). Switch to `embedding.backend: api` in `dsagt_config.yaml` to route through a hosted embedder via LiteLLM. Cross-encoder reranking is available via `knowledge.rerank: true`.

Hybrid search (dense + sparse BM25) is on by default and controlled per-route via the `hybrid` flag.
