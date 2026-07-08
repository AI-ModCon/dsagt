# MCP Server

DSAgt exposes its capabilities through a single MCP server, **`dsagt-server`**, configured in the per-agent runtime file (`.mcp.json` for Claude Code, `goose.yaml` for Goose, etc.) and launched automatically when the agent starts. It combines four capabilities — a code registry, a [knowledge base](knowledge-base.md), [explicit memory](memory.md), and [skill discovery](skills.md) — behind one process with one shared embedder and one ChromaDB.

The 20 tools split across four concerns, all on the one process.

## Registry tools (8)

Code registration, execution helpers, dependency installation, and pipeline reconstruction. See [Provenance](provenance.md) for how registered codes are captured.

| Tool | Description |
|------|-------------|
| `search_registry` | Semantic search over registered + built-in code specs |
| `get_registry` | List every registered code with its MCP-compatible schema |
| `save_code_spec` | Register a CLI code as `codes/<name>/SKILL.md` (executable auto-wrapped with `dsagt-run` + `uv run --with`), mirrored into the agent's native skills dir |
| `install_dependencies` | Install a code's Python dependencies via `uv run --with` |
| `run_command` | Execute a command to capture its help/usage output |
| `read_file` | Read a text file from disk |
| `http_request` | Fetch documentation or an API spec over HTTP(S) |
| `reconstruct_pipeline` | Render `trace_archive/` as a dependency-ordered bash script or Snakemake workflow |

Codes are markdown files with YAML frontmatter under `<project>/codes/`. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies.

## Knowledge tools (5)

Semantic search and ingestion over indexed document collections. See the [Knowledge Base](knowledge-base.md) for the retrieval model.

| Tool | Description |
|------|-------------|
| `kb_search` | Hybrid semantic search across one or more collections (optional metadata, regex, and substring filters) |
| `kb_ingest` | Index a folder as a new collection — returns a `job_id`; poll `kb_job_status` until complete |
| `kb_append` | Add documents to an existing collection (background job) |
| `kb_list_collections` | List collections with their embedding model and vector DB |
| `kb_job_status` | Check the status of a background ingest/append job |

## Memory tools (2)

User-confirmed facts that persist across sessions. See [Memory](memory.md).

| Tool | Description |
|------|-------------|
| `kb_remember` | Store a user-confirmed fact as an explicit memory (`supersedes` to replace an outdated one) |
| `kb_get_memories` | Load active explicit memories for this project (call at session start) |

## Skill tools (5)

Discover, install, and author agent skills, and manage external skill sources. See [Skills](skills.md).

| Tool | Description |
|------|-------------|
| `search_skills` | Search installed skills + the external corpus (corpus hits are tagged `[catalog]` in results) |
| `install_skill` | Copy a skill from the corpus into `<project>/skills/` and mirror it into the agent's native skills dir |
| `save_skill` | Register an agent-authored skill into `<project>/skills/<name>/SKILL.md` |
| `add_skill_source` | Enable + index an external skill source (a known name or a Git URL) into the searchable corpus |
| `list_skill_sources` | List known and synced external skill sources |