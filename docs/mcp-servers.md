# MCP Server

DSAgt exposes its capabilities through a single MCP server, **`dsagt-server`**, configured in the per-agent runtime file (`.mcp.json` for Claude Code, `goose.yaml` for Goose, etc.) and launched automatically when the agent starts. It combines four capabilities — a code registry, a [knowledge base](knowledge-base.md), [explicit memory](memory.md), and [skill discovery](skills.md) — behind one process with one shared embedder and one ChromaDB.

The 17 tools split across four concerns, all on the one process.

## The server's environment

The per-agent MCP config carries an `env` block with two kinds of variable: routing (the project name and directory, the trace store URI, the embedding backend) and the launching shell's activated environment (`PATH`, `VIRTUAL_ENV`, `CONDA_PREFIX`, `PYTHONPATH`, the library paths, the `module` variables, plus any names listed under `mcp.env_passthrough` in `.dsagt/config.yaml`). Codex and Cline start the server from that block alone, so it is what makes the server's Python the one with the user's packages. The block is written at `dsagt init` and `dsagt start`; after changing the activated environment, run `dsagt start` (or re-init) so the block matches. A credential never enters the block: a name ending in `_KEY`, `_TOKEN`, or `_SECRET`, or containing `SECRET` or `PASSW`, is refused, and dsagt's own service credentials come from the shell or `~/.config/dsagt/env`.

## Registry tools (5)

Code registration, the readiness reports on record, and pipeline reconstruction. See [Provenance](provenance.md) for how registered codes are captured.

| Tool | Description |
|------|-------------|
| `search_registry` | Semantic search over registered + built-in code specs |
| `get_registry` | List every registered code with its MCP-compatible schema |
| `save_code_spec` | Register a CLI code as `skills/<name>/SKILL.md` (executable wrapped with `dsagt-run` + `uv run --with`), linked into the agent's native skills directory |
| `reconstruct_pipeline` | Render `trace_archive/` as a bash script (in run order, output directories created, recorded stdout files redirected) or a Snakemake workflow; `output` saves it under the project |
| `readiness_reports` | The AI-readiness reports on record for a file, each with whether the file is unchanged since that run |

Codes are markdown files with YAML frontmatter under `<project>/skills/`, beside the instruction skills; the `executable` in the frontmatter is what makes a skill a code. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies.

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
| `kb_get_memories` | Retrieve active explicit memories for this project |

## Skill tools (5)

Discover, install, and author agent skills, and manage external skill sources. See [Skills](skills.md).

| Tool | Description |
|------|-------------|
| `search_skills` | Search the external corpus (each hit tagged `[catalog · install_skill to add]`); `skill_name` looks up one installed skill |
| `install_skill` | Copy a skill from the corpus into `<project>/skills/` and link it into the agent's native skills directory |
| `save_skill` | Register an agent-authored skill into `<project>/skills/<name>/SKILL.md` |
| `add_skill_source` | Enable + index an external skill source (a known name or a Git URL) into the searchable corpus |
| `list_skill_sources` | List known and synced external skill sources |