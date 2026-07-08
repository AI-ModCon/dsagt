# Architecture

![DSAgt architecture](assets/architecture.png)

DSAgt provides a preconfigured agent platform with augmented capabilities for AI-ready scientific data processing and curation. Its capabilities are designed to complement rather than compete with the fast-moving agent platforms it runs on — Claude Code, Codex, Cline, opencode, and Goose. Most are exposed to the agent through a central [MCP server](mcp-servers.md): skill discovery and installation, data-processing code execution with provenance, knowledge-base extension and retrieval, and explicit and cross-session memory. Others are spawned by the server and run in the background: observability through MLflow traces, episodic memory management, and vector-store indexing. DSAgt also provides a suite of agent skills for AI-ready data workflows.

## Capabilities

**Code Registry** (`dsagt-server`)
The agent registers CLI data processing codes as markdown files with YAML frontmatter under `<project>/codes/`. DSAgt handles dependency installation via `uv run --with` and wraps every execution with `dsagt-run` for provenance capture. The agent discovers codes via `search_registry`.

**[Knowledge Base](knowledge-base.md)** (`dsagt-server`)
Hybrid semantic + BM25 search over ChromaDB collections partitioned by concern — code specs, the skill corpus, scientific documents, and per-project memory — served by the same process as the code registry. A first `dsagt init` installs the built-in code specs and a default `genesis` skill corpus; heavier scientific collections (NeMo Curator, AIDRIN) and additional skill sources are also available (e.g. k-dense, anthropic), and new collections can be added for the documents of a specific scientific pursuit. The agent searches via `kb_search`, ingests via `kb_ingest`, and saves user-confirmed facts via `kb_remember`. Opt-in episodic memory distills each session turn into the per-project `session_memory` collection.

**[Provenance](provenance.md)** (`dsagt-run`)
A wrapper around every registered-code execution. Records the command, arguments, exit code, duration, file counts, and truncated stderr to `<project>/trace_archive/<record_id>.json` and emits a `code.execute` span to the trace store. The server incrementally embeds those records into a `code_use` collection so past executions are retrievable, and the agent calls `reconstruct_pipeline` to render the trace archive as a reproducible workflow.

**[Observability](observability.md)** (serverless MLflow)
Traces land in a serverless MLflow store — a SQLite file at `<project>/mlflow.db`. DSAgt emits its own spans live; the agent's LLM-call traces are recovered post-hoc from the on-disk session transcript by the MCP server's in-session reader. View with `dsagt traces <project>`.

**[Skills Discovery](skills.md)**
DSAgt exposes MCP tools to connect to external GitHub skill repositories and search them for skills that enhance scientific workflows. On top of the agent's own progressive disclosure of the skills already installed in its native skills directory, DSAgt maintains an extendable corpus of skills that can be searched and installed on demand — without flooding the agent's context window with skills it isn't using. The agent searches via `search_skills` and installs via `install_skill`.

**[Memory](memory.md)**
DSAgt adds two memory extensions that complement the host agent's own memory (which distills session facts into a managed set of Markdown files loaded into context). *Explicit memory* records user-confirmed facts as YAML (`kb_remember` / `kb_get_memories`). *Episodic memory* (opt-in) keeps a vector store of semantically chunked turn blocks and searches them by successive filtering — first to a session, then by regex over the query's key themes, then a final vector ranking of what remains — so long, multi-session history can augment agent context without the noisy retrieval that undifferentiated episodic accrual would otherwise produce.

## Project Layout

<!-- Shared with README.md — edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:project-tree:start -->"
   end="<!-- md-shared:project-tree:end -->"
%}

Projects are registered in `~/dsagt-projects/projects.yaml` so `dsagt <cli-cmd> <project>` works from any directory.  The project's data is agent-agnostic — re-running `dsagt init <project>` for an existing project and choosing a different agent switches platforms while preserving all accumulated knowledge and traces. `dsagt rm <project>` deletes the project directory (and all accumulated project assets within) and updates the `projects.yaml` registry. 
