---
language:
- en
tags:
- project:genesis
- team:BASE-DATA
- type:agent
- science:general
- risk:general
license: Apache-2.0
base_model: N/A  # DSAgt is agent-platform-agnostic; it wraps Claude Code, Goose, Codex, opencode, or Cline

datasets:
    - # NeMo Curator reference corpus (optional knowledge collection, indexed at dsagt init)
    - # AI Data Readiness Inspector (AIDRIN) reference corpus (optional knowledge collection)

metrics:
    - # Code registration success rate
    - # Knowledge base search precision
    - # Provenance trace completeness

agent_card:
  name: "DSAgt (DataSmith Agent Toolkit)"
  description: "AI-assisted data pipeline builder developed under the DOE Genesis Mission. Wraps an MCP-compatible agent CLI with code registration, a semantic knowledge base, skill discovery, execution provenance, and observability infrastructure to accelerate AI-ready scientific data preparation."
  provider:
    organization: "DOE AI ModCon Base Data Team (DOE Genesis Mission)"
    url: "https://github.com/AI-ModCon/dsagt"
  version: "0.2.0"
  documentation_url: "https://ai-modcon.github.io/dsagt/"
  protocol_version: "N/A"  # DSAgt extends existing agent CLIs via MCP; it is not itself an A2A service
  preferred_transport: "stdio"
  capabilities:
    streaming: false
    push_notifications: false
    state_transition_history: true  # trace_archive + the serverless MLflow store record full execution history

  authentication:
    schemes: []  # BYOA — the agent platform owns its own LLM-provider auth; DSAgt never proxies or stores credentials
    credentials: "N/A (optional: EMBEDDING_API_KEY in the shell for a hosted embedding backend — never written to disk)"

  default_input_modes:
    - "text/plain"
  default_output_modes:
    - "text/plain"
    - "application/json"  # code execution records, pipeline reconstructions

  skills:
    - id: "code_registration"
      name: "Code Registration"
      description: "Register CLI codes as skill-standard directories (codes/<name>/SKILL.md) with machine-readable parameters; install dependencies via uv; every execution is wrapped with dsagt-run for provenance. Registered codes are also mirrored into the agent's native skills directory for in-context discovery."
      tags: [registry, provenance, mcp, skills]
      examples:
        - "Register this analysis script as a reusable code."
        - "Install Python dependencies for a custom analysis script."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "application/json"]

    - id: "knowledge_base"
      name: "Knowledge Base"
      description: "Hybrid dense+sparse (sentence-transformers + BM25) semantic search over ChromaDB collections: code specs, skill catalogs, domain knowledge, code-use records, and session memory. Optional cross-encoder reranking and regex/substring document filters."
      tags: [knowledge, chromadb, semantic-search, mcp]
      examples:
        - "Ingest domain documentation into a named collection."
        - "Search for codes matching 'CSV statistics'."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "application/json"]

    - id: "provenance"
      name: "Execution Provenance"
      description: "Record every code invocation (command, stdout/stderr, exit code, timing, file I/O) to trace_archive/ and emit spans to the project's serverless MLflow store. Reconstruct the full execution history as a dependency-ordered pipeline."
      tags: [provenance, mlflow, reproducibility]
      examples:
        - "Reconstruct a reproducible pipeline from prior code executions."
        - "View execution traces in the MLflow UI."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "application/json"]

    - id: "memory"
      name: "Explicit + Episodic Memory"
      description: "Explicit memory stores user-confirmed facts (YAML + vector mirror) via kb_remember / kb_get_memories. Opt-in episodic memory mechanically chunks, tags, and embeds every session turn into a recency-weighted searchable collection — no LLM calls."
      tags: [memory, chromadb]
      examples:
        - "Put this in explicit memory: samples.csv has null values in the status column."
        - "What do you remember about the samples dataset?"
      input_modes: ["text/plain"]
      output_modes: ["text/plain"]

    - id: "skill_discovery"
      name: "Skill Discovery and Installation"
      description: "Search external skill catalogs (Genesis, Anthropic, K-Dense, and others cloned+indexed at init) and install skills into the project, where the agent auto-discovers them natively. Agents can also author and save their own skills."
      tags: [skills, catalog, mcp]
      examples:
        - "Search the skill catalog for a literature-search skill and install it."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "text/markdown"]

Extensions:
  agent_runtime:
    framework: "MCP (Model Context Protocol) over stdio; supported agent platforms: Claude Code, Goose, Codex, opencode, Cline"
    service_endpoint: "stdio (the single dsagt-server is launched as a subprocess by the agent platform)"
    rate_limits: "Determined by the underlying LLM provider configured in the agent platform (BYOA — DSAgt never proxies LLM traffic)."
    logging: "Serverless MLflow store at sqlite:///<project>/mlflow.db (no server to run); full code execution records written to <project>/trace_archive/. Agent LLM-call history is recovered post-hoc from the agent's on-disk transcript."
    memory: "Stateful per-project. Explicit memory: <project>/.dsagt/explicit_memories.yaml + ChromaDB mirror. Episodic memory (opt-in): session_memory ChromaDB collection, embedded per turn. Code-use records: <project>/trace_archive/ + code_use ChromaDB collection."

---

# DSAgt (DataSmith Agent Toolkit)

DSAgt is an AI-assisted data pipeline builder. It connects an MCP-compatible agent CLI (Claude Code, Goose, Codex, opencode, or Cline) to code registration, a semantic knowledge base, skill discovery, execution provenance, and observability infrastructure — without modifying the agent itself.

*Last Updated*: **2026-07-02**

## Developed by

DOE AI ModCon Base Data Team as part of the Genesis Mission.

## Contributed by

- Aaron Tuor (PNNL)
- Andrew Tritt (LBNL)
- Jean Luca Bez (LBNL)
- Jong Choi (ORNL)
- Kyle Parfrey (PPPL)
- Rohith Anand Varikoti (PNNL)
- Shreyas Cholia (LBNL)

See https://github.com/AI-ModCon/dsagt/graphs/contributors for full list.

## Agent Changelog

+ **2026-07-08** v0.2.0 — single merged `dsagt-server` (20 tools); serverless SQLite MLflow store (no ports, no OTel, no proxy); external skill catalogs; proxy-free agent-transcript trace pipeline + opt-in episodic memory; registered executables are **codes** (skill-standard dirs), natively discoverable as soon as they are added; `dsagt traces` viewer
+ **2026-06-30** initial public version (v0.1.0)

## Agent short description

Scaffolding layer that gives any MCP-compatible agent CLI persistent code registration, semantic knowledge retrieval, skill discovery, execution provenance, and session memory — exposed as 20 tools on a single MCP server (`dsagt-server`).

## Agent description

DSAgt wraps an unmodified agent CLI with four independently-operable concerns, exposed by one MCP server the agent discovers through the standard MCP tool protocol:

1. **Code Registry** — The agent registers CLI codes as skill-standard directories (`codes/<name>/SKILL.md`, frontmatter carrying executable + parameters); the server handles dependency installation (`uv run --with`) and wraps each stored command with `dsagt-run` for provenance. Discovery is dual-path: semantic search via `search_registry`, plus a mirror into the agent's native skills directory so the exact runnable command is in context at invocation time.
2. **Knowledge Base** — ChromaDB collections with hybrid dense (sentence-transformers) + sparse (BM25) search and optional cross-encoder reranking. Code specs and selected skill catalogs are indexed at `dsagt init`; per-project collections (code-use records, session memory) fill in during use. Long ingests run as background jobs.
3. **Provenance** — `dsagt-run` captures every code execution (command, stdout/stderr, exit code, timing, file I/O) to `trace_archive/` and emits spans to the project's serverless MLflow store. `reconstruct_pipeline` renders the archive as a dependency-ordered execution history.
4. **Observability & Memory** — All self-logging lands in `sqlite:///<project>/mlflow.db` (MLflow's serverless backend — nothing to run). Agent LLM-call traces are recovered post-hoc from the agent's on-disk transcript, uniformly across all five platforms. Explicit memory stores user-confirmed facts; opt-in episodic memory embeds every session turn for recency-weighted recall.

The data layer is agent-platform-agnostic: switching platforms preserves all accumulated knowledge, codes, skills, and traces.

## Underlying model(s)

- Primary model(s): N/A — DSAgt is platform-agnostic and delegates LLM calls to the configured agent CLI (BYOA; DSAgt never proxies LLM traffic)
- Embedding model: `sentence-transformers` (`bge-small-en-v1.5`, local, CPU-side, default); optionally any OpenAI-compatible hosted embedder (`embedding.backend: api`)
- Cross-encoder reranking: optional, enabled via `knowledge.rerank: true` in `.dsagt/config.yaml`

## Inputs and outputs

### Default interaction modes

- defaultInputModes: `["text/plain"]`
- defaultOutputModes: `["text/plain", "application/json"]`

The agent accepts natural-language instructions (text). Outputs include text responses, registered code specs (skill-standard markdown), execution trace records (JSON), pipeline reconstructions, and installed skills (markdown).

### Skills

- **Skill ID**: `code_registration`
  **Name**: Code Registration
  **Description**: Register CLI codes as skill-standard spec directories; install dependencies; wrap executions with `dsagt-run` for provenance; mirror specs into the agent's native skills dir.
  **Tags**: registry, provenance, mcp, skills
  **Examples**: "Register this analysis script as a reusable code.", "Install Python dependencies for a custom analysis script."
  **Input/Output Modes**: text/plain → text/plain, application/json

- **Skill ID**: `knowledge_base`
  **Name**: Knowledge Base
  **Description**: Hybrid semantic search (dense + BM25, optional reranking, regex/substring filters) over code specs, skill catalogs, domain knowledge, code-use records, and session memory.
  **Tags**: knowledge, chromadb, semantic-search, mcp
  **Examples**: "Ingest domain documentation into a named collection.", "Search for codes matching 'CSV statistics'."
  **Input/Output Modes**: text/plain → text/plain, application/json

- **Skill ID**: `provenance`
  **Name**: Execution Provenance
  **Description**: Record every code invocation to `trace_archive/` + the serverless MLflow store; reconstruct the full execution history as a dependency-ordered pipeline.
  **Tags**: provenance, mlflow, reproducibility
  **Examples**: "Reconstruct a reproducible pipeline from prior code executions."
  **Input/Output Modes**: text/plain → text/plain, application/json

- **Skill ID**: `memory`
  **Name**: Explicit + Episodic Memory
  **Description**: User-confirmed facts via `kb_remember` / `kb_get_memories` (YAML + vector mirror); opt-in episodic memory embeds every turn for recency-weighted cross-session recall (mechanical — no LLM calls).
  **Tags**: memory, chromadb
  **Examples**: "Put this in explicit memory: samples.csv has null values in the status column."
  **Input/Output Modes**: text/plain → text/plain

- **Skill ID**: `skill_discovery`
  **Name**: Skill Discovery and Installation
  **Description**: Search external skill catalogs cloned + indexed at init (Genesis, Anthropic, K-Dense, Composio, and others); install skills into the project for native auto-discovery; save agent-authored skills.
  **Tags**: skills, catalog, mcp
  **Examples**: "Search the skill catalog for a literature-search skill and install it."
  **Input/Output Modes**: text/plain → text/plain, text/markdown

### Tools and permissions

All 20 tools live on the single `dsagt-server` (stdio), split across four concerns.

**Registry (8):**

- `search_registry` — semantic search over registered + bundled code specs. Side effects: reads data.
- `get_registry` — list every registered code with its MCP-compatible schema. Side effects: reads data.
- `save_code_spec` — register a code as `codes/<name>/SKILL.md` (executable auto-wrapped with `dsagt-run` + `uv run --with`). Side effects: writes to the project dir; indexes into ChromaDB.
- `install_dependencies` — install a code's Python dependencies via uv. Side effects: executes uv, network calls (PyPI).
- `run_command` — execute a shell command with a timeout. Side effects: executes subprocesses.
- `read_file` — read a file from disk. Side effects: reads data.
- `http_request` — issue an HTTP(S) request. Side effects: network calls.
- `reconstruct_pipeline` — render `trace_archive/` as a dependency-ordered execution history. Side effects: reads data.

**Knowledge (5):**

- `kb_search` — hybrid semantic search over one or more collections (optional metadata, regex, and substring filters). Side effects: reads data.
- `kb_ingest` — index a file or directory into a named collection (background job for large corpora). Side effects: reads sources, writes `<project>/kb_index/`.
- `kb_append` — add documents to an existing collection (background job). Side effects: writes `<project>/kb_index/`.
- `kb_list_collections` — list collections with document counts. Side effects: reads data.
- `kb_job_status` — poll a background ingest/append job. Side effects: none.

**Memory (2):**

- `kb_remember` — save a user-confirmed fact to explicit memory. Side effects: writes `<project>/.dsagt/explicit_memories.yaml` + ChromaDB.
- `kb_get_memories` — retrieve explicit memories (optionally query-filtered). Side effects: reads data.

**Skills (5):**

- `search_skills` — rank installable skills across synced external catalogs. Side effects: reads data.
- `install_skill` — copy a catalog skill into `<project>/skills/` (with upstream attribution). Side effects: writes to the project dir.
- `save_skill` — register an agent-authored skill into `<project>/skills/`. Side effects: writes to the project dir.
- `add_skill_source` — clone + index a new external skill catalog. Side effects: network calls (git), writes `kb_index/`.
- `list_skill_sources` — list known/synced catalog sources. Side effects: reads data.

### Service endpoint and discovery

- Base URL: `https://github.com/AI-ModCon/dsagt`
- The MCP server runs as a local subprocess; there is no remote HTTP endpoint.
- Server: `dsagt-server` (stdio), self-sufficient — it derives the project from its working directory (`.dsagt/config.yaml`).

## Runtime Infrastructure

DSAgt runs locally as a CLI tool. The MCP server is launched as a subprocess by the configured agent platform. There are no services to run: all self-logging goes to a serverless SQLite MLflow store (`sqlite:///<project>/mlflow.db`), browsable on demand with `mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db`.

### Hardware

Runs on any developer workstation or compute node with Python 3.12+. The default embedding backend is CPU-only (no GPU required). Tested on macOS (arm64, x86_64) and Linux (x86_64).

### Software

Python 3.12 or 3.13, `uv` package manager. Key dependencies:

- `mcp>=1.0.0` — MCP server framework
- `mlflow==3.11.1` — trace store and observability (serverless SQLite backend)
- `chromadb>=1.5.1` — vector store
- `sentence-transformers==5.4.0` — local embeddings and reranking
- `llama-index-core>=0.11` — document and code chunking
- `rank-bm25>=0.2.2` — sparse keyword retrieval for hybrid search
- `questionary>=2.0` — interactive `dsagt init` menus

See `pyproject.toml` for the complete dependency set.

## Papers and Scientific Outputs

N/A — no associated publication at this time.

## Agent License

Apache-2.0

## Contact Info and Card Authors

See https://github.com/AI-ModCon/dsagt/graphs/contributors

# Intended Uses

## Intended Use

DSAgt is intended to assist researchers and data engineers in building, documenting, and reproducing scientific data pipelines using AI agent platforms. Developed under the DOE Genesis Mission, it helps researchers prepare and evaluate AI-ready scientific data through an agent.

### Primary Intended Users

Researchers and data engineers working on scientific data pipelines, particularly in domains like genomics, structural biology, materials science, and other areas with complex CLI toolchains.

### Mission Relevance

DSAgt was developed as part of the DOE Genesis Mission to enable AI-ready scientific data preparation.
Tested Use cases include:

- Microbial genomics pipelines (short-read QC and assembly)
- Cryo-EM data curation (EMPIAR datasets)
- Materials science DFT workflows (VASP via ISAAC)
- Tokamak stability analysis (fusion energy, M3D-C1)
- AI data readiness assessment (AIDRIN)

## Out-of-Scope Use Cases

- Using DSAgt on controlled or proprietary data that should not be shared with the configured LLM inference provider.

# How to use

## Install Instructions

```bash
# For use:
pip install "git+https://github.com/AI-ModCon/dsagt.git"

# For development:
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync --all-groups
source .venv/bin/activate

# Create a project (interactive: pick agent platform, knowledge
# collections, and skill-catalog sources; the knowledge base is
# provisioned on first run):
dsagt init
```

## Agent configuration

- **System prompt / instructions**: generated by `dsagt init` as `CLAUDE.md` (Claude Code), `AGENTS.md` (Codex/opencode), `.goosehints` (Goose), or `.clinerules/dsagt_instructions.md` (Cline)
- **MCP server config**: generated by `dsagt init` as `.mcp.json` (Claude Code), `goose.yaml`, `.codex-data/config.toml`, `opencode.json`, or via `cline mcp add`
- **LLM provider auth**: owned entirely by the agent platform (BYOA) — configure the agent before pointing DSAgt at it; DSAgt never stores or proxies credentials
- **Embedding backend**: set `embedding.backend: api` in `.dsagt/config.yaml` to use an OpenAI-compatible hosted embedder; the key comes from `EMBEDDING_API_KEY` in the shell (never on disk)
- **Reranking**: set `knowledge.rerank: true` in `.dsagt/config.yaml`
- **Episodic memory**: opt in at init (`dsagt init --episodic` or the interactive prompt)

## Invocation / integration

```bash
# Launch the agent from the project directory (no env exports, no services):
cd ~/dsagt-projects/my-project && claude    # …or goose / codex / opencode / cline

# Or let dsagt own the launch + post-session extraction trigger:
dsagt start my-project
```

The agent discovers DSAgt tools via MCP and can invoke `search_registry`, `kb_search`, `save_code_spec`, etc. directly in the conversation. Registered codes and installed skills also appear in the agent's native skills directory after the next `dsagt start`.

# Code snippets of how to use the agent

```bash
# Full quickstart (see README for step-by-step prompts):
dsagt init                                  # interactive; name it "quickstart", pick agent + collections
dsagt start quickstart

# After the session:
dsagt info quickstart                       # config + session/trace summary
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db

# Non-interactive smoke test (asserts every artifact):
dsagt smoke-test --agent claude
```

```python
# The DSAgt MCP server is invoked by the agent platform, not directly from Python.
# To integrate programmatically, use the MCP Python SDK:
# from mcp.client.stdio import StdioServerParameters
# Server command: dsagt-server (run from the project directory).
```

# Limitations

## Risks

DSAgt executes arbitrary CLI codes registered by the agent. The registry wraps code invocations with `dsagt-run` for provenance, but does not sandbox or restrict what commands the agent can register or execute. Users should review code specs before registration and restrict filesystem access as appropriate.

### Agent-specific risk notes (tool use)

- **Code execution side effects**: Registered codes can read/write files, make network calls, and execute arbitrary subprocesses. The agent must be trusted to register only appropriate codes.
- **Prompt injection**: Knowledge base documents and installed catalog skills are retrieved and injected into the agent context; malicious content in indexed documents or third-party skill catalogs could influence agent behavior.
- **Secrets handling**: No credentials are written to disk by DSAgt. A hosted embedding backend reads `EMBEDDING_API_KEY` from the shell at runtime.
- **Data exfiltration**: If a hosted embedding backend is configured, document chunks are sent to that external service during ingestion and search.

## Limitations

- Local-first: designed for single-user local or HPC use; no multi-user access control
- Embedding model quality: default local `sentence-transformers` model (~130 MB) is effective for general text but may underperform on highly domain-specific technical corpora
- Agent LLM-call traces are recovered post-hoc from the agent's on-disk transcript (uniform across all five platforms) — recovery granularity follows what each platform records
- Cline batch mode is unsupported (cline's provider rewrites unrecognized model names); interactive cline use works
- No GUI: all interaction is through the agent CLI or the MLflow web UI

# Agent evaluation details

- **Smoke test**: `dsagt smoke-test --agent <platform>` runs two full agent sessions non-interactively and asserts 18 artifacts: code registration + execution provenance, knowledge ingest + retrieval, skill catalog install, native skill mirroring, explicit + episodic memory, cross-session recall, agent-trace recovery, and session state
- **Unit tests**: `uv run python -m pytest tests/` (~640 tests; integration tests requiring credentials live in `test_*_integration.py`)
- **Code-call correctness**: verified by checking `trace_archive/` records for expected exit codes and captured output
- **Knowledge base precision**: evaluated via retrieval assertions in the smoke test (the agent must answer from ingested docs)

# More Information

- Full documentation: https://ai-modcon.github.io/dsagt/
- Source code: https://github.com/AI-ModCon/dsagt
- Use case walkthroughs: `use_cases/` (Microbial Isolates, Cryo-EM, ISAAC/VASP, Tokamak Stability, AIDRIN)
