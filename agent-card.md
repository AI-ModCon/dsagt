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
base_model: N/A  # DSAgt is agent-platform-agnostic; it wraps Claude Code, Goose, Codex, opencode, Roo Code, or Cline

datasets:
    - # NeMo Curator reference corpus (indexed into global KB by dsagt setup-kb)
    - # AI Data Readiness Inspector (AIDRIN) reference corpus

metrics:
    - # Tool registration success rate
    - # Knowledge base search precision
    - # Provenance trace completeness

agent_card:
  name: "DSAgt (DataSmith Agent)"
  description: "AI-assisted data pipeline builder developed under the DOE Genesis Mission. Wraps an MCP-compatible agent CLI with tool registration, a semantic knowledge base, execution provenance, and observability infrastructure to accelerate AI-ready scientific data preparation."
  provider:
    organization: "DOE AI ModCon Base Data Team (DOE Genesis Mission)"
    url: "https://github.com/AI-ModCon/dsagt"
  version: "0.1.0"
  documentation_url: "https://ai-modcon.github.io/dsagt/"
  protocol_version: "N/A"  # DSAgt extends existing agent CLIs via MCP; it is not itself an A2A service
  preferred_transport: "HTTP" 
  capabilities:
    streaming: false
    push_notifications: false
    state_transition_history: true  # trace_archive + MLflow record full execution history

authentication:
  schemes:
    - "Bearer"  # used by the optional LiteLLM proxy and hosted embedding backends
  credentials: "N/A"  # public repo; agent CLI auth is handled by the underlying platform (Claude, Goose, etc.)

  default_input_modes:
    - "text/plain"
  default_output_modes:
    - "text/plain"
    - "application/json"  # tool execution records, pipeline reconstructions

  skills:
    - id: "tool_registration"
      name: "Tool Registration"
      description: "Register CLI tools as markdown files with YAML frontmatter; install dependencies via uv; wrap executions with dsagt-run for provenance."
      tags: [registry, provenance, mcp]
      examples:
        - "Register csvkit tools csvcut, csvgrep, csvstat, csvlook."
        - "Install Python dependencies for a custom analysis script."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "application/json"]

    - id: "knowledge_base"
      name: "Knowledge Base"
      description: "Hybrid dense+sparse semantic search over six ChromaDB collections: Tool Specs, Skills, Domain Knowledge, Explicit Memory, Episodic Memory, and Tool Use Records."
      tags: [knowledge, chromadb, semantic-search, mcp]
      examples:
        - "Ingest domain documentation into a named collection."
        - "Search for tools matching 'CSV statistics'."
        - "Save a user-confirmed fact to explicit memory."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "application/json"]

    - id: "provenance"
      name: "Execution Provenance"
      description: "Record every tool invocation (command, args, exit code, duration, file counts, stderr) to trace_archive/ and emit OTLP spans to MLflow. Reconstruct the full execution history as a bash script or Snakemake workflow."
      tags: [provenance, mlflow, otlp, reproducibility]
      examples:
        - "Reconstruct a reproducible pipeline from prior tool executions."
        - "View tool execution traces in the MLflow UI."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "application/x-sh", "text/x-snakemake"]

    - id: "episodic_memory"
      name: "Episodic Memory Distillation"
      description: "Distill new MLflow traces into per-category episodic memory using embedding-centroid outlier detection. Surfaces novel facts from prior sessions."
      tags: [memory, mlflow, distillation]
      examples:
        - "Distill today's session traces into episodic memory."
      input_modes: ["text/plain"]
      output_modes: ["text/plain"]

    - id: "datacard_generator"
      name: "Datacard Generator"
      description: "Bundled skill workflow for generating standardized dataset documentation (datacards) from indexed knowledge and agent-driven analysis."
      tags: [datacard, documentation, skill]
      examples:
        - "Generate a datacard for samples.csv."
      input_modes: ["text/plain"]
      output_modes: ["text/plain", "text/markdown"]

Extensions:
  agent_runtime:
    framework: "MCP (Model Context Protocol) over stdio; supported agent platforms: Claude Code, Goose, Codex, opencode, Roo Code, Cline"
    service_endpoint: "stdio (MCP servers launched as subprocesses by the agent platform)"
    rate_limits: "Determined by the underlying LLM provider and optional LiteLLM proxy configuration."
    logging: "OTLP HTTP spans emitted to local MLflow instance (http://localhost:<mlflow_port>/v1/traces); full tool execution records written to <project>/trace_archive/"
    memory: "Stateful per-project. Explicit memory: <project>/explicit_memories.yaml + ChromaDB. Episodic memory: ChromaDB (distilled by dsagt memory). Tool use records: <project>/trace_archive/ + ChromaDB."

---

# DSAgt (DataSmith Agent)

DSAgt is an AI-assisted data pipeline builder. It connects an MCP-compatible agent CLI (Claude Code, Goose, Codex, opencode, Roo Code, or Cline) to tool registration, a semantic knowledge base, execution provenance, and observability infrastructure — without modifying the agent itself.

*Last Updated*: **2026-06-30**

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

+ **2026-06-30** initial public version (v0.1.0)

## Agent short description

Tool-using scaffolding layer that gives any MCP-compatible agent CLI persistent tool registration, semantic knowledge retrieval, execution provenance, and session memory — exposed via two MCP servers (Registry and Knowledge).

## Agent description

DSAgt wraps an unmodified agent CLI with four independently-operable layers, each implemented as an MCP server the agent discovers through the standard MCP tool protocol:

1. **Tool Registry** — The agent registers CLI tools as markdown files with YAML frontmatter; the registry server handles dependency installation (`uv run --with`) and wraps each invocation with `dsagt-run` for provenance. Discovery is via `search_registry`.
2. **Knowledge Base** — Six independently-partitioned ChromaDB collections with hybrid dense (sentence-transformers) + sparse (BM25) search. Global collections (Tool Specs, Skills, Domain Knowledge) are populated by `dsagt setup-kb`; per-project collections (Explicit Memory, Episodic Memory, Tool Use Records) fill in during use. Discovery is via `kb_search`, `kb_ingest`, `kb_remember`, `search_skills`.
3. **Provenance** — `dsagt-run` captures every tool execution (command, args, exit code, duration, file I/O, stderr) to `trace_archive/` and emits OTLP spans to MLflow. `reconstruct_pipeline` renders the archive as a reproducible bash or Snakemake script.
4. **Observability** — Local MLflow instance with OTLP ingestion. All four layers emit spans. The agent's own LLM-call traces land in the same store when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (printed by `dsagt init`).

The data layer is agent-platform-agnostic: switching platforms preserves all accumulated knowledge, tools, and traces.

## Underlying model(s)

- Primary model(s): N/A — DSAgt is platform-agnostic and delegates LLM calls to the configured agent CLI (Claude, GPT-4o, etc.)
- Embedding model: `sentence-transformers` (local, CPU-side, default); optionally any LiteLLM-compatible hosted embedder
- Cross-encoder reranking: optional, enabled via `knowledge.rerank: true` in `dsagt_config.yaml`

## Inputs and outputs

### Default interaction modes

- defaultInputModes: `["text/plain"]`
- defaultOutputModes: `["text/plain", "application/json"]`

The agent accepts natural-language instructions (text). Outputs include text responses, registered tool specs (markdown), execution trace records (JSON), pipeline reconstructions (bash / Snakemake), and datacard documents (markdown).

### Skills

- **Skill ID**: `tool_registration`
  **Name**: Tool Registration
  **Description**: Register CLI tools as markdown+YAML specs; install dependencies; wrap executions with `dsagt-run` for provenance.
  **Tags**: registry, provenance, mcp
  **Examples**: "Register csvkit tools csvcut, csvgrep, csvstat, csvlook.", "Install Python dependencies for a custom analysis script."
  **Input/Output Modes**: text/plain → text/plain, application/json

- **Skill ID**: `knowledge_base`
  **Name**: Knowledge Base
  **Description**: Hybrid semantic search (dense + BM25) over Tool Specs, Skills, Domain Knowledge, Explicit Memory, Episodic Memory, and Tool Use Records collections.
  **Tags**: knowledge, chromadb, semantic-search, mcp
  **Examples**: "Ingest domain documentation into a named collection.", "Search for tools matching 'CSV statistics'.", "Save a user-confirmed fact to explicit memory."
  **Input/Output Modes**: text/plain → text/plain, application/json

- **Skill ID**: `provenance`
  **Name**: Execution Provenance
  **Description**: Record every tool invocation to `trace_archive/` + MLflow; reconstruct full execution history as a bash script or Snakemake workflow.
  **Tags**: provenance, mlflow, otlp, reproducibility
  **Examples**: "Reconstruct a reproducible pipeline from prior tool executions."
  **Input/Output Modes**: text/plain → text/plain, application/x-sh

- **Skill ID**: `episodic_memory`
  **Name**: Episodic Memory Distillation
  **Description**: Distill MLflow traces into per-category episodic memory using embedding-centroid outlier detection (`dsagt memory --project <name>`).
  **Tags**: memory, mlflow, distillation
  **Examples**: "Distill today's session traces into episodic memory."
  **Input/Output Modes**: text/plain → text/plain

- **Skill ID**: `datacard_generator`
  **Name**: Datacard Generator
  **Description**: Bundled skill workflow (`src/dsagt/skills/datacard-generator/`) for generating standardized dataset documentation.
  **Tags**: datacard, documentation, skill
  **Examples**: "Generate a datacard for samples.csv."
  **Input/Output Modes**: text/plain → text/markdown

### Tools and permissions

- Tool: `search_registry` (Registry MCP server)
  - Purpose: Semantic search over registered tool specs
  - Inputs: query string, optional tag filters
  - Outputs: ranked list of matching tool specs
  - Side effects: reads data
  - Required permissions: none

- Tool: `save_tool_spec` (Registry MCP server)
  - Purpose: Register a new CLI tool as a markdown file with YAML frontmatter
  - Inputs: tool name, command, dependencies, tags, usage description
  - Outputs: path to written tool spec file
  - Side effects: writes data to `<project>/tools/`
  - Required permissions: filesystem write access to project directory

- Tool: `install_dependencies` (Registry MCP server)
  - Purpose: Install tool dependencies via `uv run --with`
  - Inputs: list of package names
  - Outputs: installation confirmation
  - Side effects: executes jobs (uv), network calls (PyPI)
  - Required permissions: network access, uv installed

- Tool: `reconstruct_pipeline` (Registry MCP server)
  - Purpose: Render the trace archive as a reproducible bash script or Snakemake workflow
  - Inputs: optional time range or session filter
  - Outputs: bash or Snakemake script
  - Side effects: reads data from `<project>/trace_archive/`
  - Required permissions: none

- Tool: `kb_search` (Knowledge MCP server)
  - Purpose: Hybrid semantic search over one or more knowledge collections
  - Inputs: query string, collection name(s), optional filters
  - Outputs: ranked document chunks with metadata
  - Side effects: reads data
  - Required permissions: none

- Tool: `kb_ingest` (Knowledge MCP server)
  - Purpose: Index a file or directory into a named ChromaDB collection (runs in background for large corpora)
  - Inputs: file path or directory, collection name
  - Outputs: ingestion job ID; status queryable
  - Side effects: reads data, writes data to `<project>/kb_index/`
  - Required permissions: filesystem read access to source files

- Tool: `kb_remember` (Knowledge MCP server)
  - Purpose: Save a user-confirmed fact to explicit memory
  - Inputs: fact string
  - Outputs: confirmation
  - Side effects: writes data to `<project>/explicit_memories.yaml` and ChromaDB
  - Required permissions: none

- Tool: `kb_get_memories` (Knowledge MCP server)
  - Purpose: Retrieve explicit and episodic memories for the current project
  - Inputs: optional query string
  - Outputs: list of memory entries
  - Side effects: reads data
  - Required permissions: none

- Tool: `search_skills` (Knowledge MCP server)
  - Purpose: Discover agent skill workflows
  - Inputs: query string
  - Outputs: ranked list of matching skills with SKILL.md content
  - Side effects: reads data
  - Required permissions: none

### Service endpoint and discovery

- Base URL: `https://github.com/AI-ModCon/dsagt`
- MCP servers run as local subprocesses; there is no remote HTTP endpoint by default.
- Registry server: `dsagt-registry-server` (stdio)
- Knowledge server: `dsagt-knowledge-server` (stdio)
- Optional LiteLLM proxy: `dsagt-proxy` (HTTP, for routing embeddings/LLM calls to hosted providers)

## Runtime Infrastructure

DSAgt runs locally as a CLI tool. Both MCP servers are launched as subprocesses by the configured agent platform. MLflow runs locally at a port pinned at `dsagt init` time.

### Hardware

Runs on any developer workstation or compute node with Python 3.12+. The default embedding backend is CPU-only (no GPU required). Tested on macOS (arm64, x86_64) and Linux (x86_64).

### Software

Python 3.12+, `uv` package manager. Key dependencies:

- `mcp>=1.0.0` — MCP server framework
- `chromadb>=1.5.1` — vector store
- `faiss-cpu>=1.8` — FAISS index backend
- `sentence-transformers==5.4.0` — local embedding model
- `llama-index-core>=0.11` — document chunking and indexing
- `mlflow>=3.11.1,<4.0` — observability and trace storage
- `opentelemetry-*>=1.27` — OTLP span emission
- `litellm[proxy]>=1.83.7` — optional hosted embedder / LLM proxy routing
- `rank-bm25>=0.2.2` — sparse keyword retrieval for hybrid search

```txt
# To get full dependency list:
uv sync --all-groups
pip freeze
```

See `pyproject.toml` for the complete pinned dependency set.

## Papers and Scientific Outputs

N/A — no associated publication at this time.

## Agent License

Apache-2.0

## Contact Info and Card Authors

See https://github.com/AI-ModCon/dsagt/graphs/contributors

# Intended Uses

## Intended Use

DSAgt is intended to assist researchers and data engineers in building, documenting, and reproducing scientific data pipelines using AI agent platforms. Developed under the DOE Genesis Mission, DSAgt assists in AI-LLM assisted preparation and evaluation of AI-Ready data using an agentic harness.

### Primary Intended Users

Researchers and data engineers working on scientific data pipelines, particularly in domains like genomics, structural biology, materials science, and other areas with complex CLI toolchains.

### Mission Relevance

DSAgt was developed as part of the DOE Genesis Mission to enable AI-ready scientific data preparation.
Tested Use cases include:

- Microbial genomics pipelines (short-read QC and assembly)
- Cryo-EM data curation (EMPIAR datasets)
- Materials science DFT workflows (VASP via ISAAC)
- Tokamak stability analysis (fusion energy)
- Combustion flow simulations (BlastNet)

## Out-of-Scope Use Cases

- Using DSAgt on controlled or proprietary data that should not be shared with configured LLM inference provider. 

# How to use

## Install Instructions

```bash
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync
source .venv/bin/activate

# Build shared knowledge base collections (Tool Specs, Skills, Domain Knowledge):
dsagt setup-kb

# Create a project:
dsagt init my-project --agent claude  # or goose / codex / opencode / roo / cline
```

## Agent configuration

- **System prompt / instructions**: generated by `dsagt init` as `CLAUDE.md` (Claude Code), `AGENTS.md` (Codex/opencode), `.goosehints` (Goose), or `.roomodes` (Roo Code)
- **MCP server config**: generated by `dsagt init` as `.mcp.json` (Claude Code), `goose.yaml`, `.codex-data/config.toml`, `opencode.json`, or `.roo/mcp.json`
- **Embedding backend**: set `embedding.backend: api` in `<project>/dsagt_config.yaml` to route through a hosted embedder via LiteLLM; provide `embedding.base_url` and `embedding.api_key`
- **Reranking**: set `knowledge.rerank: true` in `dsagt_config.yaml`
- **Observability**: `dsagt mlflow <name>` prints `OTEL_EXPORTER_OTLP_ENDPOINT` and `MLFLOW_TRACKING_URI` exports for the agent's own traces

## Invocation / integration

```bash
# Start MLflow and get OTel exports:
dsagt mlflow my-project

# Paste the printed export block, then launch the agent from the project directory:
cd ~/dsagt-projects/my-project && claude
```

The agent discovers DSAgt tools via MCP and can invoke `search_registry`, `kb_search`, `save_tool_spec`, etc. directly in the conversation.

# Code snippets of how to use the agent

```bash
# Full quickstart (see README for step-by-step prompts):
export SMOKE_DIR="$(pwd)/tests/smoke_test"
dsagt init quickstart --agent claude
dsagt mlflow quickstart
# paste printed exports, then:
cd ~/dsagt-projects/quickstart && claude

# After the session:
dsagt memory --project quickstart
dsagt stop quickstart
dsagt info quickstart

# Non-interactive smoke test:
dsagt smoke-test --agent claude
```

```python
# DSAgt MCP servers are invoked by the agent platform, not directly from Python.
# To integrate programmatically, use the MCP Python SDK:
# from mcp.client.stdio import StdioServerParameters
# See docs/mcp-servers.md for server command and tool schemas.
```

# Limitations

## Risks

DSAgt executes arbitrary CLI tools registered by the agent. The registry server wraps tool invocations with `dsagt-run`, but does not sandbox or restrict what commands the agent can register or execute. Users should review tool specs before registration and restrict filesystem access as appropriate.

### Agent-specific risk notes (tool use)

- **Tool execution side effects**: Registered tools can read/write files, make network calls, and execute arbitrary subprocesses. The agent must be trusted to register only appropriate tools.
- **Prompt injection**: Knowledge base documents are retrieved and injected into the agent context; malicious content in indexed documents could influence agent behavior.
- **Secrets handling**: API keys for hosted embedding backends are stored in `dsagt_config.yaml`. Do not commit this file to version control if it contains secrets. The LiteLLM proxy credential file is similarly sensitive.
- **Data exfiltration**: If a hosted embedding backend is configured, document chunks are sent to that external service during ingestion and search.

## Limitations

- Local-first: designed for single-user local or HPC use; no multi-user access control
- Embedding model quality: default local `sentence-transformers` model (~130 MB) is effective for general text but may underperform on highly domain-specific technical corpora
- Agent OTel coverage varies: Claude Code and Goose emit full LLM-call traces; Codex emits token counts and tool names; opencode emits nothing natively
- Memory distillation requires MLflow traces to be present; sessions without MLflow running will not generate episodic memories
- No GUI: all interaction is through the agent CLI or the MLflow web UI

# Agent evaluation details

- **Smoke test**: `dsagt smoke-test --agent <platform>` runs the full quickstart non-interactively and asserts all artifacts (tool specs, trace records, explicit memory) are produced
- **Unit and integration tests**: `pytest tests/` (requires `uv sync --all-groups`); integration tests marked with `@pytest.mark.integration` can be deselected with `-m 'not integration'`
- **Tool-call correctness**: verified by checking `trace_archive/` records for expected exit codes and file counts
- **Knowledge base precision**: evaluated informally via `kb_search` recall in the smoke test

# More Information

- Full documentation: https://ai-modcon.github.io/dsagt/
- Source code: https://github.com/AI-ModCon/dsagt
- Architecture diagram: `docs/assets/architecture.png`
- Use case walkthroughs: `use_cases/` (Microbial Isolates, Cryo-EM, ISAAC/VASP, Tokamak Stability, Combustion)
