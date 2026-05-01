# DSAGT

**D**ata **S**cience **A**gent **T**oolkit — AI-assisted data pipeline builder.

DSAGT connects an MCP-compatible AI agent to MCP services for building scientific data pipelines:

1. **Registry Builder** — Analyzes CLI tools, documentation, and APIs to generate and store tool specifications.
2. **Knowledge Base** — Semantic search over indexed document collections (FAISS + optional cross-encoder reranking).

The servers are platform-agnostic and communicate over MCP stdio.

## Installation

### Prerequisites

- Python 3.10–3.13
- [uv](https://github.com/astral-sh/uv) — required for portable MCP server configs across agent platforms
- An MCP-compatible agent (see [Agent Setup](#agent-setup))

### Install

```bash
git clone <repository-url>
cd dsagt
uv sync --all-groups
```

This installs the CLI entry points:

- `dsagt-registry-server`
- `dsagt-knowledge-server`

## How It Works

### Tool Registration

The default registry ships with general-purpose data tools. To register additional tools, use the registry server through your agent:

1. Point the agent at a CLI tool, its `--help` output, or documentation
2. The agent uses the registry builder to analyze the interface (it can read files, fetch URLs, and run commands)
3. The agent proposes a tool spec and saves it via `save_tool_spec`
4. The tool is immediately available for agent use

Registry builder tools exposed to the agent:

- `read_file` — Read local files
- `http_request` — Fetch URLs
- `run_command` — Execute commands (e.g., `tool --help`)
- `save_tool_spec` — Save a tool specification to the registry
- `get_registry` — List all registered tools
- `search_registry` — Search tools by name or description

### Tool Execution

1. A session copies the base `registry.yaml` to `runtime/registry.yaml`
2. The agent runs tools from the session registry
3. Each execution is logged to a provenance file

### Knowledge Base

The knowledge base provides semantic search over indexed document collections using FAISS, with optional cross-encoder reranking.

To set up the core collections (NeMo Curator, AIDRIN):

```bash
export LLM_API_KEY="your-api-key"
uv run python scripts/setup_core_kb.py
```

This clones repos, downloads papers, chunks content, and builds FAISS indexes in `kb_index/`. Run `python scripts/setup_core_kb.py --help` for options.

## Running the Servers

Note: This is typically done through your MCP config, so you don't need to run these commands explicitly

### Registry Builder Server

```bash
# Default: writes to ./runtime/registry.yaml
uv run dsagt-registry-server

# Custom registry path
uv run dsagt-registry-server --registry path/to/registry.yaml
```

### Knowledge Base Server

```bash
# Defaults: base=./kb_index, runtime=./runtime
uv run dsagt-knowledge-server

# Custom directories, with reranking
uv run dsagt-knowledge-server --base-index-dir path/to/kb_index --runtime-dir ./runtime --rerank
```

## Agent Setup

DSAGT works with any MCP-compatible agent. The servers are identical across platforms; only the agent configuration format differs.

Platform-specific configs and quickstart guides live in `agents/`:

- **Goose**: [`agents/goose/README.md`](agents/goose/README.md)
- **Roo Code** (VS Code): [`agents/roo/README.md`](agents/roo/README.md)
- **Claude Code**: [`agents/claude-code/README.md`](agents/claude-code/README.md)
- **Cline** (VS Code): [`agents/cline/README.md`](agents/cline/README.md)

### Path Considerations

Servers use relative paths by default (`--runtime-dir ./runtime`, `--base-index-dir ./kb_index`), resolved from the agent's working directory. This works as long as the agent launches from the DSAGT project root.

If your agent launches from elsewhere, use absolute paths:

```bash
uv run dsagt-knowledge-server --base-index-dir /absolute/path/to/kb_index
```

### Example Session

```
User: I have a script at /bin/ls that reads directory contents.
      Register it as a pipeline tool.

Agent: [reads the file, runs --help, proposes a spec]
       Registered list_directory in the session registry. It wraps /bin/ls with parameters for path (required), long_format (-l), and show_hidden (-a). Registry now has 2 tools.

User: Run it on current directory

Agent: [executes the registered tool]
        Contents of /Users/shreyas/Dev/git/BaseData_pipeline_agent:
        LICENSE         README.md       agents
        data            dsagt_knowledge_server.log      kb_index
        pyproject.toml  runtime         scripts         src
        tests           use_cases       uv.lock
```

## Smoke Test

`tests/smoke_test/` contains the tracked fixtures for verifying the MCP services end-to-end. Knowledge base steps require an embedding API key; skip them if you don't have one.

```
tests/smoke_test/
├── greet.py                  # Simple CLI tool to register and execute
└── knowledge/                # Documents for KB ingestion
    ├── DESCRIPTION.md
    ├── api_reference.md
    ├── installation.md
    └── troubleshooting.md
```

### 1. Verify the test script

```bash
uv run python tests/smoke_test/greet.py World
```

Expected output: `{"message": "Hello, World!", "status": "ok"}`

### 2. Start a session with all MCP servers

Follow `agents/<platform>/README.md` to launch a session. The knowledge server runs without reranking by default; add `--rerank` to enable cross-encoder reranking (triggers model download on first use).

Without an embedding API key, omit the knowledge server and skip steps 5–6.

### 3. Register a tool

```
Register tests/smoke_test/greet.py as a tool.
Run "python tests/smoke_test/greet.py --help" to see its interface.
```

The agent should run `--help` via the registry builder, then call `save_tool_spec`.

### 4. Execute the tool

```
Run the greet tool with name "World" and greeting "Hi".
```

Expected: JSON output with `"message": "Hi, World!"`.

### 5. Ingest documents

```
Ingest the folder tests/smoke_test/knowledge into the knowledge base.
```

The agent should call `kb_ingest` and report file/chunk counts.

### 6. Search the knowledge base

```
Search the knowledge collection for "how to handle large files".
```

Top results should come from `troubleshooting.md` (lazy loading, OOM errors).

```
List all knowledge base collections.
```

Should show `knowledge` with the description from `DESCRIPTION.md`.

### 7. Verify artifacts

After the session, check from the project root:

**Registry** (should contain default tools plus `greet`):
```bash
cat runtime/registry.yaml
```

**Provenance log** (timestamped entry with tool name, args, full command):
```bash
cat runtime/provenance.log
```

**Knowledge base index**:
```bash
ls runtime/kb_index/knowledge/
# Expected: index.faiss, chunks.jsonl, DESCRIPTION.md
```

**Chunk format** (JSON objects with `id`, `text`, `metadata`):
```bash
head -3 runtime/kb_index/knowledge/chunks.jsonl
```

### Cleanup

```bash
rm -rf runtime
```

## Tool Registry Format

Tools are defined in YAML:

```yaml
tools:
  - name: tool_name
    description: What the tool does
    executable: command to run (e.g., "python script.py")
    dependencies:                        # optional
      - pandas>=2.0
      - scikit-learn
    parameters:
      param_name:
        type: string|integer|number|boolean|array|object
        required: true|false
        description: Parameter description
        default: optional_default_value
```

Required parameters are passed as positional arguments. Optional parameters use `--flag value` syntax.

When a tool spec includes `dependencies`, the registry server automatically installs them via `uv pip install` at registration time. Dependencies are stored in the registry YAML for reproducibility. Use `install_dependencies` to reinstall all deps from an existing registry (e.g., after setting up a fresh environment).

## Demo Folder

`demo/` contains a complete microbial isolate demonstration package:

- `demo/isolate_demo.md` — End-to-end demo instructions (asset collection, setup, prompts, and validation)
- `demo/isolate_session.txt` — Prompt script used to drive the demo interaction
- `demo/genomics.md` — Pipeline context document for knowledge ingestion
- `demo/fastp_megahit_best_practices.md` — Fastp/Megahit best-practices reference
- `demo/demoplan.md` — Short demo plan outline

Run the isolate demo by following `demo/isolate_demo.md` from the DSAGT project root.

## Project Structure

```
├── agents/
│   ├── Agents.md
│   ├── README.md
│   ├── claude-code/
│   │   ├── README.md
│   │   ├── claude_code_mcp.json
│   │   └── dsagt_instructions.md
│   ├── cline/
│   │   ├── README.md
│   │   ├── cline_mcp.json
│   │   └── dsagt_instructions.md
│   ├── goose/
│   │   ├── .goosehints
│   │   ├── README.md
│   │   └── goose.yaml
│   └── roo/
│       ├── README.md
│       ├── roo_mcp.json
│       └── roomodes
├── scripts/
│   └── setup_core_kb.py
├── src/dsagt/
│   ├── __init__.py
│   ├── knowledge.py
│   ├── knowledge_server.py
│   ├── mcp_utils.py
│   ├── registry.py
│   ├── registry_server.py
│   ├── skills/
│   │   └── scan_directory.md
│   └── tools/
│       └── scan_directory.py
├── tests/
│   ├── __init__.py
│   ├── smoke_test/
│   │   ├── greet.py
│   │   └── knowledge/
│   │       ├── DESCRIPTION.md
│   │       ├── api_reference.md
│   │       ├── installation.md
│   │       └── troubleshooting.md
│   ├── test_dependency_integration.py
│   ├── test_knowledge_base.py
│   ├── test_knowledge_integration.py
│   ├── test_knowledge_server.py
│   ├── test_registry.py
│   ├── test_registry_server.py
│   └── test_server_startup.py
├── use_cases/
│   ├── microbial_isolates/
│   │   ├── demoplan.md
│   │   ├── fastp_megahit_best_practices.md
│   │   ├── genomics.md
│   │   ├── isolate_demo.md
│   │   └── isolate_session.txt
│   └── tokamak_stability/
│       ├── hdf5.py
│       └── test_hdf5_tools.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## Tests

```bash
uv run pytest
```

Run specific tests:

```bash
uv run pytest tests/test_registry.py
uv run pytest tests/test_registry_server.py
uv run pytest tests/test_knowledge_server.py
uv run pytest tests/test_knowledge_base.py
uv run pytest tests/test_registry.py::TestCallTool::test_success -v
```

Registry tests mock `subprocess.run`. Server tests invoke MCP handlers directly (no stdio transport, no network). Knowledge base tests mock the embedding API and use FAISS on temp files. Knowledge server tests use async helpers for background ingest/append jobs.

## Troubleshooting

### MCP Server Not Found

```bash
uv run which dsagt-registry-server
uv run which dsagt-knowledge-server

# Reinstall if needed
uv sync --reinstall
```

### Tools Not Executing

Check what command was run:

```bash
cat runtime/provenance.log
```

Verify the executable path and that any interpreter (python, Rscript, etc.) is in your PATH.

### Registry File Not Found

Use absolute paths in configuration:

```yaml
args:
  - --registry
  - /full/path/to/registry.yaml
```

# DSAGT Development Plan (2026-03-19) 

Note: This plan was written on 2026-03-19 and maybe superceded by newer plans

## Motivation

DSAGT currently shows high run-to-run variance (even at temperature 0). The root issue appears to be limited structure in tool description, invocation, and management. This plan focuses on improving reliability, reproducibility, and observability.

## Track 1: Structured Tool Execution

- **Goal:** Reduce stochastic behavior with stronger execution structure.
- **Problem:** Routing tool execution through an MCP layer adds indirection and limits direct access to Unix return codes/process signals. Registry entries are too sparse for consistent tool selection and invocation.
- **Plan:**
  - Extend the registry so each tool is paired with a skill (usage patterns, expected I/O, invocation examples).
  - Use an agent-driven skill builder (in progress by Jean-Luca) to generate/refine tool skills as tools are registered.
  - Move tool execution to direct shell invocation so the agent can use return codes, stderr, and standard Unix process controls.

## Track 2: Container-Based Package Management

- **Goal:** Improve reproducibility and isolation of tool execution environments.
- **Problem:** Ad hoc dependency handling creates conflicts and weak cross-machine reproducibility.
- **Plan:**
  - Build a container-builder skill to generate Dockerized tool runtimes (dependencies, data paths, runtime config).
  - Evaluate the Docker Agent tool shared by Shreyas (assessment led by Andrew) for reuse or adaptation.

## Track 3: Observability and Logging

- **Goal:** Add structured telemetry for debugging, evaluation, and cost tracking.
- **Problem:** No consistent logs for agent decisions, tool calls, or token usage, which blocks diagnosis and improvement tracking.
- **Plan:**
  - Integrate OpenTelemetry for standards-based tracing/logging.
  - Add MLflow wrappers for experiment tracking and token-usage monitoring.

## Track 4: Resource management

- **Goal:** Add skills that estimate compute needs and prevent resource overcommit during pipeline execution.
- **Plan:**
  - Create a resource-assessment skill that probes tool behavior, learns feasible operating ranges under current constraints, and recommends alternatives when resources are insufficient.
