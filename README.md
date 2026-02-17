# DSAGT

**D**ata **S**cience **A**gent **T**oolkit — AI-assisted data pipeline builder.

DSAGT connects an MCP-compatible AI agent to three servers for building scientific data pipelines:

1. **Pipeline Server** — Runs registered tools, logs provenance. Supports general-purpose processing and American Science Cloud targets; extendable to domain-specific workflows.
2. **Registry Builder** — Analyzes CLI tools, documentation, and APIs to generate and store tool specifications.
3. **Knowledge Base** — Semantic search over indexed document collections (FAISS + optional cross-encoder reranking).

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

This installs three CLI entry points:

- `dsagt-pipeline-server`
- `dsagt-registry-server`
- `dsagt-knowledge-server`

## How It Works

### Tool Registration

The default registry ships with general-purpose data tools. To register additional tools, use the registry server through your agent:

1. Point the agent at a CLI tool, its `--help` output, or documentation
2. The agent uses the registry builder to analyze the interface (it can read files, fetch URLs, and run commands)
3. The agent proposes a tool spec and saves it via `save_tool_spec`
4. The tool is immediately available through the pipeline server

Registry builder tools exposed to the agent:

- `read_file` — Read local files
- `http_request` — Fetch URLs
- `run_command` — Execute commands (e.g., `tool --help`)
- `save_tool_spec` — Save a tool specification to the registry
- `get_registry` — List all registered tools
- `search_registry` — Search tools by name or description

### Pipeline Execution

1. The pipeline server copies the base `registry.yaml` to `runtime/registry.yaml` at startup
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

### Pipeline Server

```bash
# Default bundled registry
uv run dsagt-pipeline-server

# Custom registry from a previous session
uv run dsagt-pipeline-server --registry my_registry.yaml

# Specify registry file and runtime directory
uv run dsagt-pipeline-server --registry path/to/registry.yaml --runtime-dir ./my_session
```

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
User: I have a script at scripts/preprocess.py that cleans CSV files.
      Register it as a pipeline tool.

Agent: [reads the file, runs --help, proposes a spec]
       Registered "preprocess" with parameters for input_file,
       output_file, and --drop-nulls. Want to try it?

User: Run it on data/raw.csv

Agent: [executes via pipeline server]
       Output written to data/cleaned.csv. 142 rows processed,
       3 null rows dropped.
```

## Smoke Test

`tests/smoke_test/` contains fixtures for verifying all three servers end-to-end. Knowledge base steps require an embedding API key; skip them if you don't have one.

```
tests/smoke_test/
├── greet.py                  # Simple CLI tool to register and execute
└── knowledge/                # Documents for KB ingestion
    ├── DESCRIPTION.md
    ├── installation.md
    ├── api_reference.md
    └── troubleshooting.md
```

### 1. Verify the test script

```bash
uv run python tests/smoke_test/greet.py World
```

Expected output: `{"message": "Hello, World!", "status": "ok"}`

### 2. Start a session with all three servers

Follow `agents/<platform>/README.md` to launch a session. The knowledge server runs without reranking by default; add `--rerank` to enable cross-encoder reranking (triggers model download on first use).

Without an embedding API key, omit the knowledge server and skip steps 5–6.

### 3. Register a tool

```
Register tests/smoke_test/greet.py as a pipeline tool.
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

## Project Structure

```
├── src/dsagt/
│   ├── __init__.py
│   ├── mcp_utils.py                # Shared MCP server utilities
│   ├── registry.py                 # Tool registry management
│   ├── registry.yaml               # Default tool registry (bundled)
│   ├── knowledge.py                # Semantic search over document collections
│   ├── pipeline_server.py          # MCP server: tool execution
│   ├── registry_server.py          # MCP server: tool registration
│   └── knowledge_server.py         # MCP server: knowledge base search
├── agents/
│   ├── goose/                      # Goose agent config and quickstart
│   ├── roo/                        # Roo Code (VS Code) config and quickstart
│   └── claude-code/                # Claude Code config and quickstart
├── tests/
│   ├── test_registry.py
│   ├── test_registry_server.py
│   ├── test_knowledge_base.py
│   ├── test_knowledge_server.py
│   ├── test_knowledge_integration.py   # Requires API key
│   └── smoke_test/
│       ├── greet.py
│       └── knowledge/
├── scripts/
│   └── setup_core_kb.py
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
uv run which dsagt-pipeline-server
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

## License

TBD