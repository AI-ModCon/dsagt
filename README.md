# DSAGT

**D**ata **S**mith **A**gent **T**oolkit — AI-assisted data pipeline builder.

DSAGT is a compartmentalized agent setup for building AI-ready scientific data pipelines. It pairs an MCP-compatible AI agent with three specialized servers and curated agent guidance, giving you a simple, extendable architecture:

1. **Pipeline Server** — General-purpose AI-ready data processing, evaluation, and shipping tools (including American Science Cloud targets), extendable to domain-specific processing workflows
2. **Registry Builder** — Generation and registry of domain-specific bespoke processing tools by analyzing documentation, help output, and API specs
3. **Knowledge Base** — Base knowledge for AI-ready scientific data practices, extendable to domain-specific concerns and existing vector databases

Each agent platform gets its own directory under `agents/` containing config files and curated instructions that condition the agent's behavior. The core servers are platform-agnostic — all three speak MCP over stdio — so the same pipeline, registry, and knowledge base work identically regardless of which agent you use. The result is a clean separation: servers provide capability, agent guidance provides direction, and the config wires them together.

## Installation

### Prerequisites

- Python 3.10–3.13
- An MCP-compatible agent (see [Agent Setup](#agent-setup))
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install DSAGT

```bash
git clone <repository-url>
cd dsagt

# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

This installs three command-line tools:

- `dsagt-pipeline-server` — Run the pipeline execution server
- `dsagt-registry-server` — Run the registry builder server
- `dsagt-knowledge-server` — Run the knowledge base server

## How It Works

### Agent-Driven Tool Registration

The default registry ships with a set of standard data tools (CSV loading, profiling, validation, splitting). You can use these out of the box, or work with the agent to discover and register additional tools:

1. Point the agent at a CLI tool, its `--help` output, or its documentation
2. The agent uses the **registry builder** to analyze the interface — it can read files, fetch URLs, and run commands
3. The agent proposes a tool specification and saves it to the registry via `save_tool_spec`
4. You review the spec and guide the agent to adjust if needed
5. The tool is immediately available for execution through the pipeline server

The registry builder exposes these tools to the agent:

- `read_file` — Read local documentation files
- `http_request` — Fetch documentation from URLs
- `run_command` — Execute commands to get help output (e.g., `--help`)
- `save_tool_spec` — Save a tool specification to the registry
- `get_registry` — View all registered tools
- `search_registry` — Search for tools by name or description

### Pipeline Execution

1. The pipeline server copies the base `registry.yaml` to `runtime/registry.yaml` at startup
2. The agent uses the pipeline server to run tools from the session registry
3. Each tool execution is logged to a provenance file for reproducibility

### Knowledge Base

The knowledge base provides semantic search over indexed documentation collections. It uses FAISS for vector search and optionally reranks results with a cross-encoder model.

To set up the core collections (NeMo Curator, AIDRIN):

```bash
export LLM_API_KEY="your-api-key"
python scripts/setup_core_kb.py
```

This clones repositories, downloads papers, chunks the content, and builds FAISS indexes in `kb_index/`. See `python scripts/setup_core_kb.py --help` for options.

## Running the MCP Servers

### Pipeline Server

```bash
# Use default bundled registry
dsagt-pipeline-server

# Use a custom registry from a previous session
dsagt-pipeline-server --registry my_registry.yaml

# Use a specific registry file and runtime directory
dsagt-pipeline-server --registry path/to/registry.yaml --runtime-dir ./my_session
```

### Registry Builder Server

```bash
# Writes to default ./runtime/registry.yaml
dsagt-registry-server

# Specify a different registry file
dsagt-registry-server --registry path/to/registry.yaml
```

### Knowledge Base Server

```bash
# Use defaults (base: ./kb_index, runtime: ./runtime)
dsagt-knowledge-server

# Specify directories and disable reranking
dsagt-knowledge-server --base-index-dir path/to/kb_index --runtime-dir ./runtime --no-rerank
```

## Agent Setup

DSAGT works with any MCP-compatible agent. All three platforms speak MCP over stdio — the servers are identical, only the configuration format differs.

Platform-specific configuration and quickstart guides are in the `agents/` directory:

- **Goose**: [`agents/goose/README.md`](agents/goose/README.md)

Each directory contains the config files to copy into your project and setup instructions.

### Path Considerations

The servers use relative paths by default (`--runtime-dir ./runtime`, `--base-index-dir ./kb_index`). These resolve relative to the working directory where the agent launches the server process. This works for all platforms as long as the agent's working directory is the DSAGT project root.

If your agent launches servers from a different working directory, pass absolute paths via args:

```bash
dsagt-knowledge-server --base-index-dir /absolute/path/to/kb_index
```

### Example Session

```
User: I have a script at scripts/preprocess.py that cleans CSV files.
      Can you register it as a pipeline tool?

Agent: Let me look at that script to understand its interface.
       [reads the file, runs --help, proposes a spec]
       I've registered "preprocess" with parameters for input_file,
       output_file, and an optional --drop-nulls flag. Want to try it?

User: Yes, run it on data/raw.csv

Agent: [executes the tool via the pipeline server]
       Done — output written to data/cleaned.csv. 142 rows processed,
       3 null rows dropped.
```

## Smoke Test

The `tests/smoke_test/` directory ships with everything needed to verify all three servers end-to-end. The knowledge base steps require an embedding API key — skip them if you don't have one yet.

```
tests/smoke_test/
├── greet.py                  # Simple CLI tool for the agent to register and execute
└── knowledge/                # Documents for knowledge base ingestion
    ├── DESCRIPTION.md
    ├── installation.md
    ├── api_reference.md
    └── troubleshooting.md
```

### 1. Verify the test script

```bash
python tests/smoke_test/greet.py World
```

Should print `{"message": "Hello, World!", "status": "ok"}`.

### 2. Start a session with all three servers

Follow your platform's quickstart in `agents/<platform>/README.md` to launch a session with all three servers. Use `--no-rerank` on the knowledge server to skip the cross-encoder model download for faster startup.

If you don't have an embedding API key, omit the knowledge server and skip steps 5–6.

### 3. Register a tool

Prompt the agent:

```
Register tests/smoke_test/greet.py as a pipeline tool.
Run "python tests/smoke_test/greet.py --help" to see its interface.
```

The agent should run the `--help` command via the registry builder, then call `save_tool_spec` with a spec. Confirm the agent reports the tool was "added successfully".

### 4. Execute the tool

Prompt the agent:

```
Run the greet tool with name "World" and greeting "Hi".
```

The agent should call the tool via the pipeline server and show you the JSON output with `"message": "Hi, World!"`.

### 5. Ingest documents into the knowledge base

Prompt the agent:

```
Ingest the folder tests/smoke_test/knowledge into the knowledge base.
```

The agent should call `kb_ingest` and report the number of files and chunks created.

### 6. Search the knowledge base

Prompt the agent:

```
Search the knowledge collection for "how to handle large files".
```

The agent should call `kb_search` and return chunks from `troubleshooting.md` (about lazy loading and out of memory errors) as the top results, since those are semantically closest to the query.

Then try:

```
List all knowledge base collections.
```

The agent should show `knowledge` with the description from `DESCRIPTION.md`.

### 7. Verify

After the session, check these artifacts from the project root:

**Registry contains default tools plus greet:**

```bash
cat runtime/registry.yaml
```

You should see the bundled default tools (scan_directory) plus a `greet` entry added by the agent, with `name`, `executable`, `description`, and `parameters`.

**Provenance was logged:**

```bash
cat runtime/provenance.log
```

You should see a timestamped entry showing the `greet` tool, the arguments passed, and the full command that was executed.

**Knowledge base was indexed:**

```bash
ls runtime/kb_index/knowledge/
```

You should see `index.faiss`, `chunks.jsonl`, and `DESCRIPTION.md`.

**Chunks are well-formed:**

```bash
head -3 runtime/kb_index/knowledge/chunks.jsonl
```

Each line should be a JSON object with `id`, `text`, and `metadata` fields. The metadata should include `source_file`, `collection`, `chunk_index`, and `file_type`.

### Cleanup

The `runtime/` directory is created during the smoke test and can be deleted afterward:

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
    parameters:
      param_name:
        type: string|integer|number|boolean|array|object
        required: true|false
        description: Parameter description
        default: optional_default_value
```

Required parameters are passed as positional arguments. Optional parameters use `--flag value` syntax.

## Development

### Project Structure

```
├── src/dsagt/
│   ├── __init__.py
│   ├── mcp_utils.py                # Shared MCP server utilities
│   ├── registry.py                 # Tool registry management
│   ├── registry.yaml               # Default tool registry (bundled with package)
│   ├── knowledge.py                # Semantic search over document collections
│   ├── pipeline_server.py          # MCP server: tool execution
│   ├── registry_server.py          # MCP server: tool discovery and registration
│   └── knowledge_server.py         # MCP server: knowledge base search
├── agents/
│   └── goose/                      # Goose agent config and quickstart
├── tests/
│   ├── test_registry.py                # ToolRegistry unit tests
│   ├── test_registry_server.py         # Registry builder server handler tests
│   ├── test_knowledge_base.py          # KnowledgeBase and APIEmbeddingClient tests
│   ├── test_knowledge_server.py        # Knowledge server handler tests
│   ├── test_knowledge_integration.py   # Integration tests (require API key)
│   └── smoke_test/                     # End-to-end smoke test fixtures
│       ├── greet.py                    # Sample CLI tool
│       └── knowledge/                  # Sample documents for KB ingestion
├── scripts/
│   └── setup_core_kb.py            # Knowledge base setup script
├── pyproject.toml
└── README.md
```

### Running Tests

Install dev dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

To run a specific test file or test class:

```bash
# Run only the registry tests
pytest tests/test_registry.py

# Run only the server handler tests
pytest tests/test_registry_server.py
pytest tests/test_knowledge_server.py

# Run the knowledge base tests
pytest tests/test_knowledge_base.py

# Run a single test class
pytest tests/test_registry.py::TestCallTool

# Run a single test with verbose output
pytest tests/test_registry.py::TestCallTool::test_success -v
```

The registry tests mock `subprocess.run` so they don't execute real commands. The server tests invoke MCP handlers directly without starting the stdio transport, so they run fast and need no network access. The knowledge base tests mock the embedding API and use real FAISS indexes on temp files — no API key or network needed. The knowledge server tests use async helpers to exercise the background job pattern for ingest and append operations.

## Troubleshooting

### MCP Server Not Found

```bash
# Verify installation
which dsagt-pipeline-server
which dsagt-registry-server
which dsagt-knowledge-server

# Reinstall if needed
pip install -e . --force-reinstall
```

### Tools Not Executing

Check the provenance log to see what command was run:

```bash
cat runtime/provenance.log
```

Verify that the executable path is correct and any interpreter (python, Rscript, etc.) is in your PATH.

### Registry File Not Found

Use absolute paths in configuration for reliability:

```yaml
args:
  - --registry
  - /full/path/to/registry.yaml
```

## License

TBD
