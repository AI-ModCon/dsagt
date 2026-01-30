# DSAGT

**D**ata **S**mith **A**gent **T**oolkit - AI-assisted data pipeline builder.

DSAGT helps you build reproducible data processing pipelines using AI agents like Goose. It provides three MCP servers:

1. **Pipeline Server** - Executes data processing tools from a registry
2. **Registry Builder** - Helps you add new tools by analyzing documentation
3. **Knowledge Base** - Semantic search over indexed documentation

## Installation

### Prerequisites

- Python 3.10 or higher
- [Goose](https://github.com/block/goose) installed
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install DSAGT

Clone the repository and install the package:

```bash
git clone <repository-url>
cd data_agent.git

# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

This installs three command-line tools:
- `dsagt-pipeline-server` - Run the pipeline execution server
- `dsagt-registry-builder` - Run the registry builder server
- `dsagt-knowledge-server` - Run the knowledge base server

### Set Up the Knowledge Base

The knowledge base provides semantic search over documentation for NeMo Curator, AIDRIN, and other indexed sources. Run the setup script to download and index the core collections:

```bash
# Set your API key for embeddings (PNNL AI Incubator or OpenAI-compatible)
export LLM_API_KEY="your-api-key"

# Run the setup script
python scripts/setup_core_kb.py
```

This will:
1. Clone the NeMo Curator repository (docs, code, tutorials)
2. Clone the AIDRIN repository (data readiness framework)
3. Download relevant arXiv papers
4. Chunk and embed all documents into a FAISS index

The index is saved to `kb_index/` by default. You can customize the location:

```bash
# Use a custom index directory
python scripts/setup_core_kb.py --index-dir /path/to/my_index

# Set up only one collection
python scripts/setup_core_kb.py --collection nemo_curator
python scripts/setup_core_kb.py --collection aidrin
```

**Note:** The initial setup requires network access to GitHub and arXiv, and may take several minutes depending on your connection and the embedding API response time.

## Running the MCP Servers

### Pipeline Server

The pipeline server executes tools defined in a registry YAML file:

```bash
# Use default registry.yaml in current directory
dsagt-pipeline-server

# Use a specific registry file
dsagt-pipeline-server --registry path/to/my_registry.yaml

# Specify runtime directory for session data
dsagt-pipeline-server --registry registry.yaml --runtime-dir ./my_session
```

### Registry Builder Server

The registry builder helps you add new tools to the session registry:

```bash
# Writes to default ./runtime/registry.yaml (pipeline's session registry)
dsagt-registry-builder

# Specify a different registry file
dsagt-registry-builder --registry path/to/registry.yaml
```

### Knowledge Base Server

The knowledge base server provides semantic search over indexed documentation. It copies base indexes to a session-specific runtime directory at startup:

```bash
# Use defaults (base: ./kb_index, runtime: ./runtime)
dsagt-knowledge-server

# Specify directories
dsagt-knowledge-server --base-index-dir path/to/kb_index --runtime-dir ./runtime

# Disable reranking for faster (but less accurate) results
dsagt-knowledge-server --no-rerank
```

To index documentation, place files in a folder and ingest via the `kb_ingest` tool. Add a `DESCRIPTION.md` file to the folder for agent discovery.

## Using with Goose

**Important**: The MCP server commands (`dsagt-pipeline-server` and `dsagt-registry-builder`) must be accessible in your PATH when running Goose. Depending on how you installed DSAGT, you may need to:

- **If installed with `pip install -e .`**: Commands should be available globally
- **If installed with `uv pip install -e .`**: Prefix commands with `uv run` (see examples below)
- **If using a virtual environment**: Activate the environment before running Goose, or use absolute paths to the commands

There are three ways to configure Goose to use DSAGT's MCP servers:

### Option 1: Using `--with-extension` Flag (One-time use)

Run Goose with the DSAGT servers for a single session:

**If installed globally (pip install -e .):**
```bash
goose session \
  --with-extension 'dsagt-pipeline-server --registry registry.yaml' \
  --with-extension 'dsagt-registry-builder'
```

**If using uv:**
```bash
goose session \
  --with-extension 'uv run dsagt-pipeline-server --registry registry.yaml' \
  --with-extension 'uv run dsagt-registry-builder'
```

### Option 2: Configuration File (Persistent)

Add the DSAGT extension configurations to one of these files:

- **Global**: `~/.config/goose/config.yaml` - Available in all Goose sessions
- **Project-local**: `goose.yaml` in your project directory - Only for this project

Then add this configuration under the `extensions:` section:

**If installed globally (pip install -e .):**
```yaml
extensions:
  dsagt:
    enabled: true
    name: dsagt
    type: stdio
    cmd: dsagt-pipeline-server
    args:
      - --registry
      - /path/to/registry.yaml  # Base registry to copy from
      - --runtime-dir
      - /path/to/runtime
    timeout: 300

  registry_builder:
    enabled: true
    name: tool-registry-server
    type: stdio
    cmd: dsagt-registry-builder
    args:
      - --registry
      - /path/to/runtime/registry.yaml  # Session registry (written by pipeline server)
    timeout: 300

  knowledge_base:
    enabled: true
    name: knowledge-base
    type: stdio
    cmd: dsagt-knowledge-server
    args:
      - --base-index-dir
      - /path/to/kb_index
      - --runtime-dir
      - /path/to/runtime
    timeout: 300
```

**If using uv:**
```yaml
extensions:
  dsagt:
    enabled: true
    name: dsagt
    type: stdio
    cmd: uv
    args:
      - run
      - dsagt-pipeline-server
      - --registry
      - /path/to/registry.yaml  # Base registry to copy from
      - --runtime-dir
      - /path/to/runtime
    timeout: 300

  registry_builder:
    enabled: true
    name: tool-registry-server
    type: stdio
    cmd: uv
    args:
      - run
      - dsagt-registry-builder
      - --registry
      - /path/to/runtime/registry.yaml  # Session registry (written by pipeline server)
    timeout: 300

  knowledge_base:
    enabled: true
    name: knowledge-base
    type: stdio
    cmd: uv
    args:
      - run
      - dsagt-knowledge-server
      - --base-index-dir
      - /path/to/kb_index
      - --runtime-dir
      - /path/to/runtime
    timeout: 300
```

Then run Goose:
```bash
# If using global config
goose session

# If using project-local config
goose session --config goose.yaml
```

## Quick Start Example

1. **Install DSAGT**:
   ```bash
   uv pip install -e .
   ```

2. **Create a tool registry** (`registry.yaml`):
   ```yaml
   tools:
     - name: load_csv
       description: Load a CSV file and output basic info
       executable: python tools/load_csv.py
       parameters:
         location:
           type: string
           required: true
           description: Path to CSV file
   ```

3. **Start a Goose session**:
   ```bash
   goose session --config goose.yaml
   ```

4. **Ask Goose to build a pipeline**:
   ```
   User: I have data at data/customers.csv.
         Can you help me prepare it for machine learning?

   Goose: I'll help you build that pipeline. Let me gather some information first:
          1. Have you worked with this data before? Any known quality issues?
          2. What are you trying to predict or analyze?
          3. What train/validation/test split ratios would you prefer?
          ...
   ```

## How It Works

### Pipeline Execution

1. The pipeline server copies the base `registry.yaml` to `runtime/registry.yaml` at startup
2. Goose uses the **dsagt** extension to run tools from the session registry
3. Each tool execution is logged to a provenance file for reproducibility

### Adding New Tools

1. Goose uses the **tool-registry-server** extension to analyze documentation
2. Available tools for building the registry:
   - `read_file` - Read local documentation files
   - `http_request` - Fetch documentation from URLs
   - `run_command` - Execute commands to get help output (e.g., `--help`)
   - `save_tool_spec` - Save a tool specification to the registry
   - `get_registry` - View all registered tools
   - `search_registry` - Search for tools by name or description

3. New tools are saved directly to the session registry and are immediately available

## Tool Registry Format

Tools are defined in YAML format:

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

## Examples

See the `demo/` directory for complete examples:

```bash
# Run the demo
cd demo
python demo_test.py
```

This demonstrates:
- Loading and profiling data
- Registering custom preprocessing scripts
- Building a complete pipeline
- Viewing provenance logs

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_registry_builder.py

# Run with coverage
pytest --cov=dsagt
```

### Project Structure

```
.
├── src/
│   └── dsagt/
│       ├── core/           # Core registry management
│       ├── knowledge/      # Knowledge base for semantic search
│       ├── servers/        # MCP servers
│       └── tools/          # Package metadata
├── scripts/
│   └── setup_core_kb.py    # Knowledge base setup script
├── tools/                  # Example data processing tools
├── demo/                   # Example usage
├── tests/                  # Test suite
├── kb_index/               # Knowledge base index (generated)
├── registry.yaml           # Example tool registry
└── goose.yaml             # Goose configuration
```

## Troubleshooting

### MCP Server Not Found

If Goose reports that it can't find the MCP servers:

```bash
# Verify installation
which dsagt-pipeline-server
which dsagt-registry-builder
which dsagt-knowledge-server

# Reinstall if needed
uv pip install -e . --force-reinstall
```

### Registry File Not Found

Make sure the registry paths in your configuration are absolute or relative to where you run Goose:

```yaml
# Absolute path (recommended for global config)
args:
  - --registry
  - /full/path/to/registry.yaml

# Relative path (works from project directory)
args:
  - --registry
  - registry.yaml
```

### Tools Not Executing

Check the provenance log to see what command was run:

```bash
cat runtime/provenance.log
```

Verify that:
- The executable path is correct
- Any interpreter (python, Rscript, etc.) is in your PATH
- The tool script has necessary dependencies installed

## Contributing

See `IMPROVEMENTS.md` for a roadmap of planned enhancements.

## License

[Add your license here]
