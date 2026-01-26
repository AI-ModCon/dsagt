# DSAGT

**D**ata **S**mith **A**gent **T**oolkit - AI-assisted data pipeline builder.

DSAGT helps you build reproducible data processing pipelines using AI agents like Goose. It provides two MCP servers:

1. **Pipeline Server** - Executes data processing tools from a registry
2. **Registry Builder** - Helps you add new tools by analyzing documentation

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

This installs two command-line tools:
- `dsagt-pipeline-server` - Run the pipeline execution server
- `dsagt-registry-builder` - Run the registry builder server

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

The registry builder helps you add new tools to your registry:

```bash
# Save to default tool_registry.yaml
dsagt-registry-builder

# Save to a specific file
dsagt-registry-builder --registry path/to/my_tools.yaml
```

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
  --with-extension 'dsagt-registry-builder --registry registry.yaml'
```

**If using uv:**
```bash
goose session \
  --with-extension 'uv run dsagt-pipeline-server --registry registry.yaml' \
  --with-extension 'uv run dsagt-registry-builder --registry registry.yaml'
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
      - /path/to/registry.yaml  # Use absolute path for global config
    timeout: 300

  registry_builder:
    enabled: true
    name: tool-registry-server
    type: stdio
    cmd: dsagt-registry-builder
    args:
      - --registry
      - /path/to/registry.yaml  # Same registry file
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
      - /path/to/registry.yaml  # Use absolute path for global config
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
      - /path/to/registry.yaml  # Same registry file
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

1. Goose uses the **dsagt** extension to run tools defined in `registry.yaml`
2. Each tool execution is logged to a provenance file for reproducibility
3. Results are saved to a runtime directory (default: `./runtime`)

### Adding New Tools

1. Goose uses the **tool-registry-server** extension to analyze documentation
2. Available tools for building the registry:
   - `read_file` - Read local documentation files
   - `http_request` - Fetch documentation from URLs
   - `run_command` - Execute commands to get help output (e.g., `--help`)
   - `save_tool_spec` - Save a tool specification to the registry
   - `get_registry` - View all registered tools
   - `search_registry` - Search for tools by name or description

3. New tools are saved to `tool_registry.yaml` in the correct format
4. You can then copy tools from `tool_registry.yaml` to your main `registry.yaml`

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
│       ├── servers/        # MCP servers
│       └── tools/          # Package metadata
├── tools/                  # Example data processing tools
├── demo/                   # Example usage
├── tests/                  # Test suite
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
