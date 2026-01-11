# BASEDATA

AI-assisted data pipeline builder using Goose and MCP.

## Overview

BASEDATA helps you build data processing pipelines interactively with an AI agent. The agent:
- Gathers information about your data and goals
- Registers your custom scripts as tools
- Orchestrates tools to build a pipeline
- Records all steps for reproducibility

## Quick Start

```bash
# 1. Install Goose
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash

# 2. Setup BASEDATA
./setup.sh --api-key YOUR_PNNL_API_KEY

# 3. Start the agent
cd /path/to/basedata
goose session
```

## Files

```
basedata/
├── mcp_server.py              # MCP server exposing tools
├── registry.yaml              # Tool definitions
├── .goosehints                 # Agent instructions
├── setup.sh                   # Setup script
├── config.yaml                # Goose config template
├── custom_providers/
│   └── pnnl.json              # PNNL API provider config
├── tools/                     # Tool executables
│   ├── load_csv.py
│   ├── profile_data.py
│   ├── validate_schema.py
│   ├── split_data.py
│   └── summarize.sh
└── runtime/                   # Created at runtime
    ├── registry.yaml          # Working registry (editable)
    └── provenance.log         # Execution history
```

## How It Works

1. **Start a session** - `goose session` in the basedata directory
2. **Agent asks questions** - About your data, scripts, and goals
3. **Register your scripts** - Agent adds your tools to the registry
4. **Build pipeline** - Agent orchestrates tools based on your goals
5. **Get provenance** - All steps recorded in `runtime/provenance.log`

## Example Interaction

```
You: I have some CSV files in ./data/ that I need to clean and split for ML training.
     I have a custom Python script clean.py that handles missing values.

Agent: I'll help you build that pipeline. Let me:
       1. Register your clean.py script as a tool
       2. Load and profile your CSV files
       3. Run your cleaning script
       4. Split into train/dev/test sets
       
       First, what's the exact path to your CSV files and clean.py script?
```

## Adding Tools

### Via the Agent

Tell the agent about your script:
```
You: I have a script at ./scripts/normalize.py that normalizes numeric columns.
     It takes --input and --output arguments.

Agent: I'll register that as a tool...
```

### Via registry.yaml

Edit `runtime/registry.yaml` directly:
```yaml
  - name: normalize
    description: Normalize numeric columns
    executable: python ./scripts/normalize.py
    parameters:
      input:
        type: string
        required: true
      output:
        type: string
        required: true
```

## Provenance

Every tool call is logged:
```
2024-01-10T14:32:01 | load_csv | {"location": "data.csv"} | python tools/load_csv.py data.csv
2024-01-10T14:32:03 | clean_data | {"input": "data.csv"} | python ./scripts/clean.py data.csv
2024-01-10T14:32:05 | split_data | {"input": "cleaned.csv", "output_dir": "./splits"} | ...
```

Use for:
- Debugging failed pipelines
- Reproducing results
- Extracting deterministic scripts

## Configuration

### PNNL API

Set your API key:
```bash
export PNNL_API_KEY="your-key-here"
```

### Different Provider

Edit `~/.config/goose/config.yaml`:
```yaml
GOOSE_PROVIDER: openai  # or anthropic, ollama, etc.
GOOSE_MODEL: gpt-4o
```

### Custom .goosehints

Edit `.goosehints` to customize agent behavior for your workflow.

## Dependencies

- [Goose](https://github.com/block/goose) - AI agent framework
- Python 3.10+
- PyYAML: `pip install pyyaml`
- MCP: `pip install mcp`
