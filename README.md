# DSAGT

**D**ata **S**mith **A**gent **T**oolkit - AI-assisted data pipeline builder.

## Context

This is a **first-pass implementation** of an agent-based data curation pipeline, developed according to discussions with the BASE-DATA team over the week of January 6-10, 2026. The goal is a minimal working prototype that can be iteratively improved based on real usage.

### Design Philosophy

The codebase is structured to facilitate **independent parallel development** across multiple concerns—agent behavior, tool catalog, registration mechanisms, deployment patterns—while maintaining a **lightweight end-to-end testbed** (`demo/demo_test.py` and `goose session`) that allows contributors to validate their work asynchronously without requiring full team coordination. Each component can be improved in isolation and tested against the common integration points.

### How This Codebase Meets Team Requirements

The following table maps key decisions from the January 9th team meeting to their implementation in this codebase:

| Team Decision | Implementation |
|---------------|----------------|
| **Tools = CLI executables** (language-agnostic) | `registry.yaml` defines tools as `executable: <command>`. Tools in `tools/` include Python scripts and a bash script (`summarize.sh`) as proof of concept. |
| **No pipeline ontology** (flat, simple approach) | No abstract hierarchy. Tools take files in, produce files out. Chained via agent conversation. |
| **Two-phase architecture** (interactive → deterministic) | Interactive: Goose agent session. Deterministic: `provenance.log` records all tool calls for replay. |
| **Use existing agent CLI** (Goose, not custom) | Goose configured via `setup.sh`. Architecture is modular for future replacement. |
| **Tool Registry with MCP interface** | `registry.py` + `mcp_server.py` expose tools to agents via MCP protocol. |
| **Tracking/Provenance** | Every tool call logged to `runtime/provenance.log` with timestamp, arguments, and exact command. |
| **Stock demo tools + sample dataset** | `tools/` contains load_csv, profile_data, validate_schema, split_data. `demo/` has building_sensors.csv. |

### What's Not Yet Implemented

| Component | Status |
|-----------|--------|
| **Tool Onboarder** | Placeholder `register_tool` exists; robust introspection TBD (Andrew's task) |
| **Pipeline Exporter** | Provenance log exists; export to WDL/Airflow/Parsl deferred pending ASC coordination |
| **Constraints Layer** | Future: custom LangGraph for controlled agent behavior |

---

## Quick Start

### 1. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or with conda
conda create -n dsagt python=3.12
conda activate dsagt
```

### 2. Install DSAGT

```bash
cd /path/to/dsagt
pip install -e .
```

This installs:
- `pyyaml` - YAML parsing for tool registry
- `mcp` - Model Context Protocol for agent communication

### 3. Install Goose

```bash
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
# Press Ctrl+C when the config menu appears
```

### 4. Configure Environment

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# PNNL API Configuration
export PNNL_API_KEY="your-api-key-here"
export OPENAI_API_KEY="${PNNL_API_KEY}"
export OPENAI_BASE_URL="https://ai-incubator-api.pnnl.gov"
```

Then reload:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

### 5. Setup Goose Config

```bash
cd /path/to/dsagt
./setup.sh                                    # Uses default model
./setup.sh --model gpt-4o-project             # Specify model
./setup.sh --list-models                      # Show available models
```

### 6. Run

```bash
cd /path/to/dsagt
goose session
```

## LLM Backend Requirements

DSAGT requires an OpenAI-compatible API endpoint:

| Requirement | Details |
|-------------|---------|
| **API Format** | OpenAI Chat Completions API (`/v1/chat/completions`) |
| **Authentication** | Bearer token via `OPENAI_API_KEY` |
| **Tool Calling** | Function/tool calling support (required for MCP) |
| **Streaming** | Recommended but not required |
| **Context Window** | 8K+ tokens recommended |

**Tested Models (PNNL endpoint):**
- `claude-sonnet-4-20250514-v1-project` (default, recommended)
- `gpt-4o-project`
- `claude-opus-4-20250514-v1-project` (highest capability)

See `models.txt` for full list. Other compatible endpoints: OpenAI API, Azure OpenAI, Ollama.

## Demo (No Goose Required)

Test the tool registry directly:

```bash
cd demo
python demo_test.py
```

## Architecture

```
dsagt/
├── pyproject.toml         # Package configuration & dependencies
├── registry.py            # Core: tool registry, execution, provenance
├── mcp_server.py          # MCP interface for Goose
├── registry.yaml          # Tool definitions (YAML)
├── .goosehints            # Agent prompt/instructions
├── setup.sh               # Goose configuration script
├── models.txt             # Available PNNL models
├── tools/                 # Built-in tool executables
│   ├── load_csv.py
│   ├── profile_data.py
│   ├── validate_schema.py
│   ├── split_data.py
│   └── summarize.sh       # Bash example (language-agnostic proof)
├── demo/
│   ├── building_sensors.csv   # Sample dataset
│   ├── fill_missing.py        # Example custom script
│   └── demo_test.py           # Test without Goose
└── runtime/                   # Created at runtime
    ├── registry.yaml          # Working registry (editable)
    └── provenance.log         # Execution history
```

## Tool Interface

Tools are defined in `registry.yaml`:

```yaml
tools:
  - name: my_tool
    description: What the tool does
    executable: python path/to/script.py
    parameters:
      input:
        type: string
        required: true
      output:
        type: string
        required: true
```

**Tool Executable Requirements:**
- Accept arguments (positional for required, `--flag` for optional)
- Output JSON to stdout
- Exit 0 on success, non-zero on error
- Can be any language (Python, bash, R, Fortran, etc.)

## Provenance

Every tool call is logged to `runtime/provenance.log`:

```
2024-01-10T14:32:01 | load_csv | {"location": "data.csv"} | python tools/load_csv.py data.csv
2024-01-10T14:32:03 | normalize | {"input": "data.csv", "output": "norm.csv"} | python normalize.py ...
```

Use cases: debug pipelines, reproduce results, extract to standalone scripts, audit data lineage.

---

# Development Plan

## Team Tasks

### Andrew: Robust Tool Onboarder

**Goal:** Create mechanism to take arbitrary executables and expose them to agents via MCP.

**Current State:**
- Basic `register_tool` function exists
- Agent infers schema from conversation (error-prone)
- No validation or testing of registered tools
- No introspection of actual script interface

**Limitations to Address:**

| Problem | Description |
|---------|-------------|
| Schema inference is fragile | Agent guesses parameter types from conversation |
| No validation | Registered tools aren't tested before use |
| No introspection | Agent can't inspect script to discover interface |
| No error recovery | Failed registration requires manual intervention |

**Approaches to Explore:**

1. **CLI introspection tool** (`dsagt register ./script.py`)
   - Parse argparse/click definitions
   - Extract from docstrings
   - Validate before adding to registry

2. **Script annotations**
   ```python
   # DSAGT-TOOL: normalize
   # DSAGT-DESC: Normalize numeric columns
   # DSAGT-PARAM: input (string, required) - Input CSV
   ```

3. **Interactive wizard** - Run `--help`, ask clarifying questions, test with sample input

4. **Schema-first** - User provides YAML, agent validates script matches

**Deliverables:**
1. Design document for robust registration protocol
2. Prototype of at least one approach
3. Test suite for registration edge cases
4. Updated `.goosehints` with better registration guidance

**Success Criteria:** 90%+ successful registration for scripts with standard interfaces.

---

### Rohith: Agent Integration & Improvements

**Goal:** Configure and improve Goose agent performance; explore alternatives.

**Current State:**
- Basic `.goosehints` prompt
- Default Goose configuration
- No recipes or sub-agents

**Areas to Explore:**

1. **Better `.goosehints`**
   - Structured task decomposition
   - Explicit error handling instructions
   - Examples of good/bad tool calls
   - Data science workflow guidance

2. **Goose Configuration**
   - `GOOSE_MODE` settings
   - Model selection for different tasks
   - Timeout and retry behavior
   - Extension management

3. **Recipes** for common workflows:
   - `profile-and-validate.yaml`
   - `prepare-ml-dataset.yaml`
   - `register-tool-wizard.yaml`

4. **Alternative Platforms** (evaluation)
   - Claude Code, Cursor Agent, Cline, custom LangGraph
   - Criteria: tool calling reliability, context handling, extensibility

**Deliverables:**
1. Improved `.goosehints` with measurable performance gains
2. Documented configuration recommendations
3. At least 2 reusable recipes
4. Comparison report if alternatives explored

---

### Aaron: Development Tools & Realistic Dataset

**Goal:** Expand tool catalog and create stress-test dataset.

**Current State:**
- Basic tools: load_csv, profile_data, validate_schema, split_data
- Toy dataset: 39 rows, 7 columns, 1 missing value

**Tool Catalog Expansion:**

1. **AI-Ready Data Assessment**
   - AIDRIN or similar for data quality scoring
   - Detect: missing values, outliers, class imbalance, drift

2. **NVIDIA NeMo Assets**
   - Preprocessing utilities
   - Format converters
   - Schema validation

3. **Domain-Specific Tools**
   - Building energy: weather integration, occupancy modeling
   - Time series: resampling, windowing, lag features

**Realistic Development Dataset:**

Requirements:
- Multiple related files (not single CSV)
- Real-world messiness: missing values, inconsistent formats, outliers
- 10K+ rows to stress-test
- Multi-step pipeline required
- Building energy domain preferred

Candidates:
- ASHRAE Great Energy Predictor III
- Building Data Genome Project
- Synthetic multi-zone simulation

**Deliverables:**
1. 3+ new high-quality tools
2. Integration guide for external tools (AIDRIN, NeMo)
3. Realistic development dataset with documentation
4. Test pipeline exercising full tool catalog

---

## Milestones

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Setup & Familiarization | Everyone running demo, understanding codebase |
| 2 | Individual prototypes | Andrew: registry prototype, Rohith: improved hints, Aaron: 1 new tool |
| 3 | Integration | Merge prototypes, test together |
| 4 | Stress testing | Run against realistic dataset, document gaps |
| 5 | Polish & Documentation | README updates, recipes, handoff materials |

**Friday, January 16:** Hackathon to integrate components and test end-to-end flow.

## Coordination

- **Agentic Framework group** (Yadu Babuji, Kyle Chard): Agent orchestration patterns, workflow language selection
- **American Science Cloud**: API requirements, deployment targets
- **Multimodal team**: Data fusion challenges (when relevant)

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Test with `demo/demo_test.py` and `goose session`
3. Update README if adding tools or changing interfaces
4. Post updates to BASE-DATA Slack channel

## Dependencies

**Python:** 3.10-3.13 (3.14 has MCP compatibility issues)

**Packages:** (installed via `pip install -e .`)
- `pyyaml>=6.0`
- `mcp>=1.0.0`

**External:**
- [Goose](https://github.com/block/goose) - Agent CLI

**Optional dev dependencies:**
```bash
pip install -e ".[dev]"  # pytest, black, ruff
```

## License

TBD
