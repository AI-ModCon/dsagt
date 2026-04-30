# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSAGT (DataSmith Agent) is an AI-assisted data pipeline builder that exposes MCP (Model Context Protocol) servers to agent platforms (Claude Code, Goose, Roo, Cline). It helps domain scientists create reproducible, auditable data curation pipelines through iterative, knowledge-driven tool generation.

## Commands

```bash
# Install all dependencies (including dev)
uv sync --all-groups

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_registry.py

# Run a single test by name
uv run pytest tests/test_registry.py -k "test_call_tool"

# Format code
uv run black .

# Lint
uv run ruff check .
```

**Dependency management uses `uv` exclusively.** No conda or bare pip.

## Code Organization Convention

The codebase separates **commands** (entry points with argparse, launched as CLI tools or subprocesses) from **modules** (importable logic). Commands live in `src/dsagt/commands/`, modules live in `src/dsagt/`.

**Commands** (`src/dsagt/commands/`):
- `cli.py` — `dsagt init/start/setup-kb/list/mv/smoke-test` (user-facing CLI)
- `proxy_server.py` — `dsagt-proxy` (LiteLLM proxy with DSAGT callback)
- `run_tool.py` — `dsagt-run` (tool execution wrapper)
- `registry_server.py` — `dsagt-registry-server` (MCP server)
- `knowledge_server.py` — `dsagt-knowledge-server` (MCP server)
- `setup_core_kb.py` — core KB setup logic (called via `dsagt setup-kb`)
- `info.py` — `dsagt info` (project / config introspection)

**Modules** (`src/dsagt/`):
- `session.py` — Project init, agent config generation, env-var resolution, config load/validate, service start/stop, agent launch, end-of-session memory extraction orchestration
- `agents.py` — Per-agent-platform config writers (Claude Code, Goose, Cline, Roo, Codex)
- `knowledge.py` — FAISS/ChromaDB document retrieval, embedding backends, per-collection routing
- `registry.py` — `ToolRegistry` (CLI tools) + `SkillRegistry` (agent instruction skills), KB indexing
- `provenance.py` — LiteLLM callback (`create_callback`, `DSAGTCallback`), tool execution records (`run_and_record`, `ToolRecordStore`), execution-record indexing into ChromaDB, pipeline reconstruction (`reconstruct_pipeline`, dependency graph)
- `observability.py` — MLflow / OTel tracing setup, metadata stamping, sidechannel route YAML
- `memory.py` — Explicit memory (YAML), episodic-memory extraction prompt + LLM call, outlier detection, end-to-end `extract_session`

Entry points are defined in `pyproject.toml` `[project.scripts]` and all point to `dsagt.commands.*:main`.

## Architecture

### MCP Servers

Two MCP servers expose DSAGT capabilities:

1. **Registry Server** (`commands/registry_server.py` + `registry.py`) — Tool analysis, registration, dependency installation. Tools are saved as skill files (markdown with YAML frontmatter).

2. **Knowledge Server** (`commands/knowledge_server.py` + `knowledge.py`) — Semantic search over document collections using FAISS + ChromaDB with optional cross-encoder reranking. Background jobs for long operations.

### CLI + Session

`commands/cli.py` → `session.py` manages project lifecycle:
- `dsagt init <project> --agent <platform>` — Scaffolds project directory
- `dsagt start <project>` — Generates agent configs, starts proxy + MLflow, launches agent. Cleanup (memory extraction + service shutdown) runs automatically when the agent exits.
- `dsagt list` — Shows all projects with agent platform and service status
- `dsagt mv <project> <location>` — Moves a project directory

`RUNTIME_DIR` is anchored to the repo root via `pyproject.toml` discovery — `dsagt` commands work from any directory.

### Observability

- **LiteLLM proxy** (`commands/proxy.py` + `proxy_callback.py`) — Routes all LLM calls; captures intent/report layers of tool execution records to `trace_archive/`
- **dsagt-run** (`commands/run.py` + `run.py`) — Wraps tool commands; captures execution layer
- **MLflow** — Token usage, cost, latency via OTel

### Memory System

All in `memory_extraction.py` and `memory.py`:
- **Episodic memory** — End-of-session LLM extraction of facts from session logs into ChromaDB, with per-category outlier detection via embedding centroids
- **Explicit memory** (`memory.py`) — User-confirmed facts in YAML, loaded into agent context at session start

### Key Design Patterns

- **Agent-agnostic**: DSAGT is infrastructure, not an agent. Capabilities are MCP services.
- **Session isolation**: Each project gets its own directory with config, tools, skills, kb_index, trace_archive, mlflow data.
- **Provenance via three layers**: Intent (proxy callback), Execution (dsagt-run), Report (proxy callback). Records share `record_id`, `tool_name`, `session_id`.
- **Tools vs Skills**: Tools are CLI executables in `<project>/tools/` (specs with parameters, wrapped by dsagt-run). Skills are agent instruction workflows in `<project>/skills/` (SKILL.md + reference docs). Both are discoverable via ChromaDB-backed semantic search.

## DSAGT Pipeline Builder Workflow

When acting as a pipeline builder (using the MCP servers), follow these constraints:

1. **Never directly access data** — all data operations go through registered tools
2. **Tool preference hierarchy**: Registered tool → KB package tool → Custom implementation
3. **Generate paired tools** — every data operation gets a check tool (pre/post audit) and an operation tool
4. **Audit everything** — before/after JSON reports saved to `audit/`
5. **One step at a time** — iterate with the user, confirming approach before execution

## Testing Patterns

- Tests use pytest with `subprocess.run` mocking for command execution
- MCP server tests invoke handlers directly (no stdio transport)
- Async tests for server handlers
- Temp directories for isolation; `RUNTIME_DIR` is patched to `tmp_path` in test fixtures
- Integration tests in `test_knowledge_integration.py` require `LLM_API_KEY`
