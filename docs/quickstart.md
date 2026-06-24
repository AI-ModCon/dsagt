# Quick Start

This guide walks through knowledge ingest, tool registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/). The examples use `claude`; substitute another agent (`goose`, `codex`, `opencode`) if you prefer — the prompts are agent-agnostic.

## Setup

```bash
# Install
pip install https://github.com/AI-ModCon/dsagt/archive/refs/tags/0.1.0.zip

# Set a convenience variable for the smoke test directory (not a normal dsagt step)
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# 1. Create a new project called quickstart
dsagt init quickstart --agent claude

# 2. Start MLflow in the background and print the OTel routing exports
dsagt mlflow quickstart

# 3. Paste the export block from step 2 into this shell, then launch the agent
cd ~/dsagt-projects/quickstart && claude
```

## Agent Prompts

Inside the agent, paste these prompts one at a time. Replace `$SMOKE_DIR` with the absolute path you exported — the chat does not expand shell variables.

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > Register the csvkit CLI tools `csvcut`, `csvgrep`, `csvstat`, and `csvlook`.
3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`.
4. > Summarize `samples.csv` — columns, row count, quality issues using csvkit tools from the registry.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > Tell me what you remember about the samples dataset.

## Teardown

After exiting the agent, distill the session into episodic memory and stop the MLflow daemon:

```bash
# Distill traces into episodic memory
dsagt memory --project quickstart

# Stop the MLflow daemon
dsagt stop quickstart
```

## What Was Exercised

| Prompt | DSAgt layer |
|--------|-------------|
| 1 | Knowledge MCP server (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | Registry MCP server (`save_tool_spec`) — writes `tools/csvcut.md`, etc. |
| 3 | `dsagt-run` provenance wrapper — records exec layer to `trace_archive/` |
| 4 | KB recall via `kb_search` and registered tool execution |
| 5–6 | Explicit memory (`kb_remember` → `explicit_memories.yaml`) + `kb_get_memories` |

## Verify the Artifacts

```bash
dsagt info quickstart
ls ~/dsagt-projects/quickstart/{tools,trace_archive}
cat ~/dsagt-projects/quickstart/explicit_memories.yaml
```

The MLflow UI URL is printed by `dsagt mlflow quickstart`.

## Non-Interactive Smoke Test

The same flow runs non-interactively and asserts each artifact is present:

```bash
dsagt smoke-test --agent claude
```

## First-Time Knowledge Base Setup

`dsagt setup-kb` builds shared ChromaDB collections under `~/.dsagt/kb_index/` that every project on this machine reuses. Run this once after installation.

```bash
dsagt setup-kb                       # all collections (local embedder, no creds)
dsagt setup-kb --collection nemo_curator
dsagt setup-kb --embedding-backend api --embedding-base-url ... --embedding-api-key ...
```

Three collections are populated:

- **Tool Specs** — DSAgt's bundled tool specs from `src/dsagt/tools/`, tagged `source: bundled`.
- **Skills** — DSAgt's bundled skill workflows from `src/dsagt/skills/`.
- **Domain Knowledge** — NeMo Curator and AI Data Readiness Inspector reference corpora.

The Tool Specs and Skills collections are wiped and rebuilt on every run, so re-run `setup-kb` after upgrading DSAgt.

The default embedder is a local sentence-transformers model (~130 MB, CPU-only, no API key). Pass `--embedding-backend api` to route through a hosted embedder via LiteLLM.
