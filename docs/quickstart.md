# Quick Start

This guide walks through knowledge ingest, tool registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/). The examples use `claude`; substitute another agent (`goose`, `codex`, `opencode`, `roo`, `cline`) if you prefer — the prompts are agent-agnostic.

DSAgt is **BYOA**: your agent talks to its own LLM provider. There is **no proxy and no MLflow daemon** — the trace store is a serverless SQLite file per project.

## Setup

```bash
# Install (any Python 3.12/3.13 environment)
pip install "git+https://github.com/AI-ModCon/dsagt.git"

# Set a convenience variable for the smoke test directory (not a normal dsagt step)
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# 1. Create a project called quickstart.  Interactive `dsagt init` prompts for the
#    agent, location, knowledge collections, skill sources, and episodic memory,
#    and provisions the shared knowledge base on first run.  --agent makes it
#    non-interactive (a ~130 MB local embedder downloads once):
dsagt init quickstart --agent claude

# 2. Launch the agent from the project directory:
cd ~/dsagt-projects/quickstart && claude     # …or: dsagt start quickstart
```

## Agent Prompts

Inside the agent, paste these prompts one at a time. Replace `$SMOKE_DIR` with the absolute path you exported — the chat does not expand shell variables.

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > Register the csvkit CLI tools `csvcut`, `csvgrep`, `csvstat`, and `csvlook`.
3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`.
4. > Summarize `samples.csv` — columns, row count, quality issues using csvkit tools from the registry.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > Tell me what you remember about the samples dataset.

## What Was Exercised

| Prompt | DSAgt layer |
|--------|-------------|
| 1 | `dsagt-server` (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | `dsagt-server` (`save_tool_spec`) — writes `tools/csvcut.md`, etc. |
| 3 | `dsagt-run` provenance wrapper — records exec layer to `trace_archive/` |
| 4 | KB recall via `kb_search` and registered tool execution |
| 5–6 | Explicit memory (`kb_remember` → `.dsagt/explicit_memories.yaml`) + `kb_get_memories` |

## Verify the Artifacts

Exit the agent (`Ctrl+C` or `/exit`), then:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{tools,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces land in a serverless SQLite store — no server to run.  Browse them with:
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db
```

## Non-Interactive Smoke Test

The same flow runs non-interactively and asserts each artifact is present:

```bash
dsagt smoke-test --agent claude
```

## Knowledge Base Provisioning

`dsagt init` provisions the project's knowledge base. The shared, machine-wide collections live under `~/dsagt-projects/kb_index/`, built once (the first project on a machine pays the cost) and copied into each project:

- **Tool Specs** — DSAgt's bundled tool specs from `src/dsagt/tools/`, always provisioned so the agent finds them via `search_registry` from the first session.
- **Skill Catalogs** — the skill-catalog sources you chose at init (default `genesis`), cloned and frontmatter-indexed so `search_skills` returns installable skills.
- **Knowledge Collections** — optional reference corpora you chose at init (`nemo_curator`, `aidrin`).

`--include` / `--exclude` (asset names, or `all`) select the set non-interactively. The default embedder is a local sentence-transformers model (~130 MB, CPU-side, no API key).

## Optional: Episodic Memory

Pass `--episodic` at init (or choose it in the interactive prompt) to have the MCP server distill each session turn into searchable facts:

```bash
dsagt init quickstart --agent claude --episodic --domain-tags "genomics,qc"
```

This downloads a small local LLM judge (~1 GB GGUF) on first use — no API key. See [Knowledge Base → Episodic Memory](knowledge-base.md#episodic-memory).
