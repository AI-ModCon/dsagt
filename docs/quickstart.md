# Quick Start

This guide walks through knowledge ingest, code registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/). The examples use `claude`; substitute another agent (`goose`, `codex`, `opencode`, `cline`) if you prefer — the prompts are agent-agnostic.

DSAgt is **BYOA**: your agent talks to its own LLM provider directly, and the trace store is a serverless SQLite file per project.

## Setup

```bash
# Install (any Python 3.12/3.13 environment)
pip install "git+https://github.com/AI-ModCon/dsagt.git"

# Set a convenience variable for the smoke test directory (not a normal dsagt step)
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# 1. Create a project.  `dsagt init` is interactive — follow the menu to name it
#    `quickstart`, pick your agent, and choose knowledge collections + skill sources.
#    It sets up the knowledge base on first run (a ~130 MB local embedder downloads once).
dsagt init

# 2. Launch the agent in the project:
dsagt start quickstart     # …or: cd ~/dsagt-projects/quickstart && <your agent>
```

## Agent Prompts

Inside the agent, paste these prompts one at a time. Replace `$SMOKE_DIR` with the absolute path you exported — the chat does not expand shell variables.

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > Register the csvkit CLI codes `csvcut`, `csvgrep`, `csvstat`, and `csvlook`.
3. > Use the `scan_directory` code from the registry to scan `$SMOKE_DIR/data/`.
4. > Summarize `samples.csv` — columns, row count, quality issues using csvkit codes from the registry.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > Tell me what you remember about the samples dataset.

## Capabilities Covered

| Prompt | DSAgt capability |
|--------|-------------|
| 1 | `dsagt-server` (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | `dsagt-server` (`save_code_spec`) — writes `codes/csvcut.md`, etc. |
| 3 | `dsagt-run` provenance wrapper — records the execution to `trace_archive/` |
| 4 | KB recall via `kb_search` and registered code execution |
| 5–6 | Explicit memory (`kb_remember` → `.dsagt/explicit_memories.yaml`) + `kb_get_memories` |

## Verify the Artifacts

Exit the agent (`Ctrl+C` or `/exit`), then:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{codes,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces land in a serverless SQLite store.  Browse them with:
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db
```

## Non-Interactive Smoke Test

The same flow runs non-interactively and asserts each artifact is present:

```bash
dsagt smoke-test --agent claude
```

## Knowledge Base Setup

`dsagt init` sets up the project's knowledge base with three kinds of collection:

- **Code Specs** — DSAgt's bundled code specs, always set up so the agent finds them via `search_registry` from the first session.
- **Skill Catalogs** — the skill-catalog sources you chose at init (default `genesis`), cloned and indexed so `search_skills` returns installable skills.
- **Knowledge Collections** — optional reference document sets you chose at init (`nemo_curator`, `aidrin`).

`--include` / `--exclude` (asset names, or `all`) select the set non-interactively. The default embedder is a local sentence-transformers model (~130 MB, CPU-side, no API key).

## Optional: Episodic Memory

Answer **yes** to "Enable episodic memory?" in the `dsagt init` prompts to have the MCP server capture each session turn into a searchable `session_memory` collection. Capture is mechanical (chunk + embed) and reuses the local embedder, so there's nothing extra to download. See [Memory → Episodic Memory](memory.md#episodic-memory).
