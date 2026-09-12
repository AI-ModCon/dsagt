# Quick Start

This guide walks through knowledge ingest, code registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/). The examples use `claude`; substitute another agent (`goose`, `codex`, `opencode`, `cline`) if you prefer — the prompts are agent-agnostic.

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

<!-- Shared with README.md — edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:quickstart-prompts:start -->"
   end="<!-- md-shared:quickstart-prompts:end -->"
%}

`csv_summary.py` is stdlib-only, so there's no dependency to install — registration and execution work out of the box. Step 4's null-column finding is the fact you store and recall in 5–6.

## Capabilities Covered

<!-- Shared with README.md — edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:quickstart-capabilities:start -->"
   end="<!-- md-shared:quickstart-capabilities:end -->"
%}

## Verify the Artifacts

Exit the agent (`Ctrl+C` or `/exit`), then:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{codes,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces land in a serverless SQLite store.  Browse them with:
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db
```
