# Quick Start

This guide walks through knowledge ingest, code registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/). The examples use `claude`; substitute another agent (`goose`, `codex`, `opencode`, `cline`) if you prefer — the prompts are agent-agnostic.

## Setup

```bash
# Install (Python 3.12 or later on Apple Silicon or Linux x86_64; CI tests 3.12 and 3.13)
pip install "git+https://github.com/AI-ModCon/dsagt.git"

# Fetch the sample files and set a convenience variable for the prompts below
curl -sL https://github.com/AI-ModCon/dsagt/archive/refs/heads/main.tar.gz \
    | tar xz --strip-components=2 dsagt-main/tests/smoke_test
export SMOKE_DIR="$PWD/smoke_test"

# 1. Create a project.  `dsagt init` is interactive: follow the menu to name it
#    `quickstart`, pick your agent, and choose knowledge collections + skill sources.
#    It sets up the knowledge base on first run (a 133 MB local embedder downloads once).
dsagt init

# 2. Launch the agent in the project:
dsagt start quickstart     # or: cd ~/dsagt-projects/quickstart && <your agent>
```

## Agent Prompts

Inside the agent, paste these prompts one at a time. Replace `$SMOKE_DIR` with the absolute path you exported; the chat does not expand shell variables.

<!-- Shared with README.md. Edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:quickstart-prompts:start -->"
   end="<!-- md-shared:quickstart-prompts:end -->"
%}

`csv_summary.py` uses only the standard library, so registration and execution need no dependency install. Step 4's null-column finding is the fact you store and recall in 5–6.

## Capabilities Covered

<!-- Shared with README.md. Edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:quickstart-capabilities:start -->"
   end="<!-- md-shared:quickstart-capabilities:end -->"
%}

## Verify the Artifacts

Exit the agent (`Ctrl+C` or `/exit`), then:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{skills,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces are stored in a serverless SQLite store.  Browse them with:
dsagt traces quickstart
```
