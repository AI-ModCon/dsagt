# DSAgt

Documentation: **https://ai-modcon.github.io/dsagt** — quick start, capability pages, and use-case walkthroughs.

<!-- md-shared:intro:start -->
**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

![DSAgt architecture](docs/assets/overview.png)

DSAgt connects an MCP-compatible AI coding agent to code registration, a semantic knowledge base, skills discovery and creation, execution provenance, and observability infrastructure. It exposes these capabilities to a user's existing agent CLI or VS Code extension (Claude Code, OpenCode, Codex, and others).

**Prerequisites:** Python 3.12 or later (CI tests 3.12 and 3.13) on Apple Silicon or Linux x86_64 (torch has no NumPy 2 wheel for Intel Macs), and an agent platform (currently supported platforms are listed below) — already installed and authenticated against whatever LLM provider you intend to use.

<!-- md-shared:agents:start -->
| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Codex](https://github.com/openai/codex) | `npm i -g @openai/codex` (or `brew install --cask codex`) | `codex --version` |
| [opencode](https://github.com/sst/opencode) | See [opencode docs](https://opencode.ai/docs/) | `opencode --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |
<!-- md-shared:agents:end -->
<!-- md-shared:intro:end -->

## Installation

<!-- md-shared:install:start -->

```bash
python3.12 -m venv ~/.venvs/dsagt          # or: conda create -n dsagt python=3.12 && conda activate dsagt
source ~/.venvs/dsagt/bin/activate         # (Windows venv: ~\.venvs\dsagt\Scripts\activate)
pip install "git+https://github.com/AI-ModCon/dsagt.git"
dsagt --version                            # 0.2.1
```

This puts the `dsagt` CLI on your PATH. Create your first project — `dsagt init` is interactive (it walks you through the agent platform, project location, packaged knowledge collections, and skill sources) and sets up the knowledge base on first run:

```bash
dsagt init                      # interactive; pick agent, collections, sources
```

Then start dsagt (shorthand for starting your agent with the dsagt MCP server enabled), or open the project in VS Code:

```bash
dsagt start <my-project>   # ≈ cd ~/dsagt-projects/my-project && claude   (or your preferred agent)
```

Or, if you use a VS Code agent extension, just open the folder as a project in VS Code and start the agent — `dsagt init` already made the dsagt MCP server available via the native interface (e.g. for Claude, exposed in the project's `.mcp.json`).

To upgrade later, reinstall — re-running `dsagt init` reconfigures an existing project in place:

```bash
pip install --upgrade "git+https://github.com/AI-ModCon/dsagt.git"
```

> Pin to a specific release: e.g. `pip install "git+https://github.com/AI-ModCon/dsagt.git@0.2.0"`.
<!-- md-shared:install:end -->

## Quick Start

Explore DSAgt knowledge ingest, code registration, provenance, and explicit memory using the sample files in [`tests/smoke_test/`](tests/smoke_test/). Uses `claude`; substitute another agent (`goose` / `codex` / `opencode` / `cline`) if you prefer — the prompts are agent-agnostic. The same walkthrough is on the site: [Quick Start](https://ai-modcon.github.io/dsagt/quickstart/).

```bash
# 0. Install dsagt (see Installation above), then fetch the sample files:
curl -sL https://github.com/AI-ModCon/dsagt/archive/refs/heads/main.tar.gz \
    | tar xz --strip-components=2 dsagt-main/tests/smoke_test
export SMOKE_DIR="$PWD/smoke_test"   # a convenience variable for the prompts below

# 1. Create a project.  `dsagt init` is interactive — follow the menu to name it
#    `quickstart`, pick your agent, and choose knowledge collections + skill sources.
#    It sets up the knowledge base on first run (a ~130 MB local embedder downloads once).
dsagt init

# 2. Launch the agent in the project:
dsagt start quickstart   # …or: cd ~/dsagt-projects/quickstart && <your agent>
```

Inside the agent, paste these prompts one at a time (substitute the absolute path you exported as `$SMOKE_DIR` — the chat doesn't expand env vars):

<!-- md-shared:quickstart-prompts:start -->
1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > Register the CLI utility at `$SMOKE_DIR/csv_summary.py` as a code named `csv-summary` so we can reuse it.
3. > Use the `scan-directory` code from the registry to scan `$SMOKE_DIR/data/`.
4. > Run the `csv-summary` code on `$SMOKE_DIR/data/samples.csv` and tell me the columns, row count, and any columns with null values.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > Tell me what you remember about the samples dataset.
<!-- md-shared:quickstart-prompts:end -->

This exercised:

<!-- md-shared:quickstart-capabilities:start -->
| Prompt | Capability |
|---|---|
| 1 | `dsagt-server` (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | `dsagt-server` (`save_code_spec`) — writes `codes/csv-summary/SKILL.md` (a skill-standard dir), wrapping the executable with `dsagt-run` |
| 3–4 | `dsagt-run` provenance wrapper — records each execution to `trace_archive/` |
| 5–6 | Explicit memory (`kb_remember` → `.dsagt/explicit_memories.yaml`) + KB recall (`kb_get_memories`) |
<!-- md-shared:quickstart-capabilities:end -->

Step 4's null-column finding is the fact you store and recall in 5–6.

Exit the agent (`Ctrl+C` or `/exit`), then verify the artifacts and view traces:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{codes,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces land in a serverless SQLite store.  Browse them with:
dsagt traces quickstart
# Runs mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db
```

The same sequence runs unattended as `dsagt smoke-test --agent claude` (or `goose` / `codex` / `opencode` / `cline`), which checks each artifact and removes its project afterwards.

## Use Case Examples

[`use_cases/`](use_cases/) holds end-to-end domain walkthroughs — each covering data acquisition, code registration, pipeline construction, and agent-driven execution against a real dataset (genomics, cryo-EM, materials science, AI data-readiness). Browse one to see how the pieces fit on a concrete pipeline before building your own.

See the **[Use Cases documentation](https://ai-modcon.github.io/dsagt/use-cases/)** for the full catalog.

## Architecture

<!-- md-shared:architecture:start -->
![DSAgt architecture](docs/assets/architecture.png)

DSAgt provides a preconfigured agent platform with augmented capabilities for AI-ready scientific data processing and curation. Its capabilities are designed to complement rather than compete with the fast-moving agent platforms it runs on — Claude Code, Codex, Cline, opencode, etc. Capabilities are exposed to the agent through a central [MCP server](docs/mcp-servers.md): skill discovery and installation, data-processing code execution with provenance, knowledge-base extension and retrieval, and explicit and cross-session memory. Observability through MLflow traces, episodic memory management, and vector-store indexing are spawned by the server and run in the background. Every project also carries three base skills, `skill-creator`, `datacard-generator`, and `aidrin`, installed from their upstream repositories at init.

### Capabilities

**Code Registry** (`dsagt-server`)
The agent registers CLI data processing codes as skills (markdown files with YAML frontmatter) under `<project>/codes/`. DSAgt handles dependency installation via `uv run --with` (`uv` installs with dsagt) and wraps every execution with `dsagt-run` for provenance capture. The agent discovers codes via `search_registry`.

**[Knowledge Base](docs/knowledge-base.md)** (`dsagt-server`)
Hybrid semantic + BM25 search over ChromaDB collections partitioned by concern — code specs, the skill corpus, scientific documents, and per-project memory. A first `dsagt init` installs the built-in code specs and a default `genesis` skill corpus; the NeMo Curator reference collection and additional skill sources are also available (e.g. k-dense, anthropic), and new collections can be added for the documents of a specific scientific pursuit. The agent searches via `kb_search`, ingests via `kb_ingest`, and saves user-confirmed facts via `kb_remember`. Opt-in episodic memory distills each session turn into the per-project `session_memory` collection.

**[Provenance](docs/provenance.md)** (`dsagt-run`)
A wrapper around every registered-code execution. Records the command, arguments, exit code, duration, file counts, and truncated stderr to `<project>/trace_archive/<record_id>.json` and emits a `code.execute` span to the trace store. The server incrementally embeds those records into a `code_use` collection so past executions are retrievable, and the agent calls `reconstruct_pipeline` to render the trace archive as a reproducible workflow.

**[Observability](docs/observability.md)** (serverless MLflow)
Traces are stored in an MLflow SQLite file at `<project>/mlflow.db`. DSAgt server actions are recorded there and the agent's LLM-call traces are translated from the on-disk session transcript to MLflow's Claude autolog trace format. View with `dsagt traces <project>`.

**[Skills Discovery](docs/skills.md)**
DSAgt has MCP tools to connect to external GitHub skill repositories and search for skills that enhance scientific workflows. On top of the agent's own progressive disclosure of the skills already installed in its native skills directory, DSAgt maintains an extendable corpus of skills that can be searched and installed on demand — without flooding the agent's context window with skills it isn't using. The agent searches via `search_skills` and installs via `install_skill`.

**[AI-Readiness Check](docs/readiness.md)** (on by default)
AIDRIN as the check around every tabular pipeline stage. `aidrin` is a registered code in every project, and with the check on the agent's instructions say the check for a tabular stage is the `aidrin` skill's quality baseline, run before and after each data transformation with the delta reported.

**[Memory](docs/memory.md)**
DSAgt adds two memory extensions that complement the host agent's own memory (which distills session facts into a managed set of Markdown files loaded into context). *Explicit memory* records user-confirmed facts as YAML (`kb_remember` / `kb_get_memories`). *Episodic memory* (opt-in) keeps a vector store of semantically chunked turn blocks and searches them by successive filtering — first to a session, then by regex over the query's key themes, then a final vector ranking of what remains — so long, multi-session history can augment agent context without the noisy retrieval that undifferentiated episodic accrual would otherwise produce.
<!-- md-shared:architecture:end -->

### Project Layout

<!-- md-shared:project-layout:start -->
`dsagt init` prompts for the project location, defaulting to `~/dsagt-projects/<name>/` (enter `/data/runs` to place it at `/data/runs/<name>/`, or `.` for the current directory).

<!-- md-shared:project-tree:start -->
```
~/dsagt-projects/cheese-metagenome/
  .dsagt/                       # dsagt-internal state (hidden)
    config.yaml                 # project configuration (set by dsagt init)
    state.yaml                  # session log + memory cursor (owned by the MCP server)
    explicit_memories.yaml      # user-confirmed facts
  codes/<name>/                 # registered codes — skill-standard dirs (SKILL.md + scripts/)
  skills/                       # agent skills (SKILL.md + reference docs)
  trace_archive/                # code execution records (JSON, from dsagt-run)
  mlflow.db                     # serverless MLflow SQLite trace store
  kb_index/                     # knowledge base vector collections

  # Per-agent runtime config (one of, generated by dsagt init):
  #   claude:   CLAUDE.md, .mcp.json
  #   goose:    goose.yaml, .goosehints
  #   codex:    AGENTS.md, .codex-data/config.toml
  #   opencode: AGENTS.md, opencode.json
  #   cline:    .clinerules/, cline_mcp_settings.json (managed via cline mcp add)
```
<!-- md-shared:project-tree:end -->

Projects are registered in `~/dsagt-projects/projects.yaml`, so `dsagt <command> <name>` works from any directory. The project's data — knowledge base, trace store, registered codes, skills, audit records — is agent-agnostic: re-running `dsagt init` for an existing project and choosing a different agent switches platforms and keeps everything accumulated (it prompts before any destructive change). `dsagt rm <name>` deletes the project directory and its registry entry.
<!-- md-shared:project-layout:end -->

## CLI Reference

Page: [CLI](https://ai-modcon.github.io/dsagt/cli/).

<!-- md-shared:cli:start -->
| Command | Description |
|---------|-------------|
| `dsagt init` | Create or reconfigure a project — interactive menu for name, location, agent, knowledge collections, skill sources, the episodic-memory opt-in, and the AI-readiness check; sets up the KB and writes the per-agent MCP config |
| `dsagt start <name>` | Launch the agent in the project directory (equivalent to `cd <project> && <agent>`) |
| `dsagt info <name> [--json]` | Resolved config (with source per value) and a session/trace summary |
| `dsagt traces <name> [--port <n>]` | Open the MLflow trace viewer over the project's store (runs catch-up first, deep-links to the Traces tab, quiets the mlflow noise) |
| `dsagt list` | List all projects with agent and path |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project (and optionally delete its directory) |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode\|cline] [--all]` | End-to-end install verification; `--all` runs every agent in parallel |
<!-- md-shared:cli:end -->

For tests, troubleshooting, and other developer-facing material, see [docs/developer.md](docs/developer.md).
