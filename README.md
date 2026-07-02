# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

![DSAgt architecture](latex/architecture.png)

DSAgt connects an MCP-compatible AI coding agent to code registration, a semantic knowledge base, skills discovery and creation, execution provenance, and observability infrastructure. DSAgt provides data-pipeline scaffolding around a user's existing agent CLI or VS Code extension (Claude Code, Goose, Codex, …);

**Prerequisites:** Python 3.12 or 3.13, and one of the supported agent platforms below — already installed and authenticated against whatever LLM provider you intend to use.

<!-- md-shared:agents:start -->
| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Codex](https://github.com/openai/codex) | `npm i -g @openai/codex` (or `brew install --cask codex`) | `codex --version` |
| [opencode](https://github.com/sst/opencode) | See [opencode docs](https://opencode.ai/docs/) | `opencode --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |
<!-- md-shared:agents:end -->

## Installation

### For use (no development)

<!-- md-shared:install:start -->

```bash
python3.12 -m venv ~/.venvs/dsagt          # or: conda create -n dsagt python=3.12 && conda activate dsagt
source ~/.venvs/dsagt/bin/activate         # (Windows venv: ~\.venvs\dsagt\Scripts\activate)
pip install "git+https://github.com/AI-ModCon/dsagt.git"
dsagt --version                            # 0.2.0
```

This puts the `dsagt` CLI (and the `dsagt-run` / `dsagt-server` helpers) on your PATH. Create your first project — `dsagt init` is interactive (it walks you through the agent platform, project location, packaged knowledge collections, and skill-catalog sources) and sets up the knowledge base on first run:

```bash
dsagt init                      # interactive; pick agent, collections, sources
```

Then start your agent from the project directory, use dsagt cli, or open a VS code :

```bash
cd ~/dsagt-projects/my-project && claude   # …or: dsagt start my-project
```

To upgrade later, reinstall — re-running `dsagt init` reconfigures an existing project in place:

```bash
pip install --upgrade "git+https://github.com/AI-ModCon/dsagt.git"
```

> Pin to a specific release once tags are published, e.g. `pip install "git+https://github.com/AI-ModCon/dsagt.git@v0.2.0"`.
<!-- md-shared:install:end -->

### For development

Clone the repo and use `uv` (editable install with the full test suite) — see [Quick Start](#quick-start) below.

## Quick Start

Explore DSAgt knowledge ingest, code registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](tests/smoke_test/). Uses `claude`; substitute another agent (`goose` / `codex` / `opencode` / `cline`) if you prefer — the prompts are agent-agnostic.

```bash
# 0. Install
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync                      # add --all-groups for the test suite
source .venv/bin/activate    # so `dsagt` is on PATH

# A convenience variable for the demo paths below (not a normal dsagt step)
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# 1. Create a project.  `dsagt init` is interactive — follow the menu to name it
#    `quickstart`, pick your agent, and choose knowledge collections + skill sources.
#    It sets up the knowledge base on first run (a ~130 MB local embedder downloads once).
dsagt init

# 2. Launch the agent in the project:
dsagt start quickstart   # …or: cd ~/dsagt-projects/quickstart && <your agent>
```

Inside the agent, paste these prompts one at a time (substitute the absolute path you exported as `$SMOKE_DIR` — the chat doesn't expand env vars):

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > Register the csvkit CLI codes `csvcut`, `csvgrep`, `csvstat`, and `csvlook`.
3. > Use the `scan_directory` code from the registry to scan `$SMOKE_DIR/data/`.
4. > Summarize `samples.csv` — columns, row count, quality issues using csvkit codes from the registry.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > Tell me what you remember about the samples dataset.

This exercised:

| Prompt | Capability |
|---|---|
| 1 | `dsagt-server` (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | `dsagt-server` (`save_code_spec`) — writes `codes/csvcut.md`, `codes/csvgrep.md`, etc. (one per registered code) |
| 3 | `dsagt-run` provenance wrapper — records the execution to `trace_archive/` |
| 5–6 | Explicit memory (`kb_remember` → `.dsagt/explicit_memories.yaml`) + KB recall (`kb_get_memories`) |

Exit the agent (`Ctrl+C` or `/exit`), then verify the artifacts and view traces:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{codes,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces land in a serverless SQLite store.  Browse them with:
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db
```

The same flow runs non-interactively via `dsagt smoke-test --agent claude` (or `goose` / `codex` / `opencode` / `cline`), which asserts each artifact is present.

### Knowledge base setup

`dsagt init` sets up the project's knowledge base with three kinds of collection:

- **Code Specs** — DSAgt's bundled code specs, always set up so the agent finds them via `search_registry` from the first session.
- **Skill Catalogs** — the skill-catalog sources you pick at init (default `genesis`) are cloned and indexed so `search_skills` returns installable skills. The bundled `skill-creator` is discovered natively by the agent.
- **Knowledge Collections** — optional reference document sets you pick at init (`nemo_curator`, `aidrin`), downloaded and indexed for data-curation domain knowledge.

The default embedder is a local sentence-transformers model (~130 MB of weights downloaded on first run, CPU-side, no API key).

## Use Case Examples

End-to-end walkthroughs for representative scientific and data-readiness scenarios live in [`use_cases/`](use_cases/). Each one covers data acquisition, code registration, pipeline construction, and agent-driven execution against a real dataset.

| Use case | Domain | Guide |
|----------|--------|-------|
| Microbial isolate processing | Genomics — short-read QC and assembly with `fastp` + `megahit` | [isolate_demo.md](use_cases/microbial_isolates/isolate_demo.md) |
| Cryo-EM data curation | Structural biology — EMPIAR-10017 β-galactosidase micrographs via CryoPPP | [cryoem_demo.md](use_cases/cryoem/cryoem_demo.md) |
| ISAAC / VASP workflows | Materials science — DFT input/output handling with VASP | [use_cases/isaac_vasp/](use_cases/isaac_vasp/) |
| AIDRIN readiness gate (cryo-EM) | AI data readiness — `aidrin` quality metrics before/after cryo-EM curation | [cryoem_readiness_demo.md](use_cases/aidrin_readiness/cryoem_readiness_demo.md) |
| AIDRIN full feature tour | AI data readiness — all 15 `aidrin` metrics (quality, fairness, privacy) on UCI Adult | [aidrin_full_tour_demo.md](use_cases/aidrin_readiness/aidrin_full_tour_demo.md) |

## Project Directory

`dsagt init` prompts for the project location, defaulting to `~/dsagt-projects/<name>/` (e.g. enter `/data/runs` to place it at `/data/runs/my-project/`, or `.` for the current directory).

Projects are registered in `~/dsagt-projects/projects.yaml` so `dsagt info <name>` works from any directory. The project's data — knowledge base, trace store, registered codes, skills, audit records — is agent-agnostic, so re-running `dsagt init` for the same project and choosing a different agent switches platforms while preserving everything you've accumulated (it prompts before any destructive change).

```
~/dsagt-projects/cheese-metagenome/
  .dsagt/                       # dsagt-internal state (hidden)
    config.yaml                 # project configuration (set by dsagt init)
    state.yaml                  # session log + memory cursor (owned by the MCP server)
    explicit_memories.yaml      # user-confirmed facts
  codes/                        # registered CLI code specs (markdown + YAML frontmatter)
  codes/scripts/                # agent-written code scripts
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

### MCP Server

DSAGT exposes a single MCP server, **`dsagt-server`**, that an agent connects to once. Its main tool groups are:

- **Registry** — Code registration and dependency installation. Codes are markdown files with YAML frontmatter under `<project>/codes/`. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies. The agent discovers codes via `search_registry`.
- **Knowledge** — Semantic search over indexed ChromaDB document collections. Background jobs handle long ingest operations. The agent searches via `kb_search`, ingests via `kb_ingest`, and saves user-confirmed facts via `kb_remember`.

### Codes and Skills

**Codes** are CLI executables defined as markdown files with YAML frontmatter in `<project>/codes/`. The agent registers new codes via the MCP server's `save_code_spec`.

**Skills** are instruction-based agent workflows — a directory with a `SKILL.md` and optional reference docs. They come in two tiers:

- **Installed** skills live in `<project>/skills/` (DSAgt ships a bundled `skill-creator`; domain skills like the MODCON datacard generator are installed from the `genesis` catalog). These are mirrored into the agent's native skill directory (e.g. `.claude/skills/`, `.agents/skills/`) at `dsagt init`/`start`, where the agent auto-discovers and auto-invokes them — no `search_skills` needed (that covers only the catalog tier below).
- **Catalog** skills come from external Git repositories — GitHub *or* GitLab — indexed into a searchable catalog the agent browses with `search_skills` but that is **not** loaded into its context (so a catalog can hold thousands of skills). The agent enables a source with `add_skill_source(...)`, finds skills with `search_skills(...)`, then copies one into the project with `install_skill(...)`.

The catalog is **opt-in**: a source must be synced before its skills are searchable. Curated named sources ship out of the box — `k-dense-ai`, `anthropic`, `antigravity`, `composio`, and `genesis` (the OSTI GENESIS catalog: HPC, HuggingFace, LangChain, OpenAI, plasma-sim, and more) — and any Git URL or `owner/repo` works too. Manage catalogs from the agent with `list_skill_sources` / `add_skill_source` / `search_skills` / `install_skill`.

![DSAgt skills routing](latex/skills-routing.png)

The diagram traces a skill's lifecycle: **discovery** — browse the catalog with `search_skills` for skills the agent doesn't yet have → **install** — `install_skill` copies one into the project → **use** — the agent auto-discovers installed skills natively and invokes them by relevance (and authors new ones with the bundled `skill-creator`). The diagram source is [`latex/skills-routing.tex`](latex/skills-routing.tex).

### Knowledge Base

The agent searches these collections semantically:

| Collection | Source | Populated by |
|---|---|---|
| **Code Specs** | Bundled CLI code specs | `dsagt init` (always set up) |
| **Skill Catalogs** | Installable skills from external repos (one per source) | `dsagt init` (chosen sources) + `add_skill_source` |
| **Knowledge Collections** | NeMo Curator + AIDRIN reference collections; user-ingested docs | `dsagt init` (chosen collections) + agent's `kb_ingest` |
| **Explicit Memory** | User-confirmed facts | Agent's `kb_remember` (also written to `<project>/.dsagt/explicit_memories.yaml`); the agent fetches via `kb_get_memories` on demand, not auto-loaded at session start |
| **Code Execution Records** | `dsagt-run` execution traces | `dsagt-run` writes JSON to `<project>/trace_archive/`; indexed for search during the session, and before `reconstruct_pipeline` |
| **Episodic Memory** | Captured session turns | **Opt-in** (enabled in the `dsagt init` menu): DSAgt captures each completed turn into `session_memory` during the session (mechanical chunk + embed). Retrieval is recency-weighted. |

The embedding backend is local (sentence-transformers, CPU-side, no API key).

The agent searches via `kb_search` and writes via `kb_ingest` / `kb_remember`. Registered codes have their own `search_registry` route over the same backend. Installed skills are discovered natively by the agent; enabling external skill catalogs adds one collection per source, which `search_skills` browses for installable skills.

### Memory

DSAgt has two memory types, both retrievable via `kb_search` / `kb_get_memories`:

- **Explicit memory** — user-confirmed facts the agent writes with `kb_remember` (mirrored to `<project>/.dsagt/explicit_memories.yaml`). Always on; degrades to pure-YAML if the vector store is unavailable.
- **Episodic memory** — automatic session capture, **opt-in** (enabled in the `dsagt init` menu, off by default). As the session runs, DSAgt reads the agent's transcript and captures each completed turn into the `session_memory` collection — a fast, local chunk-and-embed pass.

  Retrieval is **recency-weighted** (`episodic.recency_half_life_days`): a newer turn edges out a stale one, but as a bounded boost — a strongly-relevant old turn is never buried.

### Observability

Self-logging goes to a serverless MLflow SQLite store at `<project>/mlflow.db`. Browse it with `mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db`. The trace view shows:

- **Knowledge base operations** — `kb.search` / `kb.embed` / `kb.index_search` / `kb.rerank` span trees with per-phase timing.
- **Code executions** — `code.execute` spans with exit code, duration, file counts, truncated stderr. Full payload in `trace_archive/<record_id>.json`.
- **Registry events** — `save_code_spec`, `install_dependencies`, `reconstruct_pipeline` spans.
- **Agent traces** — recovered from the on-disk session transcript, so prompts, responses, and tool calls land in the store for every supported agent.

Each launch gets a session id that every span carries, so you can filter the trace view by session. The code-execution records on disk are the provenance record — the agent calls `reconstruct_pipeline` to render them as a reproducible bash script or Snakemake workflow.

## CLI Reference

| Command | Description |
|---------|-------------|
| `dsagt init` | Create or reconfigure a project — interactive menu for name, location, agent, knowledge collections, skill sources, and the episodic-memory opt-in; sets up the KB and writes the per-agent MCP config |
| `dsagt start <name>` | Launch the agent in the project directory (equivalent to `cd <project> && <agent>`) |
| `dsagt info <name> [--json]` | Resolved config (with source per value) and a session/trace summary |
| `dsagt list` | List all projects with agent and path |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project (and optionally delete its directory) |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode\|cline]` | End-to-end install verification |

> **Deprecated `dsagt init` flags (backcompat).** The pre-menu flags still work for scripts/CI but are deprecated in favor of the interactive menu: `<name>` (positional), `--agent <platform>`, `--location <path>`, `--include … | --exclude …`, and `--episodic`. Passing any of them skips the corresponding prompt; new usage should prefer bare `dsagt init`.

Skill catalogs are managed from the agent via the MCP tools (`add_skill_source` / `search_skills` / `install_skill`), and traces are viewed with `mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db`.

For tests, troubleshooting, and other developer-facing material, see [developer.md](developer.md).
