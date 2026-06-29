# DSAgt

**D**ata**S**mith **Ag**en**t** — AI-assisted data pipeline builder.

![DSAgt architecture](latex/architecture.png)

DSAgt connects an MCP-compatible AI coding agent to tool registration, a semantic knowledge base, skills discovery and creation, execution provenance, and observability infrastructure. DSAgt provides data-pipeline scaffolding around a user's existing agent CLI or VS Code extension (Claude Code, Goose, Codex, …);

**Prerequisites:** Python 3.12 or 3.13, and one of the supported agent platforms below — already installed and authenticated against whatever LLM provider you intend to use. ([uv](https://github.com/astral-sh/uv) is only needed for the development install.)

<!-- md-shared:agents:start -->
| Agent | Install | Verify |
|-------|---------|--------|
| [Claude Code](https://github.com/anthropics/claude-code) | `npm i -g @anthropic-ai/claude-code` | `claude --version` |
| [Goose](https://github.com/block/goose) | See [Goose docs](https://github.com/block/goose#installation) | `goose --version` |
| [Codex](https://github.com/openai/codex) | `npm i -g @openai/codex` (or `brew install --cask codex`) | `codex --version` |
| [opencode](https://github.com/sst/opencode) | See [opencode docs](https://opencode.ai/docs/) | `opencode --version` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | `npm i -g @roo-code/cli` | `roo --version` |
| [Cline](https://github.com/cline/cline) | `npm i -g cline` | `cline --version` |
<!-- md-shared:agents:end -->

## Installation

### For use (no development)

<!-- md-shared:install:start -->
If you just want to *run* DSAgt against your own data and agent — no repo checkout, no `uv` — install it straight from GitHub into a virtual environment. Any Python 3.12/3.13 environment works (`venv`, conda, etc.); only the `pip install git+…` step is officially supported.

```bash
python3.12 -m venv ~/.venvs/dsagt          # or: conda create -n dsagt python=3.12 && conda activate dsagt
source ~/.venvs/dsagt/bin/activate         # (Windows venv: ~\.venvs\dsagt\Scripts\activate)
pip install "git+https://github.com/AI-ModCon/dsagt.git"
dsagt --version                            # 0.2.0
```

This puts the `dsagt` CLI (and the `dsagt-run` / `dsagt-server` helpers) on your PATH. Create your first project — `dsagt init` is interactive (it walks you through the agent platform, project location, packaged knowledge collections, and skill-catalog sources) and provisions the shared knowledge base on first run:

```bash
dsagt init                      # interactive; pick agent, collections, sources
```

Then start your agent from the project directory, use dsagt cli, or open a VS code :

```bash
cd ~/dsagt-projects/my-project && claude   # …or: dsagt start my-project
```

To upgrade later, reinstall — re-running `dsagt init my-project` reconfigures an existing project in place:

```bash
pip install --upgrade "git+https://github.com/AI-ModCon/dsagt.git"
```

> Pin to a specific release once tags are published, e.g. `pip install "git+https://github.com/AI-ModCon/dsagt.git@v0.2.0"`.
<!-- md-shared:install:end -->

### For development

Clone the repo and use `uv` (editable install with the full test suite) — see [Quick Start](#quick-start) below.

## Quick Start

Explore DSAgt knowledge ingest, tool registration, provenance, and explicit memory using the mock project in [`tests/smoke_test/`](tests/smoke_test/). Uses `claude`; substitute another agent (`goose` / `codex` / `opencode` / `roo` / `cline`) if you prefer — the prompts are agent-agnostic.

```bash
# 0. Install
git clone https://github.com/AI-ModCon/dsagt.git
cd dsagt
uv sync                      # add --all-groups for the test suite
source .venv/bin/activate    # so `dsagt` is on PATH

# A convenience variable for the demo paths below (not a normal dsagt step)
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# 1. Create a project called quickstart.  Interactive `dsagt init` prompts for
#    the agent, location, knowledge collections, and skill sources, and
#    provisions the shared knowledge base on first run (a ~130 MB local
#    embedder downloads once).  `--agent` makes it non-interactive:
dsagt init quickstart --agent claude

# 2. Launch the agent from the project directory:
cd ~/dsagt-projects/quickstart && claude   # …or: dsagt start quickstart
```

Inside the agent, paste these prompts one at a time (substitute the absolute path you exported as `$SMOKE_DIR` — the chat doesn't expand env vars):

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > Register the csvkit CLI tools `csvcut`, `csvgrep`, `csvstat`, and `csvlook`.
3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`.
4. > Summarize `samples.csv` — columns, row count, quality issues using csvkit tools from the registry.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > Tell me what you remember about the samples dataset.

What this exercised:

| Prompt | Layer |
|---|---|
| 1 | `dsagt-server` (`kb_ingest`) — chunks and indexes docs into ChromaDB |
| 2 | `dsagt-server` (`save_tool_spec`) — writes `tools/csvcut.md`, `tools/csvgrep.md`, etc. (one per registered tool) |
| 3 | `dsagt-run` provenance wrapper — records the execution layer to `trace_archive/` |
| 5–6 | Explicit memory (`kb_remember` → `.dsagt/explicit_memories.yaml`) + KB recall (`kb_get_memories`) |

Exit the agent (`Ctrl+C` or `/exit`), then verify the artifacts and view traces:

```bash
dsagt info quickstart                       # config + a session/trace summary
ls ~/dsagt-projects/quickstart/{tools,trace_archive}
cat ~/dsagt-projects/quickstart/.dsagt/explicit_memories.yaml

# Traces land in a serverless SQLite store — no server to run.  Browse them with:
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/quickstart/mlflow.db
```

The same flow runs non-interactively via `dsagt smoke-test --agent claude` (or `goose` / `codex` / `opencode`), which asserts each artifact is present.

### Knowledge base provisioning

`dsagt init` provisions the project's knowledge base. The shared, machine-wide collections live under `~/dsagt-projects/kb_index/` and are built once (the first project on a machine pays the cost), then copied into each project's `kb_index/`:

- **Tool Specs** — DSAgt's bundled tool specs from `src/dsagt/tools/`, tagged `source: bundled`, always provisioned so the agent finds them via `search_registry` from the first session.
- **Skill Catalogs** — the skill-catalog sources you check at init (default `genesis`) are cloned and frontmatter-indexed so `search_skills` returns installable skills. The bundled `skill-creator` is auto-discovered natively by the agent, not indexed.
- **Knowledge Collections** — optional reference corpora you check at init (`nemo_curator`, `aidrin`), downloaded and embedded for data-curation domain knowledge.

The default embedder is a local sentence-transformers model (~130 MB of weights downloaded on first run, CPU-side, no API key).

## Use Case Examples

End-to-end walkthroughs for representative scientific domains live in [`use_cases/`](use_cases/). Each one covers data acquisition, tool registration, pipeline construction, and agent-driven execution against a real dataset.

| Use case | Domain | Guide |
|----------|--------|-------|
| Microbial isolate processing | Genomics — short-read QC and assembly with `fastp` + `megahit` | [isolate_demo.md](use_cases/microbial_isolates/isolate_demo.md) |
| Cryo-EM data curation | Structural biology — EMPIAR-10017 β-galactosidase micrographs via CryoPPP | [cryoem_demo.md](use_cases/cryoem/cryoem_demo.md) |
| ISAAC / VASP workflows | Materials science — DFT input/output handling with VASP | [use_cases/isaac_vasp/](use_cases/isaac_vasp/) |

## Project Directory

Default location: `~/dsagt-projects/<name>/`. Override with `--location`:

```bash
dsagt init my-project --agent claude --location /data/runs   # /data/runs/my-project/
dsagt init my-project --agent claude --location .            # ./my-project/
```

Projects are registered in `~/dsagt-projects/projects.yaml` so `dsagt info <name>` works from any directory. The data layer (knowledge base, trace store, registered tools, skills, audit records) is agent-agnostic, so re-running `dsagt init <same-name>` and choosing a different agent switches platforms while preserving everything you've accumulated (it prompts before any destructive change).

```
~/dsagt-projects/cheese-metagenome/
  .dsagt/                       # dsagt-internal state (hidden)
    config.yaml                 # project configuration (set by dsagt init)
    state.yaml                  # session log + memory cursor (owned by the MCP server)
    explicit_memories.yaml      # user-confirmed facts
  tools/                        # registered CLI tool specs (markdown + YAML frontmatter)
  tools/code/                   # agent-written tool scripts
  skills/                       # agent skills (SKILL.md + reference docs)
  trace_archive/                # tool execution records (JSON, from dsagt-run)
  mlflow.db                     # serverless MLflow SQLite trace store
  kb_index/                     # knowledge base vector collections

  # Per-agent runtime config (one of, generated by dsagt init):
  #   claude:   CLAUDE.md, .mcp.json
  #   goose:    goose.yaml, .goosehints
  #   codex:    AGENTS.md, .codex-data/config.toml
  #   opencode: AGENTS.md, opencode.json
  #   roo:      .roomodes, .roo/mcp.json
  #   cline:    .clinerules/, cline_mcp_settings.json (managed via cline mcp add)
```

### MCP Server

DSAGT exposes a single MCP server, **`dsagt-server`**, that an agent connects to once. It bundles two concern areas:

- **Registry** — Tool registration and dependency installation. Tools are markdown files with YAML frontmatter under `<project>/tools/`. Executables are wrapped with `dsagt-run` for provenance and `uv run --with` for Python dependencies. The agent discovers tools via `search_registry`.
- **Knowledge** — Semantic search over indexed ChromaDB document collections. Background jobs handle long ingest operations. The agent searches via `kb_search`, ingests via `kb_ingest`, and saves user-confirmed facts via `kb_remember`.

### Tools and Skills

**Tools** are CLI executables defined as markdown files with YAML frontmatter in `<project>/tools/`. The agent registers new tools via the MCP server's `save_tool_spec`.

**Skills** are instruction-based agent workflows — a directory with a `SKILL.md` and optional reference docs. They come in two tiers:

- **Installed** skills live in `<project>/skills/` (DSAgt ships a bundled `skill-creator`; domain skills like the MODCON datacard generator are installed from the `genesis` catalog). These are mirrored into the agent's native skill directory (e.g. `.claude/skills/`, `.agents/skills/`) at `dsagt init`/`start`, where the agent auto-discovers and auto-invokes them — no `search_skills` needed (that covers only the catalog tier below).
- **Catalog** skills come from external Git repositories — GitHub *or* GitLab — indexed into a searchable catalog the agent browses with `search_skills` but that is **not** loaded into its context (so a catalog can hold thousands of skills). The agent enables a source with `add_skill_source(...)`, finds skills with `search_skills(...)`, then copies one into the project with `install_skill(...)`.

The catalog is **opt-in**: a source must be synced before its skills are searchable. Curated named sources ship out of the box — `k-dense-ai`, `anthropic`, `antigravity`, `composio`, and `genesis` (the OSTI GENESIS catalog: HPC, HuggingFace, LangChain, OpenAI, plasma-sim, and more) — and any Git URL or `owner/repo` works too. Manage catalogs from the agent with `list_skill_sources` / `add_skill_source` / `search_skills` / `install_skill`.

![DSAgt skills routing](latex/skills-routing.png)

Skill handling runs through one service over two stores. **`SkillRouter`** is the single skill-MCP entry point — every skill tool routes through it: `add_skill_source` / `list_skill_sources` manage repos, `search_skills` queries the catalog, `install_skill` adopts a catalog skill into the project. **Registration** pulls skills from External Skills Repos (the curated `k-dense-ai` / `anthropic` / `antigravity` / `composio` / `genesis` sources, *or any git URL*) into the **Skills Catalog** — a federated, searchable store of *not-yet-installed* skills (semantic search, with a zero-dependency keyword fallback when no embedder is configured). **Discovery** is the catalog's irreplaceable job: surfacing skills the agent doesn't yet have, which native discovery can't see. **Progressive exposure** is native: the **Skill Directory** holds the project's installed + created skills in each agent's own skill dir (`.claude/skills`, `.agents/skills`, `.cline/skills`, `.roo/skills`), where the agent auto-discovers and model-invokes them by relevance — and authors new ones via the bundled **`skill-creator`** skill. The diagram source is [`latex/skills-routing.tex`](latex/skills-routing.tex).

### Knowledge Base

Six independently-partitioned ChromaDB collections hold everything the agent searches semantically. The first three are machine-wide (built once under `~/dsagt-projects/kb_index/` and copied into each project); the last three are per-project (under `<project>/kb_index/`, filled automatically during use):

| Collection | Source | Populated by |
|---|---|---|
| **Tool Specs** | Bundled CLI tool specs in `src/dsagt/tools/` | `dsagt init` (always provisioned) |
| **Skill Catalogs** | Installable skills from external repos (one `skills_catalog__<slug>` per source), frontmatter-indexed | `dsagt init` (chosen sources) + `add_skill_source` |
| **Knowledge Collections** | NeMo Curator + AIDRIN reference corpora; user-ingested docs | `dsagt init` (chosen collections) + agent's `kb_ingest` |
| **Explicit Memory** | User-confirmed facts | Agent's `kb_remember` (also written to `<project>/.dsagt/explicit_memories.yaml`); the agent fetches via `kb_get_memories` on demand, not auto-loaded at session start |
| **Tool Use Records** | `dsagt-run` execution traces | `dsagt-run` writes JSON to `<project>/trace_archive/`; embedded into ChromaDB incrementally by the MCP server's heartbeat (idempotent), and on demand before `reconstruct_pipeline` |
| **Episodic Memory** | Distilled session facts | **Opt-in** (`dsagt init --episodic`): the heartbeat distills each completed turn into tagged, ≤1-sentence facts via a local LLM judge (Tier-1), falling back to mechanical chunk+tag+embed (Tier-0) on judge failure. Retrieval is recency-weighted. |

The embedding backend is local (sentence-transformers, CPU-side, no API key).

The agent searches via `kb_search` and writes via `kb_ingest` / `kb_remember`. Registered tools have their own `search_registry` route over the same backend. Installed skills are auto-discovered natively by the agent (not indexed); enabling external skill catalogs adds one `skills_catalog__<slug>` collection per source, which `search_skills` browses for installable skills.

### Memory

DSAgt has two memory types, both retrievable via `kb_search` / `kb_get_memories`:

- **Explicit memory** — user-confirmed facts the agent writes with `kb_remember` (mirrored to `<project>/.dsagt/explicit_memories.yaml`). Always on; degrades to pure-YAML if the vector store is unavailable.
- **Episodic memory** — automatic session facts, **opt-in** (`dsagt init --episodic`, off by default). The MCP server's in-session heartbeat reads the agent's transcript and, each completed turn, distills it into a few tagged, ≤1-sentence facts in the `session_memory` collection. Two tiers:
  - **Tier-1 (default when enabled)** — a small **local** LLM "judge" (`Qwen2.5-1.5B`, grammar-constrained) classifies each fact against a closed tag taxonomy (stock "AI-data-ready" tags + any project `--domain-tags`) and condenses it. Local-by-default — **no API key, no cost** — but the GGUF model (~1 GB) downloads on first use and inference uses CPU (~1 s per fact-bearing turn, off the agent's critical path).
  - **Tier-0 (fallback)** — mechanical chunk + keyword-tag + embed, no LLM; used automatically if the judge fails, so a turn is never lost.

  Retrieval is **recency-weighted** (`episodic.recency_half_life_days`, default 14): a newer fact edges out a stale one, but as a bounded boost — a strongly-relevant old fact is never buried. Enabling the local judge needs no setup beyond the bundled `llama-cpp-python` (a core dependency, installed from a prebuilt CPU wheel).

### Observability

Self-logging goes to a serverless MLflow SQLite store at `<project>/mlflow.db` — no server to run. Browse it with `mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db`. The trace view shows:

- **Knowledge base operations** — `kb.search` / `kb.embed` / `kb.index_search` / `kb.rerank` span trees with per-phase timing.
- **Tool executions** — `tool.execute` spans with exit code, duration, file counts, truncated stderr. Full payload in `trace_archive/<record_id>.json`.
- **Registry events** — `save_tool_spec`, `install_dependencies`, `reconstruct_pipeline` spans.
- **Agent traces** — recovered post-hoc from the on-disk session transcript by the MCP server's in-session heartbeat (a per-agent reader → canonical-trace translator → MLflow sink), so prompts/responses/tool-calls land in the store for every supported agent, not just claude. (claude additionally wires an `mlflow autolog` Stop hook at `dsagt init`.)

The MCP server mints a session id per launch into `<project>/.dsagt/state.yaml`, and every span carries it for filtering. Tool execution records on disk provide the canonical provenance chain — the agent calls `reconstruct_pipeline` to render the trace archive as a reproducible bash script or Snakemake workflow.

## CLI Reference

| Command | Description |
|---------|-------------|
| `dsagt init [<name>]` | Create or reconfigure a project — interactive: agent, location, knowledge collections, skill sources, and the episodic-memory opt-in; provisions the KB and writes the per-agent MCP config |
| `dsagt init <name> --agent <platform> [--location <path>] [--include … \| --exclude …] [--episodic [--domain-tags "a,b"]]` | Same, non-interactively (for scripts/CI); `--episodic` enables episodic memory (downloads the ~1 GB local judge on first use) |
| `dsagt start <name>` | Launch the agent in the project directory (equivalent to `cd <project> && <agent>`) |
| `dsagt info <name> [--json]` | Resolved config (with source per value) and a session/trace summary |
| `dsagt list` | List all projects with agent and path |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project (and optionally delete its directory) |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode]` | End-to-end install verification |

Skill catalogs are managed from the agent via the MCP tools (`add_skill_source` / `search_skills` / `install_skill`), and traces are viewed with `mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db`.

For tests, troubleshooting, and other developer-facing material, see [developer.md](developer.md).
