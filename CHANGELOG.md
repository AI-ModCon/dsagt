# Changelog

All notable changes to DSAgt are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-08

A large release. It adds an **external skill-catalog system**, consolidates the
agent-facing surface into a **single `dsagt-server`**, recovers **observability
and episodic memory without a proxy** — both read from each agent's own on-disk
transcript rather than intercepted LLM traffic — renames registered executables
to **codes**, and makes codes and skills natively discoverable as soon as they
are added to a project.

Agents discover and install skills from federated GitHub/GitLab catalogs
(Genesis, Anthropic, K-Dense, and more) and author their own; installed skills
are picked up through each agent's *native* `SKILL.md` auto-discovery, so
`search_skills` is reserved for the one job native discovery can't do: browsing
the corpus of skills you haven't installed yet. In parallel, the registry and
knowledge MCP servers — two processes that each loaded their own embedder and
opened their own ChromaDB — collapse into one `dsagt-server`: one embedder, one
Chroma owner, one connection per agent.

**Upgrading from 0.1.0.** There is no automatic migration — adopting 0.2.0 is
rebuild-not-migrate, and no project data changes:
- Re-run `dsagt start <project>` for each existing project; it regenerates the
  per-agent MCP config to point at the single `dsagt-server`.
- For **cline** only, delete `<project>/.cline-data` first — `cline mcp add`
  has no remove, so the stale `dsagt-registry`/`dsagt-knowledge` entries would
  otherwise linger next to the new one.
- Codes, skills, the KB index, traces, and memory all carry over untouched.

### Added
- **Proxy-free trace pipeline.** An in-session heartbeat in `dsagt-server`
  reads the agent's transcript and logs prompts, responses, tool calls, and
  token usage to the serverless MLflow store — every supported agent (claude,
  codex, goose, opencode, cline); no proxy, no OTel routing, no credentials.
  DSAgt's own spans flow to the same store, tagged `dsagt.source` so the debug
  view filters apart from agent traces.
- **`dsagt traces <project>`** opens the MLflow viewer over the project's
  store: runs trace catch-up first, deep-links to the Traces tab, and
  suppresses MLflow's console noise.
- **Episodic memory (opt-in).** `dsagt init --episodic`: the heartbeat
  mechanically chunks, keyword-tags, and embeds each completed turn into
  `session_memory` (no LLM); retrieval is recency-weighted
  (`episodic.recency_half_life_days`, default 14).
- **External skill catalogs**: discover and install agent skills from GitHub /
  GitLab sources via `add_skill_source`, `search_skills`, and `install_skill`
  (plus the `dsagt skills sync/add/list/search` CLI), backed by per-source
  ChromaDB collections. Curated sources are provided out of the box
  (`k-dense-ai`, `anthropic`, `antigravity`, `composio`, `genesis`); any git URL
  / `owner/repo` also works.
- **Genesis catalog integration**: the curated `genesis` source (OSTI GitLab,
  `gitlab.osti.gov/genesis/genesis-skills`) makes the BASE-Data / ModCon skills
  — `datacard-generator` (frontmatter name `generating-datacards`),
  `croissant-validator`, `hdmf-schema-builder` — pullable on demand
  (`dsagt skills add <project> genesis`, then `install_skill`) rather than
  built into the package, alongside the rest of the Genesis catalog (HPC/Slurm,
  HuggingFace, LangChain, and more).
- **Immediate native discovery.** Installing or creating a skill, or
  registering a code, mirrors it into each agent's native skills directory
  (`.claude/skills/`, `.agents/skills/`, …) right away and re-mirrors at
  `dsagt init` / `start`, so every supported agent auto-discovers it with no
  restart.
- **`skill-creator`** built-in skill for authoring new skills from the Anthropic
  template.
- **Source-qualified catalog install**: when the same skill name exists in more
  than one synced source, install a specific one with a `<source-slug>/<skill>`
  name (via `install_skill` or `dsagt skills add <project> <slug>/<skill>`)
  instead of dead-ending on the ambiguity guard.
- **Keyword fallback** for `search_skills`: a zero-dependency token-overlap
  scorer so catalog search works even when no embedding model is configured.
- **License / attribution provenance on install**: installing a catalog skill
  preserves upstream `LICENSE` / `NOTICE` files and stamps a `PROVENANCE.txt`
  recording the source repo and path into the installed skill directory.
- **New use cases**: `isaac_skills_demo` — an end-to-end, skill-oriented
  walkthrough that drives a real agent through syncing a catalog, installing a
  skill, and converting mock VASP output into an Isaac record (prompts + mock
  data included) — plus tokamak-stability and AIDRIN data-readiness walkthroughs.
- **Install-from-GitHub instructions** for non-developers (`pip install
  git+https://github.com/AI-ModCon/dsagt.git` into any Python 3.12/3.13
  environment) in the README and docs.

### Changed
- **The two MCP servers are now one `dsagt-server`** — one shared
  `KnowledgeBase`/embedder, one MCP entry per agent, one trace `service.name`.
  The tool surface is organized by concern (registry / knowledge / memory /
  skill) behind the single server.
- **"Tools" are now "codes."** Registered CLI executables are *codes*
  throughout — `<project>/codes/`, `save_code_spec`, `dsagt-run --code`, the
  `code_use` collection, `code.execute` spans — reserving "tool" for the
  MCP/agent sense.
- **Codes share the skill envelope.** A code is a skill-standard dir
  (`codes/<name>/SKILL.md`); built-in codes are copied into `<project>/codes/`
  at init — one place, one format.
- **Skill discovery is catalog-only**: installed and built-in skills are
  discovered natively by every supported agent, so `search_skills` covers only
  the not-yet-installed external catalog. Catalogs are indexed on frontmatter
  (name + description + tags) rather than the full SKILL.md body.
- **Agents are pre-authenticated.** DSAgt never touches provider credentials;
  all credential-hint machinery is gone.
- **Code-use indexing is incremental.** `dsagt-run` records embed into
  `code_use` on the heartbeat and on demand before `reconstruct_pipeline`,
  instead of waiting for the next session's startup catch-up.
- **Blocking tool work runs off the event loop.** `run_command`, dependency
  installs, registry search, and pipeline reconstruction execute in worker
  threads, so a long call no longer stalls the trace heartbeat or concurrent
  tools.
- Scientific KB collections (`nemo_curator`, `aidrin`) index documentation and
  papers only, not library source — faster first-run ingestion; source is
  better served by the agent's own file search than by vector retrieval.
- `search_skills` reports when no external catalog is synced instead of a bare
  "no match", and `list_skill_sources` flags each known source as
  `synced`/available with its indexed count.
- The package version is single-sourced from `dsagt.__version__` (pyproject
  reads it via setuptools dynamic metadata).
- Documentation home page (`docs/index.md`) pulls the supported-agents table
  and install instructions directly from the README via the
  `mkdocs-include-markdown` plugin, so the two no longer drift.

### Removed
- **BREAKING:** the `dsagt-registry-server` and `dsagt-knowledge-server` console
  scripts, replaced by `dsagt-server` (see **Upgrading** above).
- The episodic **LLM-judge** distillation layer and **outlier-suggestion**
  feature (incl. the `kb_get_suggestions` / `kb_dismiss_suggestion` MCP tools
  and the `llama-cpp-python` dependency), plus their `dsagt init` prompts and
  config keys. Episodic memory keeps the mechanical capture path so a Tier-0
  baseline can be measured first; design notes parked in
  `design-notes/judge.md`.
- The built-in `datacard-generator` skill — it lives in the Genesis catalog and
  is now installed on demand via `dsagt skills add <project> genesis`.
- Dead indexing of installed/built-in skills into the `skills` ChromaDB
  collection (nothing read it after the catalog-only search change).
- Dead `provenance.index_execution_record`.

### Fixed
- **`dsagt init` on an existing project** no longer crashes with a raw
  traceback — it referenced an embedding-choice config key the init menu no
  longer produces.
- **A KB misconfig no longer takes down the server.** An `api` embedding
  backend without its base URL or key now degrades only the KB-backed tools;
  the rest of the surface starts normally (previously every tool was lost).
- **One malformed line no longer stalls a whole pipeline.** A corrupt
  transcript line, trace record, or ack file is skipped rather than aborting
  the session's trace, episodic-memory, and code-use indexing on every
  heartbeat.
- **A code run outside a session indexes correctly** — a `dsagt-run` command
  executed by hand records a null session id that no longer poisons the
  `code_use` batch.
- **Switching away from goose** removes the stale `goose.yaml` MCP config
  instead of leaving it behind.
- **Registry and skill listing/search tolerate incomplete specs** — a code
  spec missing `description`, non-dict frontmatter, and substring tag matches
  no longer crash or mis-filter results.
- **Duplicate `code_use` entries** — indexing is now idempotent against a
  persisted ack set shared by the heartbeat and startup catch-up.
- **CLI-added skill sources** are now persisted to the project config.
- **`dsagt --version`** now works (it was documented but unimplemented —
  argparse errored). Reports the version from `dsagt.__version__`.
- **Catalog skills with technically-invalid YAML frontmatter** (e.g. an
  unquoted `description` containing a colon, like `…readiness levels: Level
  1…`) are no longer silently dropped from discovery. `_parse_frontmatter`
  falls back to a lenient flat parse that recovers `name`/`description`/`tags`.
- **cline:** per-project MCP config via `CLINE_MCP_SETTINGS_PATH`; `cline mcp
  add` works on cline 3.x; global auth and settings are never touched.
- **codex:** the trace reader follows the per-project `CODEX_HOME`.

### Security
- **Skill-install path traversal.** A catalog skill whose frontmatter `name`
  was `..`, absolute, or nested could escape `<project>/skills/` and
  delete/overwrite files elsewhere on `install_skill`; the name must now be a
  single path component.

## [0.1.0] - 2026-01-11

### Added
- Initial release: registry and knowledge MCP servers, BYOA per-agent config
  generation, MLflow/OTel observability, the tool/skill registry, execution
  provenance, and explicit + episodic memory.

[0.2.0]: https://github.com/AI-ModCon/dsagt/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/AI-ModCon/dsagt/releases/tag/0.1.0
