# Changelog

All notable changes to DSAgt are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A run ended by a signal is recorded.** A run ended by SIGTERM, SIGINT, or
  SIGHUP still writes its record with the signal's status.
- **File hashes in every record.** `execution.file_hashes` holds the SHA-256
  of each input before the run and each output after it. When a spec has no
  parameter roles, an argument that is an existing
  file is an input and one that exists only after the run is an output.
- **`readiness_reports` tool.** The AIDRIN report on record for a data file:
  the report made while the file had its present content, with the text taken
  from that run's record, or how to make one when there is none, and the
  reports from before the file changed.
- **`reconstruct_pipeline(output=...)`** saves the script under the project;
  the bash script creates the recorded output directories first and removes
  a repeated output before the step that rewrites it.
- **One install path for every skill.** `skills.register_skill_scripts`
  registers each `scripts/*.py` and `*.sh` of an installed skill as a code
  (spec from the overrides table or the script's argparse calls) and
  rewrites the skill's bare invocations to the stored `dsagt-run` line;
  `dsagt init`, `install_skill`, and `save_skill` call it and reply with the
  stored lines.
- **`kb_list_collections` and `kb_search(where=...)`.** Every collection is
  listed with its purpose, its metadata keys, and its chunk count, dsagt's
  own (`codes`, `code_use`, `session_memory`, `explicit_memory`) before their
  first write; `kb_search` takes a metadata filter.
- **The MCP env block carries the launching shell's activated environment**
  (`PATH`, `VIRTUAL_ENV`, `CONDA_PREFIX`, `PYTHONPATH`, the library paths,
  the `module` variables, plus `mcp.env_passthrough` from the config), copied
  at `dsagt init` and `dsagt start`; a credential name is refused.
- **Shared tracking server.** `MLFLOW_TRACKING_URI` in the shell redirects all
  self-logging — CLI, MCP server, `dsagt-run` — to a remote MLflow server
  instead of the project's sqlite file; `MLFLOW_TRACKING_API_KEY` authenticates
  to an `X-API-Key` gateway in front of it. `dsagt traces` and `dsagt info`
  follow the same resolution. Keys an agent cannot pass to its MCP children
  (codex, cline) are read from `~/.config/dsagt/env` at startup.
- **Experiment naming for shared servers.** Traces log to an experiment named
  `dsagt-<8 hex>` (hashed from the project directory) rather than the bare
  project name, so two users' `demo` projects do not collide; `mlflow.experiment`
  in `.dsagt/config.yaml` overrides it. The experiment carries the description
  "DSAgt (DataSmith Agent) AI-assisted data pipeline builder — project: <name>"
  and a `dsagt.project` tag, both set once on creation.

### Changed

- **`dsagt-run` takes one option.** `--code` names the registered code and
  the command follows `--`. `--input-files`, `--output-files`, `--session`,
  `--record-id` and `--records-dir` are gone: no agent command ever used
  them, and each value is derived (the project from `DSAGT_PROJECT_DIR` or
  the working directory, the session from `.dsagt/state.yaml`, the record id
  from the run, the files from the spec's roles and the arguments). The trace
  logger is `python -m dsagt.commands.log_trace <record>`, which takes the
  session from the record.
- **`dsagt-run` adds about 0.2 s to a command, down from 1.3 to 2 s.** The
  run loads no trace store. It writes the record, then starts a detached
  process that logs the `code.execute` trace from the record with the run's
  start and end times; writers take a lock on `.dsagt/run_trace.lock`, and
  an error goes to `.dsagt/run_trace.log`.
- **Codes and skills share `skills/`.** A code is a skill directory whose
  frontmatter declares an executable, and a project has no `codes/`
  directory. A project made before this keeps its `codes/` directory and the
  codes in it stay registered under the old path, which no longer resolves:
  move each `codes/<name>/` to `skills/<name>/`, change `codes/` to `skills/`
  in that skill's `SKILL.md`, and call `save_code_spec` again so the stored
  command matches. `dsagt init` does not do this for you.
- **The native skills mirror is a relative symlink per skill directory**, so
  the agent reads the live files; a skill whose description exceeds Claude
  Code's cap is copied with the description truncated.
- **A skill source is fetched as a GitHub tarball** over HTTPS (the shell's
  `GITHUB_TOKEN` when present); the user's git is the fallback for a
  repository the API refuses and for a URL that is not GitHub.
- **Trace acknowledgements are keyed by transcript**, so a resumed
  conversation (`claude --continue`, `codex exec resume`) logs only its new
  turns, and the final flush emits the last turn only once it holds a
  response.
- **Section 1 of the instructions** defines a pipeline step as a command
  that produces or transforms a dataset file, and says it runs as a
  registered code by its stored line. The foreground rule names a background
  subagent as forbidden beside a background task.
- `search_registry` lists hits by rank instead of a rank-fusion score.
- The startup catch-up reuses the server's knowledge base, so one embedder
  serves the session.

- AIDRIN has one path: the code registry. The `aidrin` package is a dependency
  of dsagt, so the CLI is in dsagt's Python environment; every `dsagt init`
  registers `aidrin` as a code, so each call the agent makes through it is an
  execution record; and the `aidrin` base skill is fetched at the release tag
  of the installed package, read when init runs. A cached skill source is
  re-cloned when the branch or tag it holds differs from the one asked for; a
  failed re-clone keeps the previous cache, and a set-aside clone a crashed
  sync left behind is never a source.
- `uv` is a dependency of dsagt: codes with declared Python dependencies run
  through `uv run --with`, so a `pip install` of dsagt is the whole install.
- `dsagt-run` appends the directory of its own interpreter to the command's
  PATH, so a CLI that is a dsagt dependency resolves under pipx or
  `uv tool install`, where only dsagt's own commands are linked onto PATH.
- Per-agent instructions files hold the dsagt text between
  `<!-- dsagt:begin -->` and `<!-- dsagt:end -->`; init and start replace that
  block and keep the user's text around it, so a changed readiness setting or
  an upgraded dsagt reaches the agent.
- The readiness gate is the AI-readiness check, on by default. `dsagt init`
  asks whether to assess tabular data before and after each data transform
  (`--no-readiness` declines); the answer adds one paragraph at the
  per-operation check rule of the instructions, making the check for a tabular
  stage the `aidrin` skill's quality baseline. The profile, the executable
  path, the shared install, the `--readiness` flag, and the appended
  instructions block are gone.
- `datacard-generator` is a base skill: every `dsagt init` installs it from the
  genesis catalog (`skills/basedata-skills/`) alongside `skill-creator` and
  `aidrin`, so the agent invokes it natively without a catalog search.
- Base skills install from the shared source cache after the knowledge base
  is provisioned, one skill at a time: each skill's codes are registered and
  indexed into the `codes` collection before the next skill starts, and a
  skill whose fetch fails is reported without blocking the others. A skill
  already in the project is kept as it is; deleting its directory and
  re-running init restores the upstream copy. A source is cloned once, on the
  first init that needs it, and reused as is by every later init;
  `add_skill_source` with `force` re-clones it on request. An init with a
  warm cache needs no network.

### Removed

- **`scan-directory`** and the bundled-code layer (`src/dsagt/codes/`,
  `CodeRegistry.ensure_bundled_copies`). The base skills' codes are the
  examples.
- **`run_command`, `read_file`, `http_request`.** Execution in the user's
  environment is `dsagt-run`'s, from the agent's shell; reads are the
  agent's own tools.
- **`install_dependencies`, and the install `save_code_spec` ran for a spec
  with dependencies.** Both installed into dsagt-server's own interpreter,
  which is not the environment the agent's shell runs a command in. A code's
  declared dependencies reach its run through the `uv run --with` prefix in
  its stored `executable`. The server has 17 tools.
- **The embedding backend's fallback credential names.** `APIEmbedder` read
  `LLM_API_KEY` and `OPENAI_API_KEY` after `EMBEDDING_API_KEY`, and
  `OPENAI_BASE_URL` after `EMBEDDING_BASE_URL`; the server also read
  `embedding.api_key` from the project config. The key comes from
  `EMBEDDING_API_KEY` and the URL from `EMBEDDING_BASE_URL`, in the shell or
  `~/.config/dsagt/env`. A setup that exported an `OPENAI_`- or `LLM_`-named
  key for the embedder exports `EMBEDDING_API_KEY` instead.
- **`tests/manual_walkthroughs/`.** The two hand-tests described roo, a
  `dsagt mlflow` command, and per-agent OTel, none of which the tree has;
  `dsagt smoke-test` covers the same ground.

### Fixed

- **`search_skills` describes the catalogs it searches.** Its description said
  it spanned installed skills; an installed skill is discovered natively and
  is not indexed, so the search covers the synced external catalogs, and the
  description gives the marker each hit carries.
- **A second `save_skill` keeps the spec the agent saved.** The argparse-derived
  spec replaced one that `save_code_spec` had given dependencies or roles;
  it is now written only when the code does not exist, and `save_code_spec`
  rewrites the owning skill's usage lines to the stored command. A file a
  command moved away is an input, not an output.
- **Records.** With no declared file parameters, an argument the run changed
  is an output (a converter's second run named none). A failed run lists only
  the outputs that exist. The reconstructed pipeline script keeps the
  finished children of a loop script that was killed.
- **Concurrent `dsagt init` runs no longer erase the project registry.** A
  run that read `~/dsagt-projects/projects.yaml` while another was writing it
  found an empty file and saved only its own project. The registry is
  changed under a lock and replaced in one step.
- Tool arguments and results recorded on a trace are bounded before they
  reach the store, with credential-bearing keys (`headers`, `api_key`,
  `token`, …) redacted and common credential shapes inside strings (`Bearer …`,
  `?api_key=…`) masked; an `http_request` bearer token or a whole `read_file`
  payload was previously written verbatim into `mlflow.db`.
- dsagt sets no MLflow active model. An MLflow LoggedModel records the
  version of the application under evaluation, and MLflow attaches the active
  one to everything the thread logs. `init_tracing` and `MLflowSink.write` set
  a LoggedModel for the dsagt release, which is not what is evaluated, so
  metrics logged afterwards in the same thread were attributed to dsagt, and
  failed with `FOREIGN KEY constraint failed` in a store without that model.
  The trace table's *Version* column is empty; the `dsagt.version` metadata on
  every trace records the release.

## [0.2.1] - 2026-09-11

### Changed

- DSAgt holds no skills of its own. `dsagt init` installs two base skills into
  `<project>/skills/` from the repositories that maintain them, re-cloning each
  so the copy matches upstream: `skill-creator` from the genesis catalog
  (`skills/basedata-skills/`) and `aidrin` from `idtlab/AIDRIN`. The `aidrin`
  skill-catalog source is gone (it held that one skill), and so is the built-in
  `aidrin` gate code: the readiness gate instructs the agent to run the
  profile's metrics through the `aidrin` skill's CLI.
- The `genesis` skill source is `github.com/AI-ModCon/genesis-skills`.
- Intel Macs are no longer a supported platform. The `darwin/x86_64` entry in
  `required-environments` held every environment on torch 2.2.2 and NumPy 1.x;
  the lock now resolves torch 2.14, NumPy 2.x, transformers 5.x,
  sentence-transformers 6.x, and tree-sitter-language-pack 1.16. `torch>=2.3`
  is a declared dependency, so an install on an Intel Mac fails at dependency
  resolution with a message naming torch. Without the floor, a fresh install
  there produced torch 2.2.2 beside NumPy 2 and the knowledge-base build
  failed at import.
- `requires-python` is `>=3.12` again; CI tests 3.12 and 3.13.
- Dependencies are declared as ranges with a next-major cap, so dsagt
  resolves beside a project that pins differently. `ruyaml`, which nothing
  imported, is removed.

### Fixed

- The MCP server copied every collection in the shared `kb_index/` into the
  project at startup, so a skill catalog excluded at `dsagt init` still
  appeared as synced. The server now only opens the project's own `kb_index`;
  `dsagt init` is the one place collections are provisioned.
- `run_command` accepted its `command` as one argv element, so a code spec's
  multi-word executable (`dsagt-run --code x -- python ...`) failed with
  "not found". The string is now split like a shell would.

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
  baseline can be measured first.
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

[0.2.1]: https://github.com/AI-ModCon/dsagt/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/AI-ModCon/dsagt/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/AI-ModCon/dsagt/releases/tag/0.1.0
