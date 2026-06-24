# Changelog

All notable changes to DSAgt are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-23

This release consolidates the agent-facing surface. The registry and knowledge
MCP servers were two processes that each loaded their own embedder and opened
their own ChromaDB — pure duplication, plus a write-here/read-there hazard on
the shared skill-catalog collections. They collapse into one `dsagt-server`
(one embedder, one Chroma owner, one connection per agent), so startup is faster
and there are fewer moving parts per project. In parallel, skill discovery now
leans on each agent's *native* SKILL.md auto-discovery, so `search_skills` is
reserved for the one job native discovery can't do — browsing the external
catalog of skills you haven't installed yet.

**Upgrading (forwards compatibility).** There is no automatic migration —
adopting 0.3.0 is rebuild-not-migrate, and no project data changes:
- Re-run `dsagt start <project>` for each existing project; it regenerates the
  per-agent MCP config to point at the single `dsagt-server`.
- For **cline** only, delete `<project>/.cline-data` first — `cline mcp add`
  has no remove, so the stale `dsagt-registry`/`dsagt-knowledge` entries would
  otherwise linger next to the new one.
- Tools, skills, the KB index, traces, and memory all carry over untouched.

### Added
- Source-qualified catalog install: when the same skill name exists in more
  than one synced source, install a specific one with a `<source-slug>/<skill>`
  name (via `install_skill` or `dsagt skills add <project> <slug>/<skill>`)
  instead of dead-ending on the ambiguity guard.

### Changed
- **The two MCP servers are now one `dsagt-server`** — one shared
  `KnowledgeBase`/embedder, one MCP entry per agent, one trace `service.name`.
- Skill discovery is now **catalog-only**: installed and bundled skills are
  discovered natively by every supported agent, so `search_skills` covers only
  the not-yet-installed external catalog. Catalogs are indexed on frontmatter
  (name + description + tags) rather than the full SKILL.md body.
- `search_skills` gains a zero-dependency **keyword fallback** (a token-overlap
  scorer) so it works even when no embedding model is configured.
- Installing a catalog skill now preserves upstream `LICENSE`/`NOTICE`
  provenance and stamps a `PROVENANCE.txt` into the installed skill directory.

### Removed
- **BREAKING:** the `dsagt-registry-server` and `dsagt-knowledge-server` console
  scripts, replaced by `dsagt-server` (see **Upgrading** above).
- The bundled `datacard-generator` skill — it lives in the Genesis catalog and
  is now installed on demand via `dsagt skills add <project> genesis`.
- Dead indexing of installed/bundled skills into the `skills` ChromaDB
  collection (nothing read it after the catalog-only search change).

### Fixed
- `dsagt --version` now works (it was documented but unimplemented — argparse
  errored). Reports the version from `dsagt.__version__`.

## [0.2.0] - 2026-06-23

### Added
- External skill catalogs: discover and install agent skills from GitHub
  sources via `add_skill_source`, `search_skills`, and `install_skill` (plus
  the `dsagt skills sync/add/list/search` CLI), backed by per-source ChromaDB
  collections.
- Native skill discovery: installed and bundled skills are mirrored into the
  agent's native skill directory (e.g. `.claude/skills/`) at init/start.
- `skill-creator` bundled skill for authoring new skills from the Anthropic
  template.
- Install-from-GitHub instructions for non-developers (`pip install
  git+https://github.com/AI-ModCon/dsagt.git` into any Python 3.12/3.13
  environment) in the README and docs.

### Changed
- `search_skills` now reports when no external catalog is synced instead of a
  bare "no match", and `list_skill_sources` flags each known source as
  `synced`/available with its indexed count.
- `install_skill` clarifies that an installed skill is usable in the current
  session immediately — a restart is only needed for hands-free native
  auto-invocation.
- The package version is single-sourced from `dsagt.__version__` (pyproject
  reads it via setuptools dynamic metadata).
- Documentation home page (`docs/index.md`) pulls the supported-agents table
  and install instructions directly from the README via the
  `mkdocs-include-markdown` plugin, so the two no longer drift.

### Fixed
- CLI-added skill sources are now persisted to the project config.

## [0.1.0] - 2026-01-11

### Added
- Initial release: registry and knowledge MCP servers, BYOA per-agent config
  generation, MLflow/OTel observability, the tool/skill registry, execution
  provenance, and explicit + episodic memory.

[Unreleased]: https://github.com/AI-ModCon/dsagt/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/AI-ModCon/dsagt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AI-ModCon/dsagt/releases/tag/v0.2.0
[0.1.0]: https://github.com/AI-ModCon/dsagt/releases/tag/v0.1.0
