# Changelog

All notable changes to DSAgt are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/AI-ModCon/dsagt/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AI-ModCon/dsagt/releases/tag/v0.2.0
[0.1.0]: https://github.com/AI-ModCon/dsagt/releases/tag/v0.1.0
