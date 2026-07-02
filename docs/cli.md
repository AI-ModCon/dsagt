# CLI Reference

All commands are available after [installation](index.md#installation) and activating your virtual environment.

## Project Management

| Command | Description |
|---------|-------------|
| `dsagt init` | Create or reconfigure a project — interactive menu for name, location, agent platform, knowledge collections, skill sources, and the episodic-memory opt-in. Sets up the knowledge base and writes the per-agent instructions + MCP config. Re-runnable as a settings editor. |
| `dsagt start <name>` | Launch the agent in the project directory (equivalent to `cd <project> && <agent>`); on exit, runs post-session catch-up. |
| `dsagt list` | List all projects with agent and path. |
| `dsagt info <name> [--json]` | Resolved config (with the source of each value) and a session/trace summary read from the SQLite store. |
| `dsagt mv <name> <new-location>` | Move a project to a new location. |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project and optionally delete its directory. |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode\|cline]` | End-to-end install verification. |

### Deprecated `dsagt init` flags (backcompat)

The pre-menu flags still work — the automation/CI path — but are **deprecated** in favor of the interactive menu. Each one skips its corresponding prompt:

| Flag | Prompt it replaces |
|------|--------------------|
| `<name>` (positional) | Project name |
| `--agent <platform>` | Agent platform |
| `--location <path>` | Project location |
| `--include … \| --exclude …` | Knowledge collections / skill sources |
| `--episodic` | "Enable episodic memory?" |

New usage should prefer bare `dsagt init` and the menu.

## Project Location

The default project location is `~/dsagt-projects/<name>/`. 

## Viewing traces

The trace store is a serverless SQLite file — browse it with MLflow's UI pointed at the file:

```bash
mlflow ui --backend-store-uri sqlite:///<project-path>/mlflow.db
```

## Server Commands

These are launched automatically by the per-agent MCP config and are not typically run directly.

| Command | Description |
|---------|-------------|
| `dsagt-server` | The single MCP server — code registry, knowledge base, memory, and skills. Also runs the in-session heartbeat (trace capture + code-use/episodic indexing). |
| `dsagt-run` | Provenance-capturing code execution wrapper; writes execution records to `<project>/trace_archive/`. |
