# CLI Reference

All commands are available after [installation](index.md#installation) and activating your virtual environment.

DSAgt is **BYOA (bring your own agent)**: the agent talks to its own LLM provider; DSAgt never interposes on that traffic. The trace store is **serverless** (a SQLite file per project) — there is no daemon to start or stop.

## Project Management

| Command | Description |
|---------|-------------|
| `dsagt init [<name>]` | Create or reconfigure a project — interactive: agent platform, location, knowledge collections, skill sources, and the episodic-memory opt-in. Provisions the knowledge base (once per machine) and writes the per-agent instructions + MCP config. Re-runnable as a settings editor. |
| `dsagt init <name> --agent <platform> [--location <path>] [--include … \| --exclude …] [--episodic [--domain-tags "a,b"]]` | Same, non-interactively (scripts/CI). `--include`/`--exclude` pick the KB asset set; `--episodic` enables episodic memory (downloads the ~1 GB local judge on first use). |
| `dsagt start <name>` | Launch the agent in the project directory (equivalent to `cd <project> && <agent>`); on exit, runs post-session catch-up. |
| `dsagt list` | List all projects with agent and path. |
| `dsagt info <name> [--json]` | Resolved config (with the source of each value) and a session/trace summary read from the SQLite store. |
| `dsagt mv <name> <new-location>` | Move a project to a new location. |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project and optionally delete its directory. |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode]` | End-to-end install verification. |

Skill catalogs are managed **from the agent** via MCP tools (`add_skill_source` / `list_skill_sources` / `search_skills` / `install_skill`), not the CLI.

## Project Location

The default project location is `~/dsagt-projects/<name>/`. Override with `--location`:

```bash
dsagt init my-project --agent claude --location /data/runs   # /data/runs/my-project/
dsagt init my-project --agent claude --location .            # ./my-project/
```

## Viewing traces

The trace store is a serverless SQLite file — browse it with MLflow's UI pointed at the file (no `dsagt` daemon involved):

```bash
mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db
```

## Server Commands

These are launched automatically by the per-agent MCP config (and `dsagt start`) and are not typically run directly.

| Command | Description |
|---------|-------------|
| `dsagt-server` | The single MCP server — tool registry, knowledge base, memory, and skills. Also runs the in-session heartbeat (trace capture + tool-use/episodic indexing). |
| `dsagt-run` | Provenance-capturing tool execution wrapper; writes execution records to `<project>/trace_archive/`. |
