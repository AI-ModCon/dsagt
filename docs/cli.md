# CLI Reference

All commands are available after [installation](index.md#installation) and activating your virtual environment.

## Project Management

| Command | Description |
|---------|-------------|
| `dsagt init <name> --agent <platform> [--location <path>] [--mlflow-port N]` | Create a project; write per-agent MCP config; print the launch one-liner |
| `dsagt list` | List all projects with agent, status, and path |
| `dsagt info <name> [--json]` | Resolved config (with source per value) and a session/error summary |
| `dsagt mv <name> <new-location>` | Move a project to a new location |
| `dsagt rm <name> [-y] [--keep-files]` | Unregister a project and optionally delete its directory |

## Session Lifecycle

| Command | Description |
|---------|-------------|
| `dsagt mlflow <name>` | Start MLflow for a project and print OTel routing exports |
| `dsagt stop <name>` | Stop the MLflow daemon |
| `dsagt memory --project <name>` | Distill new traces from MLflow into episodic memory |

## Setup

| Command | Description |
|---------|-------------|
| `dsagt setup-kb [--collection <name>]` | Build the shared core knowledge base collections |
| `dsagt smoke-test [--agent claude\|goose\|codex\|opencode]` | End-to-end install verification |

## Project Location

The default project location is `~/dsagt-projects/<name>/`. Override with `--location`:

```bash
dsagt init my-project --agent claude --location /data/runs   # /data/runs/my-project/
dsagt init my-project --agent claude --location .            # ./my-project/
```

## Server Commands

These are launched automatically by `dsagt init` via the per-agent MCP config and are not typically run directly.

| Command | Description |
|---------|-------------|
| `dsagt-server` | MCP server — tool registry + knowledge base |
| `dsagt-run` | Provenance-capturing tool execution wrapper |
| `dsagt-proxy` | LiteLLM proxy server (proxy mode only) |
| `dsagt-setup-kb` | Core knowledge base setup (called by `dsagt setup-kb`) |
