# DSAgt — Developer Notes

Material that's useful once you start poking at internals or running beyond the default `dsagt init` → agent flow.

## Tests

```bash
uv run python -m pytest -m "not integration"     # unit tests, no creds required
uv run python -m pytest -m integration -v        # integration tests (require real credentials / models)
```

Integration tests need real `EMBEDDING_*` / `LLM_*` credentials (and, for the episodic-memory judge test, the local GGUF model, downloaded on first run). Copy `.env.example` to `.env` and fill in your values where applicable.

For per-flow hand-tests (CLI, VS Code extensions), see the scripts under [`tests/smoke_test/manual_runs/`](tests/smoke_test/manual_runs/).

## Run model

DSAgt is **BYOA (bring your own agent)**: the agent talks to its own LLM provider directly — DSAgt never interposes on that traffic (there is no proxy). Trace capture instead reads the agent's own on-disk session transcript via the MCP server's in-session heartbeat, so no credentials are required and the trace store stays serverless (a SQLite file per project, no daemon).

## Troubleshooting

**Agent command not found.** The agent CLI isn't installed or isn't on PATH — see the install table in the [README](README.md#installation).

**MCP server not connecting.** DSAgt exposes a single server (`dsagt-server`; the earlier `dsagt-registry-server` / `dsagt-knowledge-server` pair was merged in 0.2.0). Verify it resolves:

```bash
uv run which dsagt-server
```

If missing, reinstall: `uv sync --reinstall`. On an existing project created against the old two-server layout, re-run `dsagt start <project>` to regenerate its MCP config (for cline, delete `<project>/.cline-data` first).

**No traces / empty MLflow UI.** The store is a serverless SQLite file — there is no daemon. Point the UI at the file and confirm the path:

```bash
dsagt info <name>           # resolved tracking URI + a session/trace summary
mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db
```

If the file is missing, no session has run in that project yet (the DB is created lazily on the first span).

**Claude keychain conflict.** If `claude` won't authenticate against a non-default gateway, run `claude /logout` to clear the macOS Keychain OAuth, then re-export `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` and re-launch.
