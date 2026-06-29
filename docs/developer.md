# Developer Guide

Material for contributors and users working beyond the default `dsagt init` → agent flow.

## Tests

```bash
uv run python -m pytest -m "not integration"     # unit tests, no creds required
uv run python -m pytest -m integration -v        # integration tests (require real credentials / models)
```

Integration tests need real `EMBEDDING_*` / `LLM_*` credentials (and, for the episodic-memory judge test, the local GGUF model, downloaded on first run). Copy `.env.example` to `.env` and fill in your values where applicable.

For per-flow hand-tests (CLI, VS Code extensions), see the scripts under [`tests/smoke_test/manual_runs/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/manual_runs/).

## Run model

DSAgt is **BYOA (bring your own agent)**: the agent talks to its own LLM provider directly — DSAgt never interposes on that traffic (there is no proxy). Trace capture instead reads the agent's own on-disk session transcript via the MCP server's in-session heartbeat, so no credentials are required and the trace store stays serverless (a SQLite file per project). Agent LLM-call history is recovered post-hoc; nothing intercepts the network.

## Troubleshooting

**Agent command not found.** The agent CLI is not installed or is not on PATH. See the [supported agents table](index.md#supported-agents).

**MCP server not connecting.** Verify the server command resolves:

```bash
uv run which dsagt-server
```

If missing, reinstall: `pip install --force-reinstall "git+https://github.com/AI-ModCon/dsagt.git"`.

**No traces / empty MLflow UI.** The store is a serverless SQLite file — there is no daemon to start. Point the UI at the file directly and confirm the path:

```bash
dsagt info <name>           # shows the resolved tracking URI + a session/trace summary
mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db
```

If the file is missing, the agent hasn't run a session in that project yet (the DB is created lazily on the first span).

**Claude keychain conflict.** If `claude` will not authenticate against a non-default gateway, run `claude /logout` to clear the macOS Keychain OAuth token, then re-export `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` and re-launch.

**Episodic memory judge won't load.** Tier-1 episodic memory needs `llama-cpp-python` (a core dependency, installed from a prebuilt CPU wheel) and downloads a ~1 GB GGUF on first use. If it fails, the heartbeat falls back to Tier-0 (mechanical, no LLM) — episodic memory still works, just without LLM distillation. Re-run `uv sync` if the wheel is missing.
