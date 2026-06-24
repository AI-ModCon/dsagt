# Developer Guide

Material for contributors and users who are working beyond the default `dsagt init` → `dsagt mlflow` → agent flow.

## Tests

```bash
uv run python -m pytest -m "not integration"     # unit tests, no creds required
uv run python -m pytest -m integration -v        # integration tests (require .env)
```

Integration tests read endpoint and key values from `.env` at the repo root. Copy `.env.example` to `.env` and fill in your values.

For per-flow hand-tests (CLI, proxy mode, VS Code extensions), see the scripts under [`tests/smoke_test/manual_runs/`](https://github.com/AI-ModCon/dsagt/tree/main/tests/smoke_test/manual_runs/).

## Proxy Mode

`dsagt init` followed by `dsagt start <project> --enable-proxy` spawns a LiteLLM proxy in front of your agent's LLM calls. This adds:

- Full LLM-call traces (request bodies, tool-use blocks, response payloads) in MLflow for agents whose native OTel does not emit those payloads (codex, opencode).
- Cache-breakpoint injection on outgoing requests (Anthropic prompt caching).
- Sidechannel detection for agent-internal title-generator / session-namer calls.
- Model-name aliasing — useful when an agent CLI hardcodes a model whitelist incompatible with your gateway's served names (cline, roo).

Proxy mode reads upstream LLM credentials from `.env` or the shell. See [`tests/smoke_test/manual_runs/proxy_walkthrough.md`](https://github.com/AI-ModCon/dsagt/blob/main/tests/smoke_test/manual_runs/proxy_walkthrough.md) for the full setup walkthrough.

## Troubleshooting

**Agent command not found.** The agent CLI is not installed or is not on PATH. See the [supported agents table](index.md#supported-agents).

**MCP server not connecting.** Verify uv resolves the server command:

```bash
uv run which dsagt-server
```

If missing, reinstall: `pip install --force-reinstall https://github.com/AI-ModCon/dsagt/archive/refs/tags/0.1.0.zip`.

**MLflow UI empty.** Confirm MLflow is running for the right project:

```bash
dsagt info <name>           # shows the pinned port
curl http://localhost:<mlflow_port>
```

**Claude keychain conflict.** If `claude` will not authenticate against a non-default gateway, run `claude /logout` to clear the macOS Keychain OAuth token, then re-export `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` and re-launch.
