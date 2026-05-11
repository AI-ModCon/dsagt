# DSAgt — Developer Notes

Material that's useful once you start poking at internals or running modes beyond the default `dsagt init` → `dsagt mlflow` → agent flow.

## Tests

```bash
uv run python -m pytest -m "not integration"     # unit tests, no creds required
uv run python -m pytest -m integration -v        # integration tests (require .env)
```

Integration tests read endpoint and key values from `.env` at the repo root. Copy `.env.example` to `.env` and fill in your values.

For per-flow hand-tests (CLI, proxy mode, VS Code extensions), see the scripts under [`tests/smoke_test/manual_runs/`](tests/smoke_test/manual_runs/).

## Advanced: Proxy mode

`dsagt init` followed by `dsagt start <project> --enable-proxy` spawns a LiteLLM proxy in front of your agent's LLM calls. This adds:

- Full LLM-call traces (request bodies, tool-use blocks, response payloads) in MLflow for agents whose native OTel doesn't emit those payloads (codex, opencode) or that don't emit OTel at all.
- Cache-breakpoint injection on outgoing requests (Anthropic prompt caching).
- Sidechannel detection for agent-internal title-generator / session-namer calls.
- Model-name aliasing — useful when an agent CLI hardcodes a model whitelist incompatible with your gateway's served names (cline, roo).

Proxy mode reads upstream LLM credentials from `.env` (or the shell). See the prerequisites and the full setup walkthrough at [`tests/smoke_test/manual_runs/proxy_walkthrough.md`](tests/smoke_test/manual_runs/proxy_walkthrough.md).

## Troubleshooting

**Agent command not found.** The agent CLI isn't installed or isn't on PATH — see the install table in the [README](README.md#installation).

**MCP servers not connecting.** Verify uv resolves the server commands:

```bash
uv run which dsagt-registry-server
uv run which dsagt-knowledge-server
```

If missing, reinstall: `uv sync --reinstall`.

**MLflow UI empty.** Confirm MLflow is running for the right project:

```bash
dsagt info <name>           # shows the pinned port
curl http://localhost:<mlflow_port>
```

**Claude keychain conflict.** If `claude` won't authenticate against a non-default gateway, run `claude /logout` to clear the macOS Keychain OAuth, then re-export `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` and re-launch.
