# Proxy mode hand-test

Tests goose, claude, roo, cline, codex, opencode through `dsagt start --enable-proxy`.

The proxy intercepts the agent's LLM calls, translates lab-gateway-aliased model names back to upstream-served names (`_AGENT_PRIMARY_ALIASES`), forwards via LiteLLM to the configured upstream, and emits OTLP spans to MLflow tagged with `service.name=dsagt-proxy`. Plus cache-breakpoint injection on outgoing requests + sidechannel detection on responses.

Replace `<AGENT>` below with one of: `goose`, `claude`, `roo`, `cline`, `codex`, `opencode`.

## Setup (once per agent)

```bash
cd ~/dsagt
source .venv/bin/activate
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# Proxy mode reads upstream creds from .env (or shell env).  These four
# are required:
export LLM_PROVIDER=openai
export LLM_MODEL=claude-haiku-4-5-20251001-v1-project
export LLM_BASE_URL=https://ai-incubator-api.pnnl.gov
export LLM_API_KEY=sk-...
# If `embedding.backend: api` in your project config, also set
# EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_BASE_URL /
# EMBEDDING_API_KEY.  Default `local` mode needs none of these.

dsagt rm smoke-<AGENT> -y >/dev/null 2>&1
dsagt init smoke-<AGENT> --agent <AGENT>
```

## Run

```bash
# Terminal 1
dsagt mlflow smoke-<AGENT>

# Terminal 2
dsagt start smoke-<AGENT> --enable-proxy
```

This spawns:
- MLflow on the pinned port (from `dsagt_config.yaml`)
- `dsagt-proxy` on a free port (its YAML config is regenerated each run)
- The agent, with its env / config files routed at the proxy URL + a sentinel API key (real upstream creds live only in the proxy subprocess)

## Scripted prompts (paste one at a time)

Same six prompts as `cli_walkthrough.md`:

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.
2. > I have a CSV utility called `csvtool`. Its reference is at `$SMOKE_DIR/knowledge/api_reference.md` — register the `filter` subcommand. Use an underscore in the name.
3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`.
4. > Look at `$SMOKE_DIR/data/samples.csv` and summarize — columns, row count, quality issues.
5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.
6. > What do you remember about the samples dataset?

Exit the agent.

## Verify

```bash
dsagt info smoke-<AGENT>
ls ~/dsagt-projects/smoke-<AGENT>/tools/
ls ~/dsagt-projects/smoke-<AGENT>/trace_archive/
test -s ~/dsagt-projects/smoke-<AGENT>/explicit_memories.yaml && echo OK
```

In MLflow UI:

- **All six agents:** agent LLM-call traces with `service.name = "dsagt-proxy"`, full request `messages` + response payloads (LiteLLM autolog shape), token counts, latency.
- Plus the usual MCP / dsagt-run spans (separate `service.name`s).

If the agent emits its own OTel too (claude/goose), you'll see those spans alongside the dsagt-proxy ones — same trace store, different service names.

## Sidechannel warnings

At end-of-session, if any agent-internal "title generator" / "session namer" calls fired (model names not in your `LLM_MODEL`), the teardown prints:

```
  ⚠ Sidechannel model calls intercepted:
      gpt-4o-mini  (2 calls)
    Two possible causes:
      (1) agent sidechannel — safe to ignore
      (2) typo in dsagt_config.yaml llm.model — these replies are canned, not real
```

Confirms the wildcard catchall + canned-response mock fired correctly.

## Per-agent expected behavior

| Agent | What proxy mode adds beyond BYOA |
|---|---|
| goose | Uniform trace shape; MLflow now has every LLM call from one parser-friendly source |
| claude | Sidesteps the macOS Keychain OAuth conflict — claude's auth token gets ignored by the proxy, which uses `LLM_API_KEY` for upstream |
| roo | **Un-punted.** Roo's anthropic SDK posts to `ANTHROPIC_BASE_URL` (the proxy URL); the proxy aliases roo's hardcoded `claude-sonnet-4-5` rewrite back to your upstream model |
| cline | **Un-punted.** `cline auth -p openai -b <proxy>` configures cline to talk openai-shape to the proxy; the proxy translates and aliases |
| codex | LLM-call message payloads now visible (codex's bundled OTel only emits token counts) |
| opencode | LLM-call traces visible (opencode emits no native OTel at all) |

## Verify proxy logs

```bash
tail -50 ~/dsagt-projects/smoke-<AGENT>/proxy.log
```

Look for:
- `init_proxy_tracing: service=dsagt-proxy mlflow=...`
- `Generated LiteLLM config at /tmp/dsagt_litellm_*.yaml`
- `Uvicorn running on http://0.0.0.0:<proxy_port>`
- One `litellm` log line per agent LLM call
