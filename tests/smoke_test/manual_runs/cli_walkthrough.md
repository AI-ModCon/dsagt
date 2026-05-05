# BYOA CLI hand-test

Tests goose, codex, opencode, and claude in the BYOA-CLI flow (no proxy, no IDE extension).

Replace `<AGENT>` below with one of: `goose`, `codex`, `opencode`, `claude`.

## Setup (once per agent)

```bash
cd ~/dsagt
source .venv/bin/activate
export SMOKE_DIR="$(pwd)/tests/smoke_test"

# Per-agent provider creds in shell — see `dsagt init` printout for the
# exact names.  Examples below cover the PNNL openai-shape gateway:
export OPENAI_API_KEY=sk-...                # goose, opencode
export OPENAI_BASE_URL=https://ai-incubator-api.pnnl.gov   # opencode
export OPENAI_HOST=https://ai-incubator-api.pnnl.gov       # goose
export GOOSE_PROVIDER=openai                # goose
export GOOSE_MODEL=claude-haiku-4-5-20251001-v1-project    # goose
export OPENCODE_MODEL=openai/claude-haiku-4-5-20251001-v1-project  # opencode
# claude: subscription auth from ~/.claude (or `claude /login`); no env needed
# codex: subscription auth from ~/.codex (or `codex login`); no env needed

dsagt rm smoke-<AGENT> -y >/dev/null 2>&1
dsagt init smoke-<AGENT> --agent <AGENT>
```

## Run

```bash
# Terminal 1
dsagt mlflow smoke-<AGENT>

# Terminal 2 (copy the launch line printed by `dsagt init`)
cd ~/dsagt-projects/smoke-<AGENT> && <agent-launch-command>
```

## Scripted prompts (paste one at a time, wait for each to complete)

1. > Ingest the docs in `$SMOKE_DIR/knowledge/` into a collection named `knowledge`.

   **Expect:** `kb_ingest` MCP call; ingest job completes; ~10 chunks indexed.

2. > I have a CSV utility called `csvtool`. Its reference is at `$SMOKE_DIR/knowledge/api_reference.md` — register the `filter` subcommand. Use an underscore in the name.

   **Expect:** `save_tool_spec` MCP call; `~/dsagt-projects/smoke-<AGENT>/tools/csvtool_filter.md` exists.

3. > Use the `scan_directory` tool from the registry to scan `$SMOKE_DIR/data/`.

   **Expect:** Agent invokes `dsagt-run --tool scan_directory ...`; one record in `~/dsagt-projects/smoke-<AGENT>/trace_archive/scan_directory_*.json`.

4. > Look at `$SMOKE_DIR/data/samples.csv` and summarize — columns, row count, quality issues.

   **Expect:** 5 columns (id, name, status, score, timestamp), 8 rows, nulls in status row 4 and timestamp row 6.

5. > Put this in explicit memory: samples.csv has null values in the status and timestamp columns.

   **Expect:** `kb_remember` MCP call; `~/dsagt-projects/smoke-<AGENT>/explicit_memories.yaml` non-empty.

6. > What do you remember about the samples dataset?

   **Expect:** Agent recalls the null-values fact via `kb_get_memories`.

Exit the agent (Ctrl+C / `/exit`).

## Verify

```bash
dsagt info smoke-<AGENT>
ls ~/dsagt-projects/smoke-<AGENT>/tools/
ls ~/dsagt-projects/smoke-<AGENT>/trace_archive/
test -s ~/dsagt-projects/smoke-<AGENT>/explicit_memories.yaml && echo OK
```

In Terminal 1's MLflow UI (http://localhost:5001 by default; check the port `dsagt mlflow` printed):

- **goose / claude:** agent LLM-call traces with full message + tool_use payloads.
- **codex:** agent LLM-call spans with token counts + tool names; `codex.tool_result` log events with full args+output; `codex.user_prompt` log events (un-redacted via the `[otel]` block we wrote into `config.toml`). Message bodies are not in spans — codex's bundled OTel doesn't emit them.
- **opencode:** no agent LLM-call traces (opencode emits no native OTel). `kb.*` / `registry.*` / `tool.execute` spans only.
- All four: `kb.search` / `kb.embed` / `tool.execute` spans from MCP servers + dsagt-run.

## Notes

- **claude keychain conflict:** if claude won't auth against your gateway with `ANTHROPIC_BASE_URL` set, run `claude /logout` first to clear the macOS Keychain OAuth that takes precedence over env vars.
- **opencode transient streaming bug:** `Error: "text part chatcmpl-... not found"` happens occasionally on first run; just re-run.
- **embedding backend:** defaults to `local` (sentence-transformers, ~130 MB BAAI/bge-small-en-v1.5 downloaded once, no API call). Switch via `embedding.backend: api` in `dsagt_config.yaml` only if you need a hosted embedder.
