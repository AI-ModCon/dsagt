# Observability

DSAgt logs traces to a **serverless MLflow store** — a SQLite file at `<project>/mlflow.db`, created lazily on the first span. View it with MLflow's UI pointed at the file:

```bash
mlflow ui --backend-store-uri sqlite:///<project>/mlflow.db
```

`dsagt info <name>` prints the resolved tracking URI and a session/trace summary. The tracking URI resolves as `MLFLOW_TRACKING_URI` env → project config → the `sqlite:///<project>/mlflow.db` default.

## Two feeds

DSAgt is **BYOA**: the agent talks to its own LLM provider directly, and DSAgt reconstructs traces from what the agent writes to disk. Traces come from two places:

1. **First-party spans (live).** DSAgt instruments its own code and emits spans directly to the store as it runs.
2. **Agent traces (post-hoc).** The MCP server's in-session heartbeat reads the agent's own on-disk session transcript, translates it to a canonical trace shape, and writes it to the same store via the MLflow sink — recovering prompts, responses, and tool calls.

## Trace Coverage

| Source | Span type | Contents |
|--------|-----------|----------|
| Knowledge base | `kb.search`, `kb.embed`, `kb.index_search`, `kb.rerank` | Per-phase timing trees |
| Code executions | `code.execute` | Exit code, duration, file counts, truncated stderr. Full payload in `trace_archive/<record_id>.json` |
| Registry events | `save_code_spec`, `install_dependencies`, `reconstruct_pipeline` | Span metadata |
| Agent traces | one AGENT subtree per turn (`llm` / `tool_<name>` children) | Prompts, responses, tool calls, and token usage where the transcript carries them |

### Agent trace coverage

Agent traces are reconstructed from each agent's on-disk session record. A per-agent reader + translator runs for every supported agent (claude, codex, goose, opencode, cline), uniformly. Fidelity is capped by what the transcript persisted (e.g. token counts and timing appear where the agent recorded them).

Every span carries the project's session id (minted per launch into `<project>/.dsagt/state.yaml`) for filtering in the MLflow trace view.

## The heartbeat

The trace scan runs as a periodic heartbeat inside the long-lived MCP server — the one DSAgt process alive in every launch flow. Each tick reads new transcript records, translates completed turns, and fans out to subscribers (the MLflow sink always; the episodic-memory extractor when enabled). Correctness rests on idempotency: each subscriber keeps its own ack set, so a re-tick or a next-session catch-up can never double-log or lose a turn. The same heartbeat incrementally indexes `dsagt-run` code-execution records into the `code_use` collection.

## Provenance and Reconstruction

Code execution records on disk (`trace_archive/<record_id>.json`) provide the canonical provenance chain. The agent calls `reconstruct_pipeline` to render the archive as a reproducible bash script or Snakemake workflow (which also flushes the latest code-use records into the searchable index first).
