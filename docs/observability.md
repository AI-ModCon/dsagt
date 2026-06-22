# Observability

DSAgt provides end-to-end trace visibility through a local MLflow instance. All internal layers emit OTLP HTTP spans to MLflow's `/v1/traces` endpoint.

## Starting MLflow

```bash
dsagt mlflow <project-name>
```

Prints the MLflow UI URL and the `export` block for routing agent OTel output. The port is pinned at `dsagt init` time and listed by `dsagt info <name>`.

## Trace Coverage

| Source | Span type | Contents |
|--------|-----------|----------|
| Knowledge base | `kb.search`, `kb.embed`, `kb.index_search`, `kb.rerank` | Per-phase timing trees |
| Tool executions | `tool.execute` | Exit code, duration, file counts, truncated stderr. Full payload in `trace_archive/<record_id>.json` |
| Registry events | `save_tool_spec`, `install_dependencies`, `reconstruct_pipeline` | Span metadata |
| Native agent OTel | LLM call spans | Coverage varies by agent (see below) |

### Agent OTel Coverage

Export the variables printed by `dsagt mlflow` before launching your agent:

| Agent | Coverage |
|-------|----------|
| claude | Full request/response payloads |
| goose | Full request/response payloads |
| codex | Token counts and tool names |
| opencode | None natively |

Every span carries the project's `session.id` for filtering in the MLflow trace view.

## Provenance and Reconstruction

Tool execution records on disk (`trace_archive/<record_id>.json`) provide the canonical provenance chain. The agent calls `reconstruct_pipeline` to render the archive as a reproducible bash script or Snakemake workflow.

## Stopping MLflow

```bash
dsagt stop <project-name>
```

Releases the port and stops the gunicorn workers. The PID is stored in `<project>/.runtime`.
