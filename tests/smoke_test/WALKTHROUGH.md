# DSAgt Smoke Test Walkthrough

This walkthrough validates all core DSAgt functionality after installation. You will drive the agent interactively through each step. The whole exercise takes about 10 minutes.

**Prerequisites:**
- DSAgt installed (`uv sync --all-groups`)
- `tests/test_site_config.yaml` configured with valid API keys and endpoints
- An agent platform installed (e.g., `claude` for Claude Code)

## Setup

```bash
cd DSAGT
uv run dsagt init smoke-test --agent claude-code
```

Edit `runtime/smoke-test/dsagt_config.yaml` — set your API keys and embedding endpoint.

```bash
uv run dsagt start smoke-test
```

The agent should launch. You're now inside the agent session.

---

## 1. Knowledge Base — Ingest and Search

**What this tests:** Knowledge server ingestion, embedding API, semantic search.

Tell the agent:

> Ingest the documentation at tests/smoke_test/knowledge/ into the knowledge base.

Wait for the ingest job to complete. Then:

> Search the knowledge base for "how to filter CSV rows by column value"

**Verify:**
- Ingest completes without errors
- Search returns results from the csvtool API reference doc
- Results include relevant text about `csvtool filter`

---

## 2. Tool Registration

**What this tests:** Registry server, tool spec creation, tool file writing.

Tell the agent:

> I have a CSV processing tool called csvtool. Read the documentation at tests/smoke_test/knowledge/api_reference.md and register the `csvtool filter` command as a tool.

**Verify:**
- Agent reads the doc via the registry server's `read_file` tool
- Agent proposes a tool spec with correct parameters (input, column, value, output)
- Agent saves the spec via `save_tool_spec`
- A new tool file appears in the project's `tools/` directory

---

## 3. Tool Execution with Provenance

**What this tests:** dsagt-run wrapper, execution records, trace archive.

Tell the agent:

> Use the scan_directory tool to scan the tests/smoke_test/data/ directory.

**Verify:**
- The agent calls `scan_directory` (bundled tool)
- A JSON execution record appears in `runtime/smoke-test/trace_archive/`
- The record contains the exact command, stdout, timing, and return code

---

## 4. Data Exploration

**What this tests:** Agent working with real data through tools.

Tell the agent:

> Look at the sample data in tests/smoke_test/data/samples.csv. Summarize what's in it — columns, row count, any quality issues.

**Verify:**
- Agent reads or scans the file
- Agent identifies the columns (id, name, status, score, timestamp)
- Agent notes quality issues: missing `status` value in row 4, missing `timestamp` in row 6

---

## 5. Memory — Explicit

**What this tests:** Explicit memory store (user-confirmed facts).

Tell the agent:

> Remember that the samples.csv dataset has null values in the status and timestamp columns.

**Verify:**
- Agent calls `kb_remember` to store the fact
- Confirm with: "What do you remember about the samples dataset?"
- Agent retrieves the stored fact

---

## 6. Exit and Verify Cleanup

Exit the agent (Ctrl+C or type `/exit`).

**Verify in the terminal:**
- Memory extraction runs (or reports "no session exchanges")
- Proxy and MLflow services are stopped
- Messages confirm cleanup completed

---

## 7. Observability — MLflow

**What this tests:** OTel trace export, MLflow UI.

Restart just MLflow to inspect traces:

```bash
cd runtime/smoke-test
python -m mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow/artifacts --port 5001
```

Open http://localhost:5001 in a browser.

**Verify:**
- Traces appear for the LLM calls made during the session
- Each trace shows: model, token counts, latency, request/response content
- Tool use calls are visible in the conversation traces

Stop MLflow when done (Ctrl+C).

---

## 8. Provenance — Execution Records

Inspect the trace archive directly:

```bash
ls runtime/smoke-test/trace_archive/
cat runtime/smoke-test/trace_archive/*.json | python -m json.tool | head -50
```

**Verify:**
- Records exist for tool calls made during the session
- Each record has `record_id`, `tool_name`, `session_id`
- Records from the proxy have `intent` + `report` layers
- Records from dsagt-run (if scan_directory used it) have an `execution` layer with `exact_command`, `stdout`, `return_code`

---

## Cleanup

```bash
rm -rf runtime/smoke-test
```

## Summary

| Step | Capability | Pass Criteria |
|------|-----------|---------------|
| 1 | Knowledge ingest + search | Search returns relevant csvtool docs |
| 2 | Tool registration | Skill file created with correct spec |
| 3 | Tool execution + provenance | Execution record in trace_archive |
| 4 | Data exploration | Agent identifies columns and nulls |
| 5 | Explicit memory | Fact stored and retrieved |
| 6 | Automatic cleanup | Services stopped on agent exit |
| 7 | MLflow observability | Traces visible in MLflow UI |
| 8 | Execution records | Records have expected structure |
