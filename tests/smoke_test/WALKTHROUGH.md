# DSAgt Smoke Test Walkthrough

This walkthrough validates all core DSAgt functionality after installation. You will drive the agent interactively through each step. The whole exercise takes about 10 minutes.

**Prerequisites:**
- DSAgt installed (`uv sync --all-groups`)
- `.env` at the DSAGT repo root with `LLM_*` and `EMBEDDING_*` values (copy from `.env.example`)
- An agent platform installed (e.g., `claude` for Claude Code)

## Setup

```bash
cd DSAGT
source .venv/bin/activate
dsagt init smoke-test --agent claude
dsagt mv smoke-test ./
```

`dsagt init` scaffolds under `~/dsagt-projects/` by default; moving the
project into `DSAGT/` is what makes the `../tests/smoke_test/...` paths
below resolve from the agent's working directory.

Edit `smoke-test/dsagt_config.yaml` — set your API keys and embedding endpoint.

```bash
dsagt start smoke-test
```

The agent should launch. You're now inside the agent session.

---

## 1. Knowledge Base — Ingest and Search

**What this tests:** Knowledge server ingestion, embedding API, semantic search.

Tell the agent:

> Ingest the documentation at ../tests/smoke_test/knowledge/ into the knowledge base.

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

> I have a CSV processing tool called csvtool. Read the documentation at ../tests/smoke_test/knowledge/api_reference.md and register the `csvtool filter` command as a tool.

**Verify:**
- Agent reads the doc via the registry server's `read_file` tool
- Agent proposes a tool spec with correct parameters (input, column, value, output)
- Agent saves the spec via `save_tool_spec`
- A new tool file appears in the project's `tools/` directory

---

## 3. Tool Execution with Provenance

**What this tests:** dsagt-run wrapper, execution records, trace archive.

Tell the agent:

> Use the scan_directory tool from the registry to scan the ../tests/smoke_test/data/ directory.

**Verify:**
- The agent calls `scan_directory` (bundled tool)
- A JSON execution record appears in `smoke-test/trace_archive/`
- The record contains the exact command, stdout, timing, and return code

---

## 4. Data Exploration

**What this tests:** Agent working with real data through tools.

Tell the agent:

> Look at the sample data in ../tests/smoke_test/data/samples.csv. Summarize what's in it — columns, row count, any quality issues.

**Verify:**
- Agent reads or scans the file
- Agent identifies the columns (id, name, status, score, timestamp)
- Agent notes quality issues: missing `status` value in row 4, missing `timestamp` in row 6

---

## 5. Memory — Explicit (save)

**What this tests:** Explicit memory write to the user-confirmed fact store.

Tell the agent:

> Remember that the samples.csv dataset has null values in the status and timestamp columns.

**Verify:**
- Agent calls `kb_remember` to store the fact
- Agent confirms the fact was saved

---

## 6. Exit and Verify Cleanup

Exit the agent (Ctrl+C or type `/exit`).

**Verify in the terminal:**
- Memory extraction runs (or reports "no session exchanges")
- Proxy and MLflow services are stopped
- Messages confirm cleanup completed

---

## 7. Memory — Explicit (recall across sessions)

**What this tests:** Explicit memory persists across session restarts.

Restart the agent:

```bash
dsagt start smoke-test
```

Tell the agent:

> What do you remember about the samples dataset?

**Verify:**
- Agent retrieves the fact stored in step 5 (null values in `status` and `timestamp`)
- The fact is sourced from explicit memory, not reconstructed from the current session's context

Exit the agent again before continuing (so MLflow's port is free for the next step).

---

## 8. Observability — MLflow

**What this tests:** MLflow LiteLLM autologging, trace UI.

Restart just MLflow to inspect traces:

```bash
dsagt mlflow smoke-test
```

Open http://localhost:5001 in a browser.

**Verify:**
- Traces appear for the LLM calls made during the session
- Each trace shows: model, token counts, latency, request/response content
- Tool use calls are visible in the conversation traces

Stop MLflow when done (Ctrl+C).

---

## 9. Provenance — Execution Records

Inspect the trace archive directly:

```bash
ls smoke-test/trace_archive/
cat smoke-test/trace_archive/*.json | python -m json.tool | head -50
```

**Verify:**
- Records exist for tool calls made during the session
- Each record has `record_id`, `tool_name`, `session_id`
- Records from the proxy have `intent` + `report` layers
- Records from dsagt-run (if scan_directory used it) have an `execution` layer with `exact_command`, `stdout`, `return_code`

---

## Cleanup

```bash
rm -rf smoke-test
```

## Summary

| Step | Capability | Pass Criteria |
|------|-----------|---------------|
| 1 | Knowledge ingest + search | Search returns relevant csvtool docs |
| 2 | Tool registration | Skill file created with correct spec |
| 3 | Tool execution + provenance | Execution record in trace_archive |
| 4 | Data exploration | Agent identifies columns and nulls |
| 5 | Explicit memory — save | Fact stored via `kb_remember` |
| 6 | Automatic cleanup | Services stopped on agent exit |
| 7 | Explicit memory — cross-session recall | Fact retrieved after restart |
| 8 | MLflow observability | Traces visible in MLflow UI |
| 9 | Execution records | Records have expected structure |
