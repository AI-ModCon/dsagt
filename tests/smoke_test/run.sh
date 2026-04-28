#!/usr/bin/env bash
# DSAGT smoke test — non-interactive end-to-end exercise.
#
# Drives the SAME `dsagt start` lifecycle as an interactive run (config
# generation → start_services → agent → run_extraction → stop_services).
# Only the agent-launch step swaps from `goose session` to `goose run -i`.
# Sources DSAGT/.env so the env-var references in dsagt_config.yaml resolve.
#
# Run from anywhere:
#   bash tests/smoke_test/run.sh
#   dsagt smoke-test
#
# Exit code 0 on success, non-zero on failed assertion or agent timeout.

set -uo pipefail

AGENT="${DSAGT_SMOKE_AGENT:-${1:-goose}}"   # arg or env var, default goose
# Per-agent project name so each agent's mlflow.db, trace_archive, and
# kb_index/ survive across runs — crucial for cross-agent comparison
# (e.g., why does claude-code use 10x the tokens roo does?).  Without this,
# `dsagt rm` at the start of each run wipes the previous agent's state.
PROJECT="smoke-test-${AGENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSAGT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_FILE="${SCRIPT_DIR}/script.txt"

case "${AGENT}" in
    goose|claude-code|cline|roo|codex) ;;
    *)
        echo "ERROR: agent must be one of: goose, claude-code, cline, roo, codex (got '${AGENT}')" >&2
        exit 2
        ;;
esac
echo "[smoke] Agent: ${AGENT}"

cd "${DSAGT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Load .env
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: ${DSAGT_ROOT}/.env not found." >&2
    echo "  Copy .env.example to .env and fill in your values." >&2
    exit 2
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

for var in LLM_PROVIDER LLM_API_KEY LLM_BASE_URL LLM_MODEL EMBEDDING_API_KEY EMBEDDING_BASE_URL EMBEDDING_MODEL; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} is empty in .env" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# 2. Clean slate (idempotent — silent if nothing exists)
# ---------------------------------------------------------------------------
dsagt stop "${PROJECT}" >/dev/null 2>&1 || true
dsagt rm "${PROJECT}" -y >/dev/null 2>&1 || true
rm -rf "${DSAGT_ROOT}/${PROJECT}"

# ---------------------------------------------------------------------------
# 3. Init + move into DSAGT/ so '../tests/smoke_test/...' paths resolve
#    from the agent's cwd.
# ---------------------------------------------------------------------------
dsagt init "${PROJECT}" --agent "${AGENT}"
dsagt mv "${PROJECT}" "${DSAGT_ROOT}"
PDIR="${DSAGT_ROOT}/${PROJECT}"

# ---------------------------------------------------------------------------
# 4. Run the FULL `dsagt start` lifecycle, with the agent in batch mode.
#    Wall-clock cap belt-and-suspenders the --max-turns inside.
#
#    Pure-bash watcher pattern instead of GNU `timeout` so the smoke test
#    works on stock macOS without `brew install coreutils`.  SIGTERM gives
#    dsagt's finally-block a chance to stop services cleanly; the
#    follow-up SIGKILL after WALL_CLOCK_GRACE catches the agent if it
#    swallows the term signal.
# ---------------------------------------------------------------------------
WALL_CLOCK_CAP=300   # seconds (5 minutes)
WALL_CLOCK_GRACE=10  # extra seconds before SIGKILL

echo
echo "[smoke] Running dsagt start --script (${WALL_CLOCK_CAP}s wall-clock cap)…"
dsagt start "${PROJECT}" --script "${SCRIPT_FILE}" --max-turns 30 &
DSAGT_PID=$!
(
    sleep "${WALL_CLOCK_CAP}"
    kill -TERM "${DSAGT_PID}" 2>/dev/null && \
        echo "[smoke] WARN: ${WALL_CLOCK_CAP}s cap exceeded — sent SIGTERM to dsagt start (pid ${DSAGT_PID})"
    sleep "${WALL_CLOCK_GRACE}"
    kill -KILL "${DSAGT_PID}" 2>/dev/null && \
        echo "[smoke] WARN: dsagt start did not exit on SIGTERM — sent SIGKILL"
) &
WATCHER_PID=$!
wait "${DSAGT_PID}"
START_EXIT=$?
# Tear down the watcher if dsagt exited on its own.
kill -TERM "${WATCHER_PID}" 2>/dev/null
wait "${WATCHER_PID}" 2>/dev/null

if [[ ${START_EXIT} -ne 0 ]]; then
    echo "WARN: dsagt start exited non-zero (${START_EXIT}) — continuing to artifact checks anyway"
fi

# Defensive: ensure no stray services if the lifecycle's finally didn't run
# (timeout SIGKILL skips Python's finally blocks).  Output kept visible —
# if any port is still in use after this, we want to see the warning so
# the next run doesn't race a half-shutdown orphan.
echo
echo "[smoke] Final cleanup…"
dsagt stop "${PROJECT}" || true

# ---------------------------------------------------------------------------
# 5. Artifact checks
# ---------------------------------------------------------------------------
echo
echo "[smoke] Verifying artifacts…"
FAIL=0
check() {
    local label="$1" cmd="$2"
    if eval "${cmd}" >/dev/null 2>&1; then
        echo "  PASS  ${label}"
    else
        echo "  FAIL  ${label}  (cmd: ${cmd})"
        FAIL=1
    fi
}

check "csvtool_filter spec written"  "test -f '${PDIR}/tools/csvtool_filter.md'"
check "trace_archive has records"    "ls '${PDIR}/trace_archive/'*.json | grep -q ."
check "scan_directory record"        "ls '${PDIR}/trace_archive/'*scan_directory*.json | grep -q ."
# Both files are written by dsagt-knowledge-server's kb_ingest_directory MCP
# tool — chroma.sqlite3 is the actual vector DB, route.json is the collection
# manifest.  Checking only `test -d kb_index/knowledge` is too weak: an agent
# can satisfy it by hand-crafting an empty directory tree, masking a broken
# MCP wiring (which is exactly what we hit when cline's dsagt-knowledge
# server crashed silently and the LLM compensated by mkdir-ing the path).
check "knowledge ingested (route)"   "test -f '${PDIR}/kb_index/knowledge/route.json'"
check "knowledge ingested (vectors)" "test -f '${PDIR}/kb_index/knowledge/chroma.sqlite3'"
# Explicit memory writes to <project>/explicit_memories.yaml (YAML at the
# project root), NOT to kb_index/.  Only kb_remember (called deliberately
# by the agent in response to "Put this in explicit memory" / "remember
# this") populates the file.  End-of-session episodic extraction writes
# elsewhere (kb_index/episodic_memory/...) and is independent.  Checking
# the YAML's existence + non-empty catches the hallucination case where
# the agent claims it stored a fact but didn't actually call the tool.
check "explicit memory recorded"     "test -s '${PDIR}/explicit_memories.yaml'"
check "mlflow has traces"            "test -s '${PDIR}/mlflow/mlflow.db'"

# ---------------------------------------------------------------------------
# 6. LLM dispatch parity: every chat/completions request that hit the proxy
#    must produce exactly one litellm-* trace in MLflow.  A delta means
#    LiteLLM's async callback dispatch dropped one — we'd never notice
#    without this check (no other artifact distinguishes "LLM call dropped"
#    from "agent didn't make that call").
# ---------------------------------------------------------------------------
# grep -c exits 1 when it finds zero matches, so pairing with `|| echo 0`
# concatenates grep's "0" with echo's "0" into "0\n0", which [[ -gt ]] then
# can't parse.  Use a pipe to wc -l instead so the final exit code is always 0
# and the output is always a single integer.
#
# Match both endpoints because different agents send different formats:
#   - goose, cline / OpenAI-format clients → /chat/completions
#   - claude-code, roo / Anthropic-format clients → /v1/messages
if [[ -f "${PDIR}/proxy.log" ]]; then
    PROXY_REQUESTS=$(grep -cE 'POST .*(/(v1/)?chat/completions|/v1/messages)' "${PDIR}/proxy.log" 2>/dev/null | head -1)
    PROXY_REQUESTS="${PROXY_REQUESTS:-0}"
else
    PROXY_REQUESTS=0
fi
LLM_TRACES=$(python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///${PDIR}/mlflow/mlflow.db')
exp = mlflow.get_experiment_by_name('${PROJECT}')
if exp is None:
    print(0); raise SystemExit
traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], max_results=500)
n = sum(1 for _, row in traces.iterrows()
        if any(s['name'].startswith('litellm-') for s in row['spans']))
print(n)
" 2>/dev/null)
LLM_TRACES="${LLM_TRACES:-0}"

# Sidechannel wildcard hits on Anthropic /v1/messages fire DSAGT's callback
# but LiteLLM skips MlflowLogger for mock_response on that path, leaving an
# orphan request with no trace.  Known quirk; not a real dispatch bug.
# Count sidechannel entries for this session and tolerate that many drops.
if [[ -f "${PDIR}/sidechannel.jsonl" ]]; then
    SIDECHANNEL_DROPS=$(wc -l < "${PDIR}/sidechannel.jsonl" 2>/dev/null | tr -d ' ')
    SIDECHANNEL_DROPS="${SIDECHANNEL_DROPS:-0}"
else
    SIDECHANNEL_DROPS=0
fi

if [[ "${PROXY_REQUESTS}" -eq 0 ]]; then
    echo "  FAIL  LLM dispatch parity: proxy log shows 0 chat/completions requests (proxy not exercised?)"
    FAIL=1
else
    DROPPED=$((PROXY_REQUESTS - LLM_TRACES))
    if [[ "${DROPPED}" -gt "${SIDECHANNEL_DROPS}" ]]; then
        UNEXPLAINED=$((DROPPED - SIDECHANNEL_DROPS))
        echo "  FAIL  LLM dispatch parity: ${PROXY_REQUESTS} requests, ${LLM_TRACES} traces — DROPPED ${DROPPED} (${UNEXPLAINED} unexplained after allowing ${SIDECHANNEL_DROPS} sidechannel)"
        FAIL=1
    elif [[ "${DROPPED}" -gt 0 ]]; then
        # All drops attributed to sidechannel wildcard mocks on the
        # Anthropic endpoint — a LiteLLM mock_response quirk, not ours.
        echo "  PASS  LLM dispatch parity: ${PROXY_REQUESTS} requests, ${LLM_TRACES} traces (-${DROPPED} expected from sidechannel mocks)"
    else
        EXTRA=$((LLM_TRACES - PROXY_REQUESTS))
        if [[ "${EXTRA}" -gt 0 ]]; then
            echo "  PASS  LLM dispatch parity: ${PROXY_REQUESTS} proxy requests, ${LLM_TRACES} traces (+${EXTRA} from non-proxy litellm calls)"
        else
            echo "  PASS  LLM dispatch parity: ${PROXY_REQUESTS} requests, ${LLM_TRACES} traces"
        fi
    fi
fi

echo
if [[ ${FAIL} -eq 0 ]]; then
    echo "[smoke] PASS"
    exit 0
else
    echo "[smoke] FAIL"
    exit 1
fi
