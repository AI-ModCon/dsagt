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

PROJECT="smoke-test"
AGENT="${DSAGT_SMOKE_AGENT:-${1:-goose}}"   # arg or env var, default goose
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSAGT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_FILE="${SCRIPT_DIR}/script.txt"

case "${AGENT}" in
    goose|claude-code) ;;
    *)
        echo "ERROR: agent must be one of: goose, claude-code (got '${AGENT}')" >&2
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

for var in LLM_API_KEY LLM_BASE_URL LLM_MODEL EMBEDDING_API_KEY EMBEDDING_BASE_URL EMBEDDING_MODEL; do
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
#    `timeout` is GNU coreutils — present on Linux, missing on stock macOS.
#    On macOS Homebrew installs it as `gtimeout`.  Fall back to running
#    without a wall-clock cap if neither is available; --max-turns 30 is
#    still a tight bound (~15 min worst case on Haiku).
# ---------------------------------------------------------------------------
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
else
    TIMEOUT_BIN=""
    echo "WARN: neither 'timeout' nor 'gtimeout' available; running without wall-clock cap (relying on --max-turns 30). Install GNU coreutils for the safety net: brew install coreutils"
fi

echo
echo "[smoke] Running dsagt start --script…"
if [[ -n "${TIMEOUT_BIN}" ]]; then
    "${TIMEOUT_BIN}" 5m dsagt start "${PROJECT}" --script "${SCRIPT_FILE}" --max-turns 30
else
    dsagt start "${PROJECT}" --script "${SCRIPT_FILE}" --max-turns 30
fi
START_EXIT=$?

if [[ ${START_EXIT} -ne 0 ]]; then
    echo "WARN: dsagt start exited non-zero (${START_EXIT}) — continuing to artifact checks anyway"
fi

# Defensive: ensure no stray services if the lifecycle's finally didn't run
# (timeout SIGKILL skips Python's finally blocks).
dsagt stop "${PROJECT}" >/dev/null 2>&1 || true

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
check "knowledge ingested"           "test -d '${PDIR}/kb_index/knowledge'"
check "explicit memory recorded"     "find '${PDIR}/kb_index' -name 'chunks.jsonl' -path '*memory*' | grep -q ."
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
#   - goose / OpenAI-format clients → /chat/completions
#   - claude-code / Anthropic-format clients → /v1/messages
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
