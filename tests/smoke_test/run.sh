#!/usr/bin/env bash
# DSAGT smoke test — non-interactive end-to-end exercise.
#
# Drives the SAME `dsagt start` lifecycle as an interactive run (config
# generation → agent in the foreground → post-session run_extraction).
# Serverless: there are no services to start or stop — all self-logging
# lands in the project's sqlite MLflow store.  Only the agent-launch
# step swaps from interactive to batch (`--script`).
#
# TWO sessions run back-to-back: session 1 exercises ingest, code
# registration + execution, provenance, KB retrieval, skill install,
# and explicit memory; session 2 exercises cross-session recall,
# registry persistence, and the startup catch-up path.
#
# BYOA: the user's shell must already have the agent's provider creds
# (per `dsagt init` hints).  No .env handling.
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
# (e.g., why does claude use 10x the tokens codex does?).  Without this,
# `dsagt rm` at the start of each run wipes the previous agent's state.
PROJECT="smoke-test-${AGENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSAGT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Project lives at the default ``dsagt init`` location so smoke-test
# artifacts stay out of the dsagt source tree.  PDIR mirrors
# DEFAULT_PROJECTS_BASE in src/dsagt/session.py.
PDIR="${HOME}/dsagt-projects/${PROJECT}"

case "${AGENT}" in
    goose|claude|codex|opencode) ;;
    cline)
        # dsagt start --script hard-errors for cline (its anthropic provider
        # rewrites unrecognized model names, so batch mode is unsupported —
        # see agents/cline.py).  Skip rather than report 15 red checks; drop
        # this arm when that guard is lifted.
        echo "[smoke] SKIP: cline batch mode is unsupported (see agents/cline.py) — hand-test via tests/manual_walkthroughs/ instead"
        exit 0
        ;;
    *)
        echo "ERROR: agent must be one of: goose, claude, cline, codex, opencode (got '${AGENT}')" >&2
        exit 2
        ;;
esac

echo "[smoke] Agent: ${AGENT}"

cd "${DSAGT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Clean slate (idempotent — silent if nothing exists)
# ---------------------------------------------------------------------------
dsagt rm "${PROJECT}" -y >/dev/null 2>&1 || true
rm -rf "${PDIR}"

# Wipe claude code's per-directory session history for the smoke project.
# Claude stashes one .jsonl per past session under ~/.claude/projects/<encoded-cwd>/
# (path with all '/' replaced by '-').  Without this, claude's project-memory
# layer can leak details from prior runs into the current one and the agent
# reports things that didn't happen — false hangs, fake api errors, ghost
# duplicates.  Only relevant for --agent claude.
if [[ "${AGENT}" == "claude" ]]; then
    smoke_path_encoded=$(echo "${PDIR}" | sed 's|/|-|g')
    rm -rf "${HOME}/.claude/projects/${smoke_path_encoded}"
fi

# ---------------------------------------------------------------------------
# 2. Init at the default ``~/dsagt-projects/`` location so smoke artifacts
#    don't pollute the dsagt source tree.  --episodic so session turns land
#    in the ``session_memory`` collection (asserted below).  The default KB
#    set includes the genesis skill catalog, which the skill-install prompt
#    relies on.
# ---------------------------------------------------------------------------
# Force the non-interactive (flag-driven) init path regardless of TTY by
# closing stdin — `dsagt init` prompts only when stdin is a TTY.
dsagt init "${PROJECT}" --agent "${AGENT}" --episodic < /dev/null

# Substitute {{SMOKE_DIR}} → absolute smoke_test/ path before the agent
# sees the scripts.  Prompts can then reference data/knowledge files via
# absolute paths regardless of the agent's cwd.
RENDERED_SCRIPT_1=$(mktemp -t dsagt-smoke-script1.XXXXXX)
RENDERED_SCRIPT_2=$(mktemp -t dsagt-smoke-script2.XXXXXX)
SESSION_LOG_1=$(mktemp -t dsagt-smoke-log1.XXXXXX)
SESSION_LOG_2=$(mktemp -t dsagt-smoke-log2.XXXXXX)
trap 'rm -f "${RENDERED_SCRIPT_1}" "${RENDERED_SCRIPT_2}" "${SESSION_LOG_1}" "${SESSION_LOG_2}"' EXIT
sed "s|{{SMOKE_DIR}}|${SCRIPT_DIR}|g" "${SCRIPT_DIR}/script.txt" > "${RENDERED_SCRIPT_1}"
sed "s|{{SMOKE_DIR}}|${SCRIPT_DIR}|g" "${SCRIPT_DIR}/script2.txt" > "${RENDERED_SCRIPT_2}"

# ---------------------------------------------------------------------------
# 3. Session runner: the FULL `dsagt start` lifecycle with the agent in
#    batch mode, under a wall-clock watchdog.
#
#    Pure-bash watcher pattern instead of GNU `timeout` so the smoke test
#    works on stock macOS without `brew install coreutils`.  SIGTERM gives
#    dsagt's finally-block a chance to run post-session extraction; the
#    follow-up SIGKILL after WALL_CLOCK_GRACE catches the agent if it
#    swallows the term signal.
#
#    Output tees to a per-session log — the retrieval and recall
#    assertions grep it for facts the agent can only have gotten from
#    the KB / memory (process substitution keeps $! on dsagt itself).
# ---------------------------------------------------------------------------
WALL_CLOCK_GRACE=10  # extra seconds before SIGKILL

run_session() {
    local script_file="$1" max_turns="$2" cap="$3" log_file="$4"
    dsagt start "${PROJECT}" --script "${script_file}" --max-turns "${max_turns}" \
        > >(tee "${log_file}") 2>&1 &
    local pid=$!
    (
        sleep "${cap}"
        kill -TERM "${pid}" 2>/dev/null && \
            echo "[smoke] WARN: ${cap}s cap exceeded — sent SIGTERM to dsagt start (pid ${pid})"
        sleep "${WALL_CLOCK_GRACE}"
        kill -KILL "${pid}" 2>/dev/null && \
            echo "[smoke] WARN: dsagt start did not exit on SIGTERM — sent SIGKILL"
    ) &
    local watcher=$!
    wait "${pid}"
    local rc=$?
    # Tear down the watcher if dsagt exited on its own.
    kill -TERM "${watcher}" 2>/dev/null
    wait "${watcher}" 2>/dev/null
    # Let the tee process-substitution drain before the log is grepped.
    sleep 1
    return "${rc}"
}

echo
echo "[smoke] Session 1: ingest / register / execute / provenance / skills / memory…"
run_session "${RENDERED_SCRIPT_1}" 40 420 "${SESSION_LOG_1}"
START_EXIT=$?
if [[ ${START_EXIT} -ne 0 ]]; then
    echo "WARN: dsagt start exited non-zero (${START_EXIT}) — continuing to artifact checks anyway"
fi

# ---------------------------------------------------------------------------
# 4. Session 2: cross-session recall + registry persistence.  Its startup
#    also runs the catch-up path over session 1 (code-use indexing + the
#    pinned trace re-collect), so the post-session-2 assertions cover it.
# ---------------------------------------------------------------------------
echo
echo "[smoke] Session 2: cross-session recall + catch-up…"
run_session "${RENDERED_SCRIPT_2}" 15 240 "${SESSION_LOG_2}"
START_EXIT_2=$?
if [[ ${START_EXIT_2} -ne 0 ]]; then
    echo "WARN: session 2 dsagt start exited non-zero (${START_EXIT_2}) — continuing to artifact checks anyway"
fi

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

# -- registry + execution + provenance --------------------------------------
check "greet spec written"           "test -f '${PDIR}/codes/greet/SKILL.md'"
# Codes share the skill-standard envelope and mirror into the agent's
# native skills dir at dsagt start: the bundled scan-directory at
# session 1's start, greet (registered mid-session-1) at session 2's.
check "bundled code mirrored natively" "find '${PDIR}' -path '*skills/scan-directory/SKILL.md' | grep -q ."
check "greet mirrored natively"       "find '${PDIR}' -path '*skills/greet/SKILL.md' | grep -q ."
# The execution went through dsagt-run iff the record captured greet's
# actual stdout — an agent that ran the script by hand can't fake the
# trace_archive record.  Match only the greeting prefix: it proves our
# custom --greeting arg flowed through the registered code, while
# tolerating an agent flubbing which word goes in the name slot (goose
# produced "Ahoy, Ahoy!").
check "greet executed via dsagt-run" "grep -l 'Ahoy,' '${PDIR}/trace_archive/'*greet*.json"
check "greet re-run in session 2"    "test \$(ls '${PDIR}/trace_archive/'*greet*.json | wc -l) -ge 2"
check "scan-directory record"        "ls '${PDIR}/trace_archive/'*scan-directory*.json"

# -- knowledge base ----------------------------------------------------------
# Both files are written by dsagt-server's kb_ingest MCP tool — chroma.sqlite3
# is the actual vector DB, chroma_ids.json the internal-collection manifest
# (route.json marks routed *external* collections, which ingest never
# creates).  Checking only `test -d kb_index/knowledge` is too weak: an agent
# can satisfy it by hand-crafting an empty directory tree, masking a broken
# MCP wiring (which is exactly what we hit when cline's dsagt server crashed
# silently and the LLM compensated by mkdir-ing the path).
check "knowledge ingested (ids)"     "test -f '${PDIR}/kb_index/knowledge/chroma_ids.json'"
check "knowledge ingested (vectors)" "test -f '${PDIR}/kb_index/knowledge/chroma.sqlite3'"
# GRT-42 lives only in knowledge/troubleshooting.md — the agent answering
# with it proves retrieval reached the ingested docs.
check "kb retrieval answered (GRT-42)" "grep -q 'GRT-42' '${SESSION_LOG_1}'"

# -- skills ------------------------------------------------------------------
check "catalog skill installed"      "ls '${PDIR}/skills/'*/SKILL.md"

# -- memory ------------------------------------------------------------------
# Explicit memory lives with the server-owned internals in .dsagt/.  Only
# kb_remember (called deliberately by the agent in response to "Put this in
# explicit memory") populates the file; checking non-empty catches the
# hallucination case where the agent claims it stored a fact but didn't
# actually call the tool.
check "explicit memory recorded"     "test -s '${PDIR}/.dsagt/explicit_memories.yaml'"
# Cross-session recall: session 2's answer must carry the stored fact's
# tokens, which only kb_get_memories (or episodic retrieval) can supply —
# session 2 never saw samples.csv.
check "cross-session recall"         "grep -qi 'null' '${SESSION_LOG_2}' && grep -qi 'status' '${SESSION_LOG_2}'"
# Episodic memory (enabled via --episodic) chunks+embeds every turn into
# the session_memory collection on the heartbeat.
check "episodic memory indexed"      "test -f '${PDIR}/kb_index/session_memory/chroma.sqlite3'"

# -- observability + session state -------------------------------------------
check "mlflow store has traces"      "test -s '${PDIR}/mlflow.db'"
# The heartbeat indexes trace_archive/ execution records into the code_use
# collection (plus a startup catch-up in session 2).
check "code_use collection indexed"  "test -f '${PDIR}/kb_index/code_use/chroma.sqlite3'"
# state.yaml is the anchor for crash catch-up: both sessions logged, and
# session 1 carries the trace_source token the session-2 catch-up pinned.
check "state.yaml logged 2 sessions" "uv run --quiet python -c \"
import yaml, sys
s = yaml.safe_load(open('${PDIR}/.dsagt/state.yaml'))
sessions = s.get('sessions') or []
sys.exit(0 if len(sessions) >= 2 and sessions[0].get('trace_source') else 1)\""
check "dsagt info runs"              "dsagt info '${PROJECT}'"

# ---------------------------------------------------------------------------
# 6. Agent LLM-call transparency: the trace pipeline recovers every agent's
#    turns from its on-disk transcript (heartbeat + graceful-shutdown flush,
#    backstopped by session 2's startup catch-up), so agent traces in the
#    store are a hard requirement for all five agents.
# ---------------------------------------------------------------------------
AGENT_TRACES=$(uv run --quiet python <<PY 2>/dev/null
import mlflow
mlflow.set_tracking_uri("sqlite:///${PDIR}/mlflow.db")
exp = mlflow.get_experiment_by_name("${PROJECT}")
if exp is None:
    print(0); raise SystemExit
df = mlflow.search_traces(
    locations=[exp.experiment_id],
    max_results=500,
)
# MLflowSink stamps every replayed agent trace with "dsagt.trace_id" in
# its trace metadata; DSAGT's internal MCP/dsagt-run debug traces carry a
# "dsagt.source" tag instead — the positive marker is the reliable
# filter.  A service.name span heuristic previously counted internal
# spans lacking that attribute as agent traces, masking a codex reader
# that collected nothing.
n = sum(
    1
    for _, row in df.iterrows()
    if "dsagt.trace_id" in (row.get("trace_metadata") or {})
)
print(n)
PY
)
AGENT_TRACES="${AGENT_TRACES:-0}"
check "agent traces recovered (${AGENT_TRACES})" "test '${AGENT_TRACES}' -gt 0"

echo
if [[ ${FAIL} -eq 0 ]]; then
    echo "[smoke] PASS"
    exit 0
else
    echo "[smoke] FAIL"
    echo "[smoke] session logs kept: ${SESSION_LOG_1} ${SESSION_LOG_2}"
    trap - EXIT
    rm -f "${RENDERED_SCRIPT_1}" "${RENDERED_SCRIPT_2}"
    exit 1
fi
