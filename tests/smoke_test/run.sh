#!/usr/bin/env bash
# DSAGT smoke test: non-interactive end-to-end exercise.
#
# Drives the SAME `dsagt start` lifecycle as an interactive run (config
# generation → agent in the foreground → post-session catch-up extraction).
# Serverless: there are no services to start or stop; all self-logging
# goes to the project's sqlite MLflow store.  Only the agent-launch
# step swaps from interactive to batch (`--script`).
#
# TWO sessions run back-to-back: session 1 exercises ingest, code
# registration + execution, provenance, KB retrieval, skill install,
# and explicit memory; session 2 exercises cross-session recall,
# registry persistence, and the startup catch-up path.
#
# The user's shell must already have the agent's provider creds (per
# `dsagt init` hints).
#
# Run from anywhere:
#   bash tests/smoke_test/run.sh
#   dsagt smoke-test
#
# Exit code 0 on success, non-zero on failed assertion or agent timeout.

set -uo pipefail

AGENT="${DSAGT_SMOKE_AGENT:-${1:-goose}}"   # arg or env var, default goose
# Per-agent project name so each agent's mlflow.db, trace_archive, and
# kb_index/ survive across runs, which cross-agent comparison (token use
# per agent, for example) depends on.  Without this, `dsagt rm` at the
# start of each run wipes the previous agent's state.
PROJECT="smoke-test-${AGENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSAGT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# The project is created at the default ``dsagt init`` location so smoke-test
# artifacts stay out of the dsagt source tree.  PDIR mirrors
# DEFAULT_PROJECTS_BASE in src/dsagt/session.py.
PDIR="${HOME}/dsagt-projects/${PROJECT}"

case "${AGENT}" in
    goose|claude|codex|opencode) ;;
    cline)
        # dsagt start --script hard-errors for cline: its headless CLI
        # (verified 3.0.34) never loads MCP servers, so a scripted session
        # has no dsagt tools to exercise (see agents/cline.py).  Skip rather
        # than report red checks.
        echo "[smoke] SKIP: cline headless CLI loads no MCP servers (see agents/cline.py) — hand-test via tests/manual_walkthroughs/ instead"
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
# 1. Clean slate (idempotent; silent if nothing exists)
# ---------------------------------------------------------------------------
dsagt rm "${PROJECT}" -y >/dev/null 2>&1 || true
rm -rf "${PDIR}"

# Wipe claude code's per-directory session history for the smoke project.
# Claude writes one .jsonl per past session under ~/.claude/projects/<encoded-cwd>/
# (path with all '/' replaced by '-').  Without this, claude's project-memory
# layer can leak details from prior runs into the current one and the agent
# reports things that did not happen: false hangs, fake api errors, duplicate
# results.  Only relevant for --agent claude.
if [[ "${AGENT}" == "claude" ]]; then
    smoke_path_encoded=$(echo "${PDIR}" | sed 's|/|-|g')
    rm -rf "${HOME}/.claude/projects/${smoke_path_encoded}"
fi

# ---------------------------------------------------------------------------
# 2. Init at the default ``~/dsagt-projects/`` location so smoke artifacts
#    stay out of the dsagt source tree.  --episodic so session turns are
#    written to the ``session_memory`` collection (asserted below).  The default KB
#    set includes the genesis skill catalog, which the skill-install prompt
#    relies on.
# ---------------------------------------------------------------------------
# Force the non-interactive (flag-driven) init path regardless of TTY by
# closing stdin; `dsagt init` prompts only when stdin is a TTY.
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
#    Output tees to a per-session log; the retrieval and recall
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
check "greet spec written"           "test -f '${PDIR}/skills/greet/SKILL.md'"
# Codes share the skill-standard envelope and mirror into the agent's
# native skills dir at dsagt start: the base-skill datacard-introspect at
# session 1's start, greet (registered mid-session-1) at session 2's.
check "base-skill code mirrored natively" "find '${PDIR}' -path '*skills/datacard-introspect/SKILL.md' | grep -q ."
check "greet mirrored natively"       "find '${PDIR}' -path '*skills/greet/SKILL.md' | grep -q ."
# The execution went through dsagt-run iff the record captured greet's
# actual stdout; an agent that ran the script by hand cannot fake the
# trace_archive record.  Match only the greeting prefix: it shows our
# custom --greeting arg passed through the registered code, while
# tolerating an agent that puts the wrong word in the name slot (goose
# produced "Ahoy, Ahoy!").
check "greet executed via dsagt-run" "grep -l 'Ahoy,' '${PDIR}/trace_archive/'*greet*.json"
check "greet re-run in session 2"    "test \$(ls '${PDIR}/trace_archive/'*greet*.json | wc -l) -ge 2"
check "datacard-introspect record"   "ls '${PDIR}/trace_archive/'*datacard-introspect*.json"

# -- knowledge base ----------------------------------------------------------
# Both files are written by dsagt-server's kb_ingest MCP tool: chroma.sqlite3
# is the vector DB, chroma_ids.json the internal-collection manifest
# (route.json marks routed *external* collections, which ingest never
# creates).  Checking only `test -d kb_index/knowledge` is too weak: an agent
# can satisfy it by creating an empty directory tree by hand, masking a
# broken MCP wiring (an agent whose dsagt server crashed silently has
# compensated by creating the path with mkdir).
check "knowledge ingested (ids)"     "test -f '${PDIR}/kb_index/knowledge/chroma_ids.json'"
check "knowledge ingested (vectors)" "test -f '${PDIR}/kb_index/knowledge/chroma.sqlite3'"
# GRT-42 appears only in knowledge/troubleshooting.md; the agent answering
# with it shows retrieval read the ingested docs.
check "kb retrieval answered (GRT-42)" "grep -q 'GRT-42' '${SESSION_LOG_1}'"

# -- skills ------------------------------------------------------------------
check "catalog skill installed"      "ls '${PDIR}/skills/'*/SKILL.md"

# -- memory ------------------------------------------------------------------
# Explicit memory is stored with the server-owned internals in .dsagt/.  Only
# kb_remember (called deliberately by the agent in response to "Put this in
# explicit memory") populates the file; checking non-empty catches the
# hallucination case where the agent claims it stored a fact but did not
# call the tool.
check "explicit memory recorded"     "test -s '${PDIR}/.dsagt/explicit_memories.yaml'"
# Cross-session recall: session 2's answer must carry the stored fact's
# tokens, which only kb_get_memories (or episodic retrieval) can supply;
# session 2 never read samples.csv.
check "cross-session recall"         "grep -qi 'null' '${SESSION_LOG_2}' && grep -qi 'status' '${SESSION_LOG_2}'"
# Episodic memory (enabled via --episodic) chunks+embeds every turn into
# the session_memory collection on the periodic pass.
check "episodic memory indexed"      "test -f '${PDIR}/kb_index/session_memory/chroma.sqlite3'"

# -- observability + session state -------------------------------------------
check "mlflow store has traces"      "test -s '${PDIR}/mlflow.db'"
# The periodic pass indexes trace_archive/ execution records into the code_use
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
#    turns from its on-disk transcript (the periodic pass, the shutdown flush,
#    and session 2's startup catch-up), so agent traces in the store are a
#    hard requirement for all five agents.
# ---------------------------------------------------------------------------
# The store is whichever one the session logged to: MLFLOW_TRACKING_URI when
# set (a shared tracking server), else the project's serverless sqlite file.
STORE_URI="${MLFLOW_TRACKING_URI:-sqlite:///${PDIR}/mlflow.db}"
TRACE_COUNTS=$(uv run --quiet python <<PY 2>/dev/null
import mlflow
from dsagt.observability import experiment_name
from dsagt.session import load_config
mlflow.set_tracking_uri("${STORE_URI}")
exp = mlflow.get_experiment_by_name(experiment_name(load_config("${PROJECT}")))
if exp is None:
    print("0 0"); raise SystemExit
df = mlflow.search_traces(
    locations=[exp.experiment_id],
    max_results=500,
)
# MLflowSink stamps every replayed agent trace with "dsagt.trace_id" in
# its trace metadata; DSAGT's internal MCP/dsagt-run debug traces carry a
# "dsagt.source" tag instead; the positive marker is the filter.  A
# span-attribute heuristic would count internal spans lacking that
# attribute as agent traces and mask a reader that collected nothing.
n = sum(
    1
    for _, row in df.iterrows()
    if "dsagt.trace_id" in (row.get("trace_metadata") or {})
)
print(len(df), n)
PY
)
read -r TOTAL_TRACES AGENT_TRACES <<< "${TRACE_COUNTS:-0 0}"
check "mlflow store has traces (${TOTAL_TRACES})" "test '${TOTAL_TRACES}' -gt 0"
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
