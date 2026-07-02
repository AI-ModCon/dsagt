#!/usr/bin/env bash
# DSAGT smoke test — non-interactive end-to-end exercise.
#
# Drives the SAME `dsagt start` lifecycle as an interactive run (config
# generation → start_services → agent → run_extraction → stop_services).
# Only the agent-launch step swaps from `goose session` to `goose run -i`.
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
SCRIPT_FILE="${SCRIPT_DIR}/script.txt"
# Project lives at the default ``dsagt init`` location so smoke-test
# artifacts stay out of the dsagt source tree.  PDIR mirrors
# DEFAULT_PROJECTS_BASE in src/dsagt/session.py.
PDIR="${HOME}/dsagt-projects/${PROJECT}"

case "${AGENT}" in
    goose|claude|cline|codex|opencode) ;;
    *)
        echo "ERROR: agent must be one of: goose, claude, cline, codex, opencode (got '${AGENT}')" >&2
        exit 2
        ;;
esac

echo "[smoke] Agent: ${AGENT}"

cd "${DSAGT_ROOT}"

# ---------------------------------------------------------------------------
# 2. Clean slate (idempotent — silent if nothing exists)
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
# 3. Init at the default ``~/dsagt-projects/`` location so smoke artifacts
#    don't pollute the dsagt source tree.  The agent's cwd will be
#    ``${PDIR}``; the script template uses ``{{SMOKE_DIR}}`` placeholders
#    that we substitute below so prompt paths resolve regardless of where
#    PDIR lives.
# ---------------------------------------------------------------------------
# Force the non-interactive (flag-driven) init path regardless of TTY by
# closing stdin — `dsagt init` prompts only when stdin is a TTY.
dsagt init "${PROJECT}" --agent "${AGENT}" < /dev/null

# Substitute {{SMOKE_DIR}} → absolute smoke_test/ path before the agent
# sees the script.  Prompts can then reference data/knowledge files via
# absolute paths regardless of the agent's cwd.
RENDERED_SCRIPT=$(mktemp -t dsagt-smoke-script.XXXXXX)
trap 'rm -f "${RENDERED_SCRIPT}"' EXIT
sed "s|{{SMOKE_DIR}}|${SCRIPT_DIR}|g" "${SCRIPT_FILE}" > "${RENDERED_SCRIPT}"

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
dsagt start "${PROJECT}" --script "${RENDERED_SCRIPT}" --max-turns 30 &
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

# Serverless: ``dsagt start`` runs the agent in the foreground and owns no
# background services, so there's nothing to reap here.

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

check "csvtool_filter spec written"  "test -f '${PDIR}/codes/csvtool_filter.md'"
check "trace_archive has records"    "ls '${PDIR}/trace_archive/'*.json | grep -q ."
check "scan_directory record"        "ls '${PDIR}/trace_archive/'*scan_directory*.json | grep -q ."
# Both files are written by dsagt-server's kb_ingest MCP tool — chroma.sqlite3
# is the actual vector DB, route.json is the collection manifest.  Checking
# only `test -d kb_index/knowledge` is too weak: an agent can satisfy it by
# hand-crafting an empty directory tree, masking a broken MCP wiring (which is
# exactly what we hit when cline's dsagt server crashed silently and the LLM
# compensated by mkdir-ing the path).
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
check "mlflow store has traces"      "test -s '${PDIR}/mlflow.db'"

# ---------------------------------------------------------------------------
# 6. Agent LLM-call observability: informational only in Phase 1.
#    DSAGT no longer forces native agent OTel emission (the OTLP-routing
#    env + telemetry flags were removed) — agent LLM-call history is
#    recovered post-hoc from the on-disk transcript, which lands in
#    Phase 2's TracePipeline.  Until then only claude (via the
#    ``mlflow autolog claude`` Stop hook over its transcript) and DSAGT's
#    own MCP / dsagt-run spans populate the serverless ``mlflow.db`` store.
#    So we report the agent-trace count but never FAIL on it here.
# ---------------------------------------------------------------------------
AGENT_OTEL_SUPPORT=$(uv run --quiet python -c "from dsagt.agents import agent_otel_support; print(agent_otel_support('${AGENT}'))" 2>/dev/null)
AGENT_OTEL_SUPPORT="${AGENT_OTEL_SUPPORT:-unknown}"

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
n = 0
for _, row in df.iterrows():
    spans = row.get("spans") or []
    # Match by service.name on root span — agent-emitted traces only.
    # MCP-server traces (kb.*, registry.*, code.execute) carry
    # service.name = "dsagt-server" / "dsagt-run" and shouldn't count
    # toward agent turn parity.
    for s in spans:
        attrs = getattr(s, "attributes", None) or (
            s.get("attributes") if isinstance(s, dict) else None
        )
        if attrs and not str(attrs.get("service.name", "")).startswith("dsagt-"):
            n += 1
            break
print(n)
PY
)
AGENT_TRACES="${AGENT_TRACES:-0}"

# Phase 1: informational only — native OTel harvest was removed, transcript
# capture lands in Phase 2.  Never FAIL on the agent-trace count here.
if [[ "${AGENT_TRACES}" -gt 0 ]]; then
    echo "  INFO  agent transparency: ${AGENT_TRACES} agent trace(s) in the mlflow.db store (claude autolog / Phase-2 transcript capture)"
else
    echo "  INFO  agent transparency: 0 agent LLM-call traces yet (${AGENT}, tier=${AGENT_OTEL_SUPPORT}) — restored by Phase 2's transcript pipeline"
fi

echo
if [[ ${FAIL} -eq 0 ]]; then
    echo "[smoke] PASS"
    exit 0
else
    echo "[smoke] FAIL"
    exit 1
fi
