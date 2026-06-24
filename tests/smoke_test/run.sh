#!/usr/bin/env bash
# DSAGT smoke test — non-interactive end-to-end exercise.
#
# Drives the SAME `dsagt start` lifecycle as an interactive run (config
# generation → start_services → agent → run_extraction → stop_services).
# Only the agent-launch step swaps from `goose session` to `goose run -i`.
# BYOA: the user's shell must already have the agent's provider creds
# (per `dsagt init` hints).  No .env handling — that returns with the
# Phase 2 `--proxy_traces` flag.
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
# (e.g., why does claude use 10x the tokens roo does?).  Without this,
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
    goose|claude|cline|roo|codex|opencode) ;;
    *)
        echo "ERROR: agent must be one of: goose, claude, cline, roo, codex, opencode (got '${AGENT}')" >&2
        exit 2
        ;;
esac

# Proxy-mode opt-in.  ``DSAGT_SMOKE_PROXY=1`` (or arg #2 == "proxy")
# routes the run through ``dsagt start --enable-proxy``, which spawns
# dsagt-proxy and forwards the agent's LLM calls through it — required
# for cline / roo (their CLIs hardcode model-ID whitelists incompatible
# with lab-gateway aliases; the proxy translates names back to the
# upstream's served IDs via _AGENT_PRIMARY_ALIASES).  Default off.
PROXY_FLAG=""
if [[ "${DSAGT_SMOKE_PROXY:-${2:-}}" == "1" || "${DSAGT_SMOKE_PROXY:-${2:-}}" == "proxy" ]]; then
    PROXY_FLAG="--enable-proxy"
    echo "[smoke] Agent: ${AGENT} (proxy mode)"
else
    echo "[smoke] Agent: ${AGENT}"
fi

# Cline / roo only work in proxy mode (see agents/cline.py + roo.py
# module docstrings for the model-whitelist + endpoint-lockout reasons).
if [[ -z "${PROXY_FLAG}" && ( "${AGENT}" == "cline" || "${AGENT}" == "roo" ) ]]; then
    echo
    echo "[smoke] ${AGENT} requires proxy mode in BYOA — skipping."
    echo "[smoke] Re-run with: DSAGT_SMOKE_PROXY=1 dsagt smoke-test --agent ${AGENT}"
    echo "[smoke] (or pass 'proxy' as the second arg to this script)"
    exit 0
fi

cd "${DSAGT_ROOT}"

# Proxy mode needs LLM_*/EMBEDDING_* in env so the proxy can forward
# upstream.  Source .env if present (BYOA runs without it; only proxy
# runs need these vars).  Validation happens at proxy spawn time —
# session.py:_start_proxy raises if config["llm"] keys are missing.
if [[ -n "${PROXY_FLAG}" && -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "[smoke] Sourced .env for proxy mode"
fi

# ---------------------------------------------------------------------------
# 2. Clean slate (idempotent — silent if nothing exists)
# ---------------------------------------------------------------------------
dsagt stop "${PROJECT}" >/dev/null 2>&1 || true
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
dsagt init "${PROJECT}" --agent "${AGENT}"

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
echo "[smoke] Running dsagt start --script ${PROXY_FLAG} (${WALL_CLOCK_CAP}s wall-clock cap)…"
dsagt start "${PROJECT}" --script "${RENDERED_SCRIPT}" --max-turns 30 ${PROXY_FLAG} &
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
check "mlflow has traces"            "test -s '${PDIR}/mlflow/mlflow.db'"

# ---------------------------------------------------------------------------
# 6. Agent LLM-call observability: every agent turn must produce at least
#    one trace in MLflow with this session's id, tagged with the agent's
#    service.name.  Replaces the proxy-log parity check from before
#    proxy removal — the agent now emits OTel directly to MLflow's OTLP
#    receiver, so MLflow IS the source of truth.  A zero count means
#    either the agent didn't emit telemetry (e.g. CLAUDE_CODE_ENABLE_TELEMETRY=1
#    not honored) or the OTel endpoint was misconfigured.
# ---------------------------------------------------------------------------
AGENT_OTEL_SUPPORT=$(uv run --quiet python -c "from dsagt.agents import agent_otel_support; print(agent_otel_support('${AGENT}'))" 2>/dev/null)
AGENT_OTEL_SUPPORT="${AGENT_OTEL_SUPPORT:-unknown}"

AGENT_TRACES=$(uv run --quiet python <<PY 2>/dev/null
import mlflow
mlflow.set_tracking_uri("sqlite:///${PDIR}/mlflow/mlflow.db")
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
    # MCP-server traces (kb.*, registry.*, tool.execute) carry
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

# Grade by the agent's verified support tier rather than fail-or-pass:
# agents we know don't emit OTel payloads (cline, roo) get SKIP, agents
# we know partial-emit (codex) get WARN, agents we know full-emit
# (claude, goose) get PASS-or-FAIL.  See agents/__init__.py module
# docstring for the matrix.
case "${AGENT_OTEL_SUPPORT}" in
    full)
        if [[ "${AGENT_TRACES}" -eq 0 ]]; then
            echo "  FAIL  agent transparency: 0 agent LLM-call traces in MLflow (agent supports full payload but emitted none — env vars not honored?)"
            FAIL=1
        else
            echo "  PASS  agent transparency: ${AGENT_TRACES} agent LLM-call trace(s) visible in MLflow"
        fi
        ;;
    partial)
        echo "  WARN  agent transparency: ${AGENT} emits only token counts + tool names natively (${AGENT_TRACES} agent trace(s)); use 'dsagt start --enable-proxy' to capture full LLM-call payloads"
        ;;
    none)
        echo "  SKIP  agent transparency: ${AGENT} emits no payload-bearing OTel traces (${AGENT_TRACES} agent trace(s)); use 'dsagt start --enable-proxy' to make agent LLM calls visible in MLflow"
        ;;
    *)
        echo "  WARN  agent transparency: support tier unknown for ${AGENT}; ${AGENT_TRACES} agent trace(s)"
        ;;
esac

echo
if [[ ${FAIL} -eq 0 ]]; then
    echo "[smoke] PASS"
    exit 0
else
    echo "[smoke] FAIL"
    exit 1
fi
