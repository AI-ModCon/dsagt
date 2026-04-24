"""
Sidechannel model-call handling.

Every agent platform hardcodes a small/fast model for internal features
(goose → gpt-4o-mini session-namer; claude-code → claude-haiku-4-5... title
generator; the next agent will pick its own).  When the user's gateway
doesn't carry that exact bare name — which is the norm for lab gateways
that alias every model — those requests would 400 and clutter MLflow.

Rather than maintain a per-vendor list of known hardcoded names (which
rots as vendors rename their sidechannel models), DSAGT catches all of
them with one wildcard LiteLLM route, records which names fired, and
surfaces a single yellow warning at session teardown so the user can
distinguish a harmless sidechannel from a typo in their own config.

All sidechannel logic — the routing YAML, the detection rule, the log
format, the warning text, the doc pointer — lives here.  Callers import
the specific thing they need:

    proxy_server._generate_config   → WILDCARD_ROUTE_YAML
    session.start_services          → PRIMARY_MODEL_ENV
    provenance._handle_success      → record()
    cli._cmd_start (teardown)       → print_warning()

Add new call sites by importing from this module; do not duplicate
constants or rewrite the detection rule elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

#: Env var the parent process exports so the proxy's DSAGT callback can tell
#: "this is the configured primary model" from "this is a sidechannel hit".
PRIMARY_MODEL_ENV = "DSAGT_PRIMARY_MODEL"

#: JSONL file (one entry per intercepted call) that the proxy subprocess
#: appends to and the parent reads at teardown.  Lives at the project
#: directory root, adjacent to ``trace_archive/``.
LOG_FILENAME = "sidechannel.jsonl"

#: Canned reply the wildcard returns.  Short enough that goose's
#: session-namer (expects ≤4 words) and claude-code's title generator both
#: accept it without error.
_CANNED_RESPONSE = "session"

#: Where the user can read the longer explanation.  Printed in the warning.
DOC_LOCATION = "README.md § Sidechannel model calls"


# ---------------------------------------------------------------------------
# Proxy-side: LiteLLM route
# ---------------------------------------------------------------------------

#: YAML fragment appended to the proxy's ``model_list`` after the primary
#: route.  LiteLLM prefers exact matches over wildcards, so the configured
#: model still routes normally; everything else falls through to the mock.
#:
#: ``api_base`` has to be a syntactically valid URL but is never dialed —
#: ``mock_response`` short-circuits upstream entirely.
WILDCARD_ROUTE_YAML = f"""\
  - model_name: "*"
    litellm_params:
      model: openai/dsagt-sidechannel-catchall
      api_base: http://invalid.local
      api_key: unused
      mock_response: "{_CANNED_RESPONSE}"
"""


# ---------------------------------------------------------------------------
# Proxy-side: detection + recording
# ---------------------------------------------------------------------------

def _client_requested_model(kwargs: dict) -> str | None:
    """Return the model name the client sent, not the post-routing target.

    LiteLLM mutates ``kwargs["model"]`` to the resolved route during
    completion, so by the time callbacks fire the original name is gone
    from there — a wildcard hit's ``kwargs["model"]`` is always the
    wildcard's ``litellm_params.model`` target.

    ``standard_logging_object.model_group`` preserves the name from the
    client's request body.  For exact matches it equals the configured
    primary; for wildcard hits it's the actual sidechannel name (e.g.
    ``gpt-4o-mini``, ``claude-haiku-4-5-20251001``) — which is what the
    warning needs to show.
    """
    slo = kwargs.get("standard_logging_object") or {}
    if isinstance(slo, dict):
        grp = slo.get("model_group")
        if grp:
            return grp.split("/", 1)[-1]
    # Fallback: whatever kwargs has. Worse (may be the catchall name) but
    # still informative if standard_logging_object isn't populated.
    m = kwargs.get("model") or ""
    return m.split("/", 1)[-1] or None


def record(records_dir: Path, kwargs: dict) -> None:
    """Append a JSONL entry when a request hit the wildcard.

    Detection rule: the LiteLLM callback's ``kwargs["model"]`` (after
    stripping any ``provider/`` prefix) differs from ``$PRIMARY_MODEL_ENV``.
    Called from the DSAGT callback's success handler, so only successful
    calls get logged — failures land in MLflow as errors regardless.

    No-ops when ``PRIMARY_MODEL_ENV`` isn't set (tests, direct-callback
    users) or when the requested model matches primary.
    """
    primary = os.environ.get(PRIMARY_MODEL_ENV)
    if not primary:
        return

    requested = _client_requested_model(kwargs)
    if not requested or requested == primary:
        return

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": requested,
        "agent": os.environ.get("DSAGT_AGENT", ""),
        "session": os.environ.get("DSAGT_SESSION_ID", ""),
    }
    path = Path(records_dir).parent / LOG_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug("sidechannel log append failed: %s", e)


# ---------------------------------------------------------------------------
# Parent-side: teardown warning
# ---------------------------------------------------------------------------

def print_warning(project_dir: Path, session_id: str | None) -> None:
    """Read ``LOG_FILENAME`` under *project_dir*, dedup within *session_id*,
    and print a yellow warning to stdout.

    No-op when nothing was logged for this session (common case).  The
    warning lists each unique model that hit the wildcard along with the
    call count, and points the user at ``DOC_LOCATION`` for the two
    possible causes (harmless sidechannel vs config typo).

    ANSI colors are only emitted when stdout is a TTY — CI logs stay clean.
    """
    log_path = Path(project_dir) / LOG_FILENAME
    if not log_path.exists():
        return

    # Dedup by model name within the current session only.  The file is
    # append-only across runs, so older sessions' entries are still present
    # but don't belong in this run's warning.
    seen: dict[str, int] = {}
    try:
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if session_id and entry.get("session") != session_id:
                continue
            model = entry.get("model") or "<unknown>"
            seen[model] = seen.get(model, 0) + 1
    except (OSError, ValueError):
        return

    if not seen:
        return

    tty = sys.stdout.isatty()
    yellow = "\033[33m" if tty else ""
    bold = "\033[1m" if tty else ""
    reset = "\033[0m" if tty else ""

    print()
    print(f"{yellow}{bold}  ⚠ Sidechannel model calls intercepted:{reset}")
    for model, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        s = "s" if count != 1 else ""
        print(f"{yellow}      {model}  ({count} call{s}){reset}")
    print(f"{yellow}    Two possible causes:{reset}")
    print(f"{yellow}      (1) agent sidechannel (e.g. title generator) — safe to ignore{reset}")
    print(f"{yellow}      (2) typo in dsagt_config.yaml llm.model — these replies are canned, not real{reset}")
    print(f"{yellow}    See: {DOC_LOCATION}{reset}")
