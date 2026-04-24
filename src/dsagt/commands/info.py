"""
dsagt info <project> — summary of MLflow traces for triage.

A terse, read-only snapshot of what the project's mlflow.db already knows:
counts and token totals grouped by session and by source (agent turn,
embedding, extraction).  Errors surface inline so a user can see "which
session broke" without scrolling.  Deep investigation still happens in
the MLflow UI (``dsagt mlflow <project>``); this command is the triage
layer that tells you *where* to look first.

All aggregation reads ``trace_metadata`` — MLflow stamps per-trace token
usage as a JSON blob under ``mlflow.trace.tokenUsage`` and our own
observability layer stamps ``dsagt.source``, ``dsagt.agent``, and
``mlflow.trace.session`` (see ``install_mlflow_logger_with_session_tag``).
No per-span aggregation needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dsagt.session import load_config, resolve_env_vars


def _tokens(metadata: dict) -> tuple[int, int]:
    """Pull (input, output) tokens from MLflow's ``mlflow.trace.tokenUsage``.

    The value is a JSON string, not a dict — MLflow encodes structured
    metadata as strings so it round-trips through the same storage path
    as arbitrary user tags.  Missing key → (0, 0); some traces (e.g. the
    hardcoded gpt-4o-mini session-namer mock) legitimately have no usage.
    """
    raw = metadata.get("mlflow.trace.tokenUsage")
    if not raw:
        return 0, 0
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return 0, 0
    return int(d.get("input_tokens", 0)), int(d.get("output_tokens", 0))


def _fmt_count(n: int) -> str:
    """Compact integer formatter: 1234 -> '1.2k', 1234567 -> '1.2M'."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def _is_error(state) -> bool:
    """Trace state comes back as an enum whose ``.value`` is a string.

    Compare via the string form rather than the enum identity so this works
    across MLflow versions that may rename or restructure the enum.
    """
    return str(state).rsplit(".", 1)[-1].upper() == "ERROR"


def _load_traces(mlflow_db: Path, project_name: str):
    """Return (traces_df, experiment_id_or_none).

    Separate from the main reporting logic so the caller can decide what to
    print when the experiment doesn't exist yet (new project, never run).
    """
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    exp = mlflow.get_experiment_by_name(project_name)
    if exp is None:
        return None, None
    traces = mlflow.search_traces(
        locations=[exp.experiment_id], max_results=5000,
    )
    return traces, exp.experiment_id


def _report(project_name: str, config: dict, traces) -> dict:
    """Build the structured report dict.  CLI formats it; --json prints it."""
    agent_header = config.get("agent", "-")
    model_header = config.get("llm", {}).get("model", "-")

    if traces is None or traces.empty:
        return {
            "project": project_name,
            "agent": agent_header,
            "model": model_header,
            "total_traces": 0,
            "total_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "by_source": [],
            "by_session": [],
            "errors": [],
        }

    # Extract flat columns we'll group on.  Using .apply over the metadata
    # column once up front keeps pandas from re-parsing the dict on every
    # groupby.
    md = traces["trace_metadata"].apply(lambda m: m or {})
    in_toks = md.apply(lambda m: _tokens(m)[0])
    out_toks = md.apply(lambda m: _tokens(m)[1])
    source = md.apply(lambda m: m.get("dsagt.source") or "unknown")
    session = md.apply(lambda m: m.get("mlflow.trace.session") or "(no-session)")
    agent = md.apply(lambda m: m.get("dsagt.agent") or "-")
    errored = traces["state"].apply(_is_error)

    df = traces.copy()
    df["_in"] = in_toks
    df["_out"] = out_toks
    df["_source"] = source
    df["_session"] = session
    df["_agent"] = agent
    df["_err"] = errored

    def _group_rows(col: str, sort_by_recency: bool = False) -> list[dict]:
        rows = []
        groups = df.groupby(col)
        for key, g in groups:
            row = {
                col.lstrip("_"): key,
                "traces": int(len(g)),
                "input_tokens": int(g["_in"].sum()),
                "output_tokens": int(g["_out"].sum()),
                "errors": int(g["_err"].sum()),
            }
            if col == "_session":
                row["agent"] = g["_agent"].mode().iloc[0] if not g["_agent"].empty else "-"
                row["latest_request_time"] = g["request_time"].max()
            rows.append(row)
        if sort_by_recency:
            rows.sort(key=lambda r: r["latest_request_time"], reverse=True)
        else:
            rows.sort(key=lambda r: r["traces"], reverse=True)
        return rows

    errors = []
    for _, row in df[df["_err"]].iterrows():
        # Trace inputs live in trace_metadata['mlflow.traceInputs'] as JSON;
        # the request column is the display-friendly form.  For an error we
        # just need "which session, which source, when" — the UI has the
        # payload.
        errors.append({
            "session": row["_session"],
            "source": row["_source"],
            "request_time": row["request_time"],
            "trace_id": row["trace_id"],
        })

    return {
        "project": project_name,
        "agent": agent_header,
        "model": model_header,
        "total_traces": int(len(df)),
        "total_errors": int(df["_err"].sum()),
        "input_tokens": int(df["_in"].sum()),
        "output_tokens": int(df["_out"].sum()),
        "by_source": _group_rows("_source"),
        "by_session": _group_rows("_session", sort_by_recency=True),
        "errors": errors,
    }


def _print_text(r: dict) -> None:
    print(f"Project: {r['project']}")
    print(f"  Agent:  {r['agent']}")
    print(f"  Model:  {r['model']}")
    print()

    if r["total_traces"] == 0:
        print("No traces recorded yet (run `dsagt start` to create a session).")
        return

    n_sessions = len(r["by_session"])
    print(
        f"Totals ({r['total_traces']} traces across {n_sessions} "
        f"session{'s' if n_sessions != 1 else ''}):"
    )
    print(f"  Tokens: {_fmt_count(r['input_tokens'])} in / "
          f"{_fmt_count(r['output_tokens'])} out")
    print(f"  Errors: {r['total_errors']}")
    print()

    print("By source:")
    for row in r["by_source"]:
        print(
            f"  {row['source']:<12} {row['traces']:>4} traces  "
            f"{_fmt_count(row['input_tokens']):>6} in / "
            f"{_fmt_count(row['output_tokens']):>6} out  "
            f"{row['errors']} error{'s' if row['errors'] != 1 else ''}"
        )
    print()

    print("By session (most recent first):")
    for row in r["by_session"]:
        print(
            f"  {row['session']:<36} {row['agent']:<12} "
            f"{row['traces']:>4} traces  "
            f"{_fmt_count(row['input_tokens']):>6} in / "
            f"{_fmt_count(row['output_tokens']):>6} out  "
            f"{row['errors']} error{'s' if row['errors'] != 1 else ''}"
        )

    if r["errors"]:
        print()
        print("Errors:")
        for e in r["errors"]:
            print(f"  {e['session']:<36} {e['source']:<12} trace={e['trace_id']}")


def run(project: str, as_json: bool) -> int:
    # Resolve ${ENV_VAR} references so the header shows the actual model
    # name the proxy will route (not the placeholder from dsagt_config.yaml).
    config = resolve_env_vars(load_config(project))
    pdir = Path(config["project_dir"])
    mlflow_db = pdir / "mlflow" / "mlflow.db"

    if not mlflow_db.exists():
        # New project, or one that's never been started.  Print the header
        # so the user can verify they got the right project, then a short
        # note — rather than crashing on a missing DB.
        r = {
            "project": project,
            "agent": config.get("agent", "-"),
            "model": config.get("llm", {}).get("model", "-"),
            "total_traces": 0,
            "total_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "by_source": [],
            "by_session": [],
            "errors": [],
        }
        if as_json:
            print(json.dumps(r, indent=2, default=str))
        else:
            _print_text(r)
        return 0

    traces, _ = _load_traces(mlflow_db, project)
    r = _report(project, config, traces)

    if as_json:
        print(json.dumps(r, indent=2, default=str))
    else:
        _print_text(r)
    return 0
