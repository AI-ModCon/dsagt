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
import os
import re
import sys
from pathlib import Path

import yaml

from dsagt.session import load_config, project_dir, resolve_env_vars

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")
_SECRET_LEAF_KEYS = {"api_key"}
# Internal/derived sections — irrelevant for "where does this credential
# come from" triage and would just clutter the output.
_CONFIG_SOURCE_SKIP_PREFIXES = ("categories.", "extraction.", "knowledge.")


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a flat dict.  Ignores comments and blanks."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _mask_secret(value: str) -> str:
    """Show first/last 4 chars of a secret, mask the middle."""
    if len(value) <= 12:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _flatten(d: dict, prefix: str = ""):
    """Yield ('dotted.path', leaf_value) pairs from a nested dict."""
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten(v, path)
        else:
            yield path, v


def _config_sources(project_name: str, env_file_path: Path) -> list[dict]:
    """Return per-leaf source info for the project's dsagt_config.yaml.

    Walks the *raw* YAML (not the env-resolved version) so ${VAR} references
    are visible.  For each leaf, reports where the resolved value came from:
    ``.env``, ``environment``, ``config`` (literal in YAML), or
    ``unresolved`` (``${VAR}`` with no value anywhere).
    """
    pdir = project_dir(project_name)
    raw = yaml.safe_load((pdir / "dsagt_config.yaml").read_text()) or {}
    env_file_vars = _read_env_file(env_file_path)

    rows: list[dict] = []
    for path, value in _flatten(raw):
        if any(path.startswith(p) for p in _CONFIG_SOURCE_SKIP_PREFIXES):
            continue
        leaf = path.rsplit(".", 1)[-1]
        is_secret = leaf in _SECRET_LEAF_KEYS

        if not isinstance(value, str) or not _ENV_VAR_RE.match(value):
            display = _mask_secret(str(value)) if is_secret else str(value)
            rows.append({"path": path, "value": display, "source": "config"})
            continue

        var = _ENV_VAR_RE.match(value).group(1)
        if var in env_file_vars:
            source = ".env"
            resolved = env_file_vars[var]
        elif var in os.environ:
            source = "environment"
            resolved = os.environ[var]
        else:
            source = "unresolved"
            resolved = value
        if is_secret and source != "unresolved":
            resolved = _mask_secret(resolved)
        rows.append({"path": path, "value": resolved, "source": source})
    return rows


def _print_config_sources(rows: list[dict], env_file_path: Path) -> None:
    if not rows:
        return
    path_w = max(len(r["path"]) for r in rows)
    val_w = max(len(str(r["value"])) for r in rows)
    print(f"Configuration (env file: {env_file_path}):")
    for r in rows:
        print(
            f"  {r['path']:<{path_w}}  {str(r['value']):<{val_w}}  "
            f"(from {r['source']})"
        )
    print()


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

    config_sources = r.get("config_sources") or []
    env_file_path = r.get("env_file_path")
    if config_sources and env_file_path:
        _print_config_sources(config_sources, Path(env_file_path))

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

    env_file_path = Path.cwd() / ".env"
    sources = _config_sources(project, env_file_path)

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
            "config_sources": sources,
            "env_file_path": str(env_file_path),
        }
        if as_json:
            print(json.dumps(r, indent=2, default=str))
        else:
            _print_text(r)
        return 0

    traces, _ = _load_traces(mlflow_db, project)
    r = _report(project, config, traces)
    r["config_sources"] = sources
    r["env_file_path"] = str(env_file_path)

    if as_json:
        print(json.dumps(r, indent=2, default=str))
    else:
        _print_text(r)
    return 0
