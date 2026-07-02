"""
dsagt info <project> — summary of MLflow traces for triage.

A terse, read-only snapshot of what the project's serverless
``sqlite:///<pdir>/mlflow.db`` store already knows: counts and token
totals grouped by session and by source (agent turn, embedding,
extraction).  Errors surface inline so a user can see "which session
broke" without scrolling.  Deep investigation still happens in the MLflow
UI (``mlflow ui --backend-store-uri sqlite:///<pdir>/mlflow.db``); this
command is the triage layer that tells you *where* to look first.

Aggregation reads ``trace_metadata`` for token totals + session id
(MLflow stamps per-trace token usage as a JSON blob under
``mlflow.trace.tokenUsage``; the live tracer stamps ``mlflow.trace.session``).

Source bucketing reads the metadata DSAGT itself stamps on each trace — no
span inspection:
  - ``memory`` / ``skill`` / ``knowledge`` / ``registry`` — internal debug
    traces, from the ``dsagt.source`` tag (the MCP tool category the agent
    invoked, set on the trace root by the dispatch shell).
  - ``execution`` — dsagt-run tool-execute traces (``dsagt.source``).
  - ``claude`` / ``goose`` / ``cline`` / ``codex`` — agent traces, from the
    ``dsagt.agent`` metadata stamped by ``MLflowSink`` (the bulk of traffic).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from dsagt.session import load_config, project_dir, resolve_env_vars

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")
_SECRET_LEAF_KEYS = {"api_key"}
# Internal/derived sections — irrelevant for "where does this credential
# come from" triage and would just clutter the output.
_CONFIG_SOURCE_SKIP_PREFIXES = ("knowledge.", "skills.")


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


def _config_sources(project_name: str) -> list[dict]:
    """Return per-leaf source info for the project's .dsagt/config.yaml.

    Walks the *raw* YAML (not the env-resolved version) so ${VAR} references
    are visible.  For each leaf, reports where the resolved value came from:
    ``config`` (literal in YAML), ``shell`` (``${VAR}`` resolved against
    ``os.environ``), or ``unresolved`` (``${VAR}`` with no value anywhere).
    No ``.env`` file is read — dsagt-internal config lives in
    ``.dsagt/config.yaml``; user-provided shell exports are the only other
    source.
    """
    pdir = project_dir(project_name)
    raw = yaml.safe_load((pdir / ".dsagt" / "config.yaml").read_text()) or {}

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
        if var in os.environ:
            source = "shell"
            resolved = os.environ[var]
        else:
            source = "unresolved"
            resolved = value
        if is_secret and source != "unresolved":
            resolved = _mask_secret(resolved)
        rows.append({"path": path, "value": resolved, "source": source})
    return rows


def _print_kb_collections(rows: list[dict]) -> None:
    """Render the per-collection chunk counts.

    For collections whose chunks carry a ``source`` metadata field
    (``tools``, ``skills``), shows the bundled-vs-project split inline
    so the user can see what the agent's local registry contributed
    over the bundled defaults.
    """
    if not rows:
        return
    name_w = max(len(r["collection"]) for r in rows)
    print("Knowledge base:")
    for r in rows:
        chunks = _fmt_count(r["chunks"])
        source_breakdown = ""
        if r["by_source"]:
            parts = [f"{k}={v}" for k, v in sorted(r["by_source"].items())]
            source_breakdown = f"  ({', '.join(parts)})"
        print(f"  {r['collection']:<{name_w}}  {chunks:>6} chunks{source_breakdown}")
    print()


def _print_kb_retrieval(rows: list[dict]) -> None:
    """Render per-session ``kb.search`` activity.

    Quiet (single line + table) when present, omitted when no kb.search
    spans exist (e.g. the agent never queried the knowledge base).
    """
    if not rows:
        return
    print("KB retrieval (kb.search calls per session):")
    sess_w = max(len(r["session"]) for r in rows)
    for r in rows:
        print(
            f"  {r['session']:<{sess_w}}  {r['searches']:>4} searches  "
            f"{r['hits']:>5} hits returned"
        )
    print()


def _print_config_sources(rows: list[dict]) -> None:
    if not rows:
        return
    path_w = max(len(r["path"]) for r in rows)
    val_w = max(len(str(r["value"])) for r in rows)
    print("Configuration:")
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
    as arbitrary user tags.  Missing key → (0, 0); some traces (e.g. an
    agent's internal title-gen / session-namer call) legitimately have no
    usage.
    """
    raw = metadata.get("mlflow.trace.tokenUsage")
    if not raw:
        return 0, 0
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return 0, 0
    return int(d.get("input_tokens", 0)), int(d.get("output_tokens", 0))


def _tokens_from_spans(spans) -> tuple[int, int]:
    """Sum ``input_tokens`` / ``output_tokens`` across all LLM spans.

    Native claude OTel emission stamps these as span attributes on
    ``claude_code.llm_request`` spans (string-encoded — MLflow's OTLP
    receiver JSON-encodes every attribute).  When ``mlflow.trace.tokenUsage``
    isn't present on the trace metadata we aggregate from the spans
    themselves so the totals aren't always zero.

    Returns ``(0, 0)`` for non-LLM traces (kb.search, tool.execute) —
    those don't carry token attributes.
    """
    if spans is None:
        return 0, 0
    in_tot = out_tot = 0
    try:
        for span in spans:
            attrs = getattr(span, "attributes", None)
            if attrs is None and isinstance(span, dict):
                attrs = span.get("attributes")
            if not attrs:
                continue
            in_tot += _coerce_token_count(attrs.get("input_tokens"))
            out_tot += _coerce_token_count(attrs.get("output_tokens"))
    except (TypeError, AttributeError):
        return 0, 0
    return in_tot, out_tot


def _coerce_token_count(v) -> int:
    """Token attrs come through as JSON-encoded strings (``'"510"'`` or
    ``'510'``), int, or None.  Defensive parse — anything unparseable
    returns 0 rather than crashing the report."""
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().strip('"')
        try:
            return int(s) if s else 0
        except ValueError:
            return 0
    return 0


def _fmt_count(n: int) -> str:
    """Compact integer formatter: 1234 -> '1.2k', 1234567 -> '1.2M'."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def _row_source_for(tags: dict, metadata: dict) -> str:
    """Bucket a trace by who emitted it, from the metadata DSAGT itself stamps.

    No span inspection: internal debug traces carry an explicit ``dsagt.source``
    tag (the MCP tool category — ``memory`` / ``skill`` / ``knowledge`` /
    ``registry`` — or ``execution`` for dsagt-run), set on the trace root by the
    MCP dispatch shell / ``code_execute_span``.  Agent traces carry ``dsagt.agent``
    metadata, stamped by ``MLflowSink``.  Neither overlaps, so the bucket is a
    direct lookup; anything else (a stray trace with neither) is ``"unknown"``.
    """
    src = (tags or {}).get("dsagt.source")
    if src:
        return src
    agent = (metadata or {}).get("dsagt.agent")
    if agent and agent != "-":
        return agent
    return "unknown"


def _is_error(state) -> bool:
    """Trace state comes back as an enum whose ``.value`` is a string.

    Compare via the string form rather than the enum identity so this works
    across MLflow versions that may rename or restructure the enum.
    """
    return str(state).rsplit(".", 1)[-1].upper() == "ERROR"


def _project_created(pdir: Path) -> str | None:
    """Best-effort project-start date from the project directory's metadata.

    Uses ``st_birthtime`` where the OS records it (macOS, BSDs); falls
    back to ``st_ctime`` on Linux (which is "change time", not "creation
    time", but is a reasonable proxy for a project directory written
    once at ``dsagt init``).  Returns ``YYYY-MM-DD`` or ``None`` if the
    stat fails.
    """
    try:
        st = pdir.stat()
    except OSError:
        return None
    ts = getattr(st, "st_birthtime", None) or st.st_ctime
    if not ts:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _kb_collections(pdir: Path) -> list[dict]:
    """Per-collection summary read directly from ``<project>/kb_index/``.

    Reports chunk count and (when present) a ``metadata.source`` breakdown
    so tools/skills collections can show bundled-vs-project at a glance.
    Counts come from line-counting ``chunks.jsonl`` rather than loading
    the vector index — fast, and survives even if Chroma's sqlite is locked.
    """
    kb_dir = pdir / "kb_index"
    if not kb_dir.exists():
        return []
    rows: list[dict] = []
    for sub in sorted(kb_dir.iterdir()):
        chunks_file = sub / "chunks.jsonl"
        if not sub.is_dir() or not chunks_file.exists():
            continue
        n_chunks = 0
        sources: dict[str, int] = {}
        with chunks_file.open() as f:
            for line in f:
                n_chunks += 1
                try:
                    src = json.loads(line).get("metadata", {}).get("source")
                except (ValueError, TypeError):
                    src = None
                if src:
                    sources[src] = sources.get(src, 0) + 1
        rows.append(
            {
                "collection": sub.name,
                "chunks": n_chunks,
                "by_source": sources,
            }
        )
    return rows


def _skills(pdir: Path) -> list[dict]:
    """Installed + bundled skills for the project.

    Reads the project's ``skills/`` plus the bundled skill dirs via
    ``SkillRegistry`` (no embedder needed — this is a directory scan, not a
    search).  Returns ``[{"name", "description"}, ...]``; empty on any
    failure so the report never crashes on a malformed skill.
    """
    try:
        from dsagt.registry import SkillRegistry

        skills = SkillRegistry(runtime_dir=pdir, kb=None).list_skills()
    except Exception:
        return []
    return [
        {"name": s.get("name", "?"), "description": s.get("description") or ""}
        for s in skills
    ]


def _print_skills(rows: list[dict]) -> None:
    """Render the installed/bundled skill list (name — truncated description)."""
    if not rows:
        return
    name_w = max(len(r["name"]) for r in rows)
    print(f"Skills ({len(rows)}):")
    for r in rows:
        desc = r["description"][:80]
        print(f"  {r['name']:<{name_w}}  {desc}")
    print()


def _kb_retrieval(traces) -> list[dict]:
    """Per-session ``kb.search`` activity pulled from MLflow trace spans.

    Each ``kb.search`` span carries a ``hits`` attribute (set by the
    ``traced`` decorator on ``KnowledgeBase.search``).  Group by
    ``mlflow.trace.session`` so the user sees which session leaned hardest
    on retrieval and how many results it actually got back.
    """
    if traces is None or traces.empty:
        return []
    rows: dict[str, dict] = {}
    for _, row in traces.iterrows():
        spans = row.get("spans")
        if not spans:
            continue
        md = row.get("trace_metadata") or {}
        session = md.get("mlflow.trace.session") or "(no-session)"
        for span in spans:
            name = getattr(span, "name", None) or (
                span.get("name") if isinstance(span, dict) else None
            )
            if name != "kb.search":
                continue
            attrs = (
                getattr(span, "attributes", None)
                or (span.get("attributes") if isinstance(span, dict) else None)
                or {}
            )
            try:
                hits = int(attrs.get("hits", 0))
            except (TypeError, ValueError):
                hits = 0
            r = rows.setdefault(
                session,
                {"session": session, "searches": 0, "hits": 0},
            )
            r["searches"] += 1
            r["hits"] += hits
    return sorted(rows.values(), key=lambda r: r["searches"], reverse=True)


def _load_traces(mlflow_db: Path, project_name: str):
    """Return (traces_df, experiment_id_or_none).

    Reads the serverless ``sqlite:///<pdir>/mlflow.db`` store directly — no
    server required.  Separate from the main reporting logic so the caller
    can decide what to print when the experiment doesn't exist yet (new
    project, never run).
    """
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    exp = mlflow.get_experiment_by_name(project_name)
    if exp is None:
        return None, None
    traces = mlflow.search_traces(
        locations=[exp.experiment_id],
        max_results=5000,
    )
    return traces, exp.experiment_id


def _report(project_name: str, config: dict, traces) -> dict:
    """Build the structured report dict.  CLI formats it; --json prints it."""
    agent_header = config.get("agent", "-")
    # BYOA: dsagt no longer records the agent's LLM model (the agent talks to
    # its provider directly).  Surface the embedding model dsagt configures.
    model_header = config.get("embedding", {}).get("model", "-")

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
            "kb_retrieval": [],
        }

    # Extract flat columns we'll group on.  Using .apply over the metadata
    # column once up front keeps pandas from re-parsing the dict on every
    # groupby.
    md = traces["trace_metadata"].apply(lambda m: m or {})
    tags = (
        traces["tags"].apply(lambda t: t or {})
        if "tags" in traces.columns
        else md.apply(lambda _: {})
    )
    session = md.apply(lambda m: m.get("mlflow.trace.session") or "(no-session)")
    agent = md.apply(lambda m: m.get("dsagt.agent") or "-")
    errored = traces["state"].apply(_is_error)

    # Source comes from the metadata DSAGT stamps (dsagt.source tag /
    # dsagt.agent), no span inspection.  Tokens still walk the spans column as
    # a fallback because not all traces carry ``mlflow.trace.tokenUsage``.
    spans_col = traces["spans"] if "spans" in traces.columns else None

    def _row_source(idx: int) -> str:
        return _row_source_for(tags.iloc[idx], md.iloc[idx])

    def _row_tokens(idx: int) -> tuple[int, int]:
        # Trace metadata first (when tokenUsage is present), then aggregate
        # span attrs (native claude OTel shape).
        m = md.iloc[idx]
        i, o = _tokens(m)
        if i or o:
            return i, o
        if spans_col is not None:
            return _tokens_from_spans(spans_col.iloc[idx])
        return 0, 0

    indices = md.reset_index(drop=True).index.to_series()
    source = indices.apply(_row_source)
    source.index = md.index
    token_pairs = indices.apply(_row_tokens)
    token_pairs.index = md.index
    in_toks = token_pairs.apply(lambda p: p[0])
    out_toks = token_pairs.apply(lambda p: p[1])

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
                row["agent"] = (
                    g["_agent"].mode().iloc[0] if not g["_agent"].empty else "-"
                )
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
        errors.append(
            {
                "session": row["_session"],
                "source": row["_source"],
                "request_time": row["request_time"],
                "trace_id": row["trace_id"],
            }
        )

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
        "kb_retrieval": _kb_retrieval(traces),
    }


def _print_text(r: dict) -> None:
    print(f"Project: {r['project']}")
    print(f"  Agent:      {r['agent']}")
    print(f"  Embedding:  {r['model']}")
    if r.get("created"):
        print(f"  Started: {r['created']}")
    print()

    config_sources = r.get("config_sources") or []
    if config_sources:
        _print_config_sources(config_sources)

    _print_kb_collections(r.get("kb_collections") or [])
    _print_skills(r.get("skills") or [])

    if r["total_traces"] == 0:
        print("No traces recorded yet (run `dsagt start` to create a session).")
        return

    n_sessions = len(r["by_session"])
    print(
        f"Totals ({r['total_traces']} traces across {n_sessions} "
        f"session{'s' if n_sessions != 1 else ''}):"
    )
    print(
        f"  Tokens: {_fmt_count(r['input_tokens'])} in / "
        f"{_fmt_count(r['output_tokens'])} out"
    )
    print(f"  Errors: {r['total_errors']}")
    print()

    print("By source:")
    src_w = max((len(row["source"]) for row in r["by_source"]), default=12)
    for row in r["by_source"]:
        print(
            f"  {row['source']:<{src_w}} {row['traces']:>4} traces  "
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

    _print_kb_retrieval(r.get("kb_retrieval") or [])

    if r["errors"]:
        print()
        print("Errors:")
        for e in r["errors"]:
            print(f"  {e['session']:<36} {e['source']:<12} trace={e['trace_id']}")


def run(project: str, as_json: bool) -> int:
    # Resolve ${ENV_VAR} references so the header shows resolved values
    # (not ${VAR} placeholders from .dsagt/config.yaml).
    config = resolve_env_vars(load_config(project))
    pdir = Path(config["project_dir"])
    mlflow_db = pdir / "mlflow.db"

    sources = _config_sources(project)
    kb_collections = _kb_collections(pdir)
    skills = _skills(pdir)
    created = _project_created(pdir)

    if not mlflow_db.exists():
        # New project, or one that's never been started.  Print the header
        # so the user can verify they got the right project, then a short
        # note — rather than crashing on a missing DB.
        r = {
            "project": project,
            "agent": config.get("agent", "-"),
            "model": config.get("embedding", {}).get("model", "-"),
            "created": created,
            "total_traces": 0,
            "total_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "by_source": [],
            "by_session": [],
            "errors": [],
            "kb_retrieval": [],
            "kb_collections": kb_collections,
            "skills": skills,
            "config_sources": sources,
        }
        if as_json:
            print(json.dumps(r, indent=2, default=str))
        else:
            _print_text(r)
        return 0

    traces, _ = _load_traces(mlflow_db, project)
    r = _report(project, config, traces)
    r["created"] = created
    r["kb_collections"] = kb_collections
    r["skills"] = skills
    r["config_sources"] = sources

    if as_json:
        print(json.dumps(r, indent=2, default=str))
    else:
        _print_text(r)
    return 0
