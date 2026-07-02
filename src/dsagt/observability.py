"""
DSAgt observability — first-party span emission over the serverless MLflow store.

DSAGT writes every trace to one ``sqlite:///<pdir>/mlflow.db`` store (no server).
TWO emission paths share that one store; they differ because MLflow's API forces
it, not by accident:

  * LIVE tracer (``mlflow.start_span``) — first-party DSAGT spans emitted *as the
    MCP server / dsagt-run runs*.  Uses MLflow's active-span context for
    auto-nesting and the ``obs`` proxy.  Each trace's root is tagged
    ``dsagt.source`` with the MCP tool *category* the agent invoked
    (memory / skill / knowledge / registry), or ``execution`` for dsagt-run —
    set at the dispatch boundary, so the UI can filter this *debugging* view
    apart from agent traces and bucket it by concern.
  * REPLAY sink (``MLflowSink`` → ``mlflow.start_span_no_context``) — a finished
    agent ``traces.Trace`` backfilled after the fact with the transcript's
    original timestamps.  ``start_span`` cannot backdate (it has no
    ``start_time_ns`` param), so replay *must* use ``start_span_no_context``;
    live *should* use ``start_span`` (no_context establishes no active span,
    which would kill the ``obs`` proxy and auto-nesting).  Hence two paths, one
    store — neither is a historical leftover.

Layout (top → bottom)
---------------------
  setup        find_project_config · resolve_tracking_uri · init_tracing
  live tracer  open_span ─┬─ traced       (decorate a function)
                          ├─ child_span   (open a sub-span)
                          └─ obs          (annotate the span you're inside)
               tagging:   open_span(source=…) → _attach_trace_metadata
                          (dsagt.source set on the trace's root only)
               factories: kb_* · registry_* · code_execute_span
  replay sink  MLflowSink  (Trace → backdated spans; a traces.TraceCollector consumer)

``traced`` and ``child_span`` *open* a span; ``obs`` *annotates* whichever span
is currently open.  All three no-op when tracing was never initialized, so
business code never imports MLflow and never branches on whether tracing is on.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

# Module-level state. _initialized guards every span helper: without a tracking
# store + experiment set (init_tracing), mlflow.start_span would write to
# MLflow's default location, so the helpers no-op until init_tracing succeeds.
# DSAGT runs one process per project, so the session id is a process-wide
# constant set once at startup and tagged onto each internal trace for grouping.
_initialized = False
_default_session_id: str | None = None


# ===========================================================================
# Setup — where am I, where do I write, wire MLflow up
# ===========================================================================


def find_project_config() -> tuple[Path | None, dict | None]:
    """Read ``./.dsagt/config.yaml`` from cwd.

    Returns ``(cwd, parsed_config)`` or ``(None, None)`` if cwd isn't a
    project directory.  No walking — services that need this info run
    with cwd == project_dir by contract; if cwd is anywhere else the
    caller is misconfigured and we fail fast.

    Project name → project_dir for arbitrary-cwd lookups (e.g., the
    user CLI typing ``dsagt info <name>``) is the registry's job
    (``~/dsagt-projects/projects.yaml``, see ``session.project_dir``).  This
    helper is only for services running inside the project.
    """
    cwd = Path.cwd().resolve()
    candidate = cwd / ".dsagt" / "config.yaml"
    if not candidate.exists():
        return None, None
    try:
        import yaml

        return cwd, yaml.safe_load(candidate.read_text()) or {}
    except Exception as e:
        logger.debug("could not parse %s: %s", candidate, e)
        return cwd, None


def resolve_tracking_uri(config: dict | None) -> str:
    """Compute the serverless MLflow tracking URI for DSAGT self-logging.

    Always ``sqlite:///<project_dir>/mlflow.db`` — DSAGT knows where it writes;
    there is no server to point at.  ``project_dir`` comes from the resolved
    config (injected by ``session.load_config``), falling back to cwd for
    in-project callers.

    The MLflow client honors a ``sqlite:`` URI directly (auto-creating +
    migrating the DB on first use), so self-logging needs no listener and this
    never has to fail.  SQLite is MLflow's supported serverless backend — the
    filesystem store (``file:`` / ``./mlruns``) is deprecated as of Feb 2026.
    DSAGT emits only traces (no runs/models), so the experiment's default
    artifact dir is never materialized.
    """
    cfg = config or {}
    pdir = cfg.get("project_dir")
    base = Path(pdir).resolve() if pdir else Path.cwd().resolve()
    return f"sqlite:///{base / 'mlflow.db'}"


def init_tracing(
    service_name: str,
    mlflow_url: str | None = None,
    session_id: str | None = None,
) -> None:
    """Point MLflow at the project's serverless store + experiment.

    After this, every ``mlflow.start_span`` (from ``@traced`` / ``child_span``
    in the MCP server and dsagt-run) lands in ``sqlite:///<pdir>/mlflow.db``.
    The tracking URI is the serverless ``sqlite:///<pdir>/mlflow.db`` computed by
    :func:`resolve_tracking_uri`.  The session id, passed by the MCP server at
    startup, tags internal traces for grouping.

    The ``mlflow_url`` and ``session_id`` keyword args are kept for tests, where
    the caller plants known values directly.

    Never raises.  When cwd isn't a dsagt project dir, logs and no-ops so
    one-shot tools / tests outside a project simply run untraced.
    """
    global _initialized, _default_session_id

    if _initialized:
        if session_id:
            _default_session_id = session_id
        return

    cfg_pdir, cfg = find_project_config()
    project_name = (cfg or {}).get("project")
    if not project_name:
        logger.warning(
            "%s: no .dsagt/config.yaml with a 'project' in cwd (%s) — tracing "
            "disabled for this process.",
            service_name,
            Path.cwd(),
        )
        return

    _default_session_id = session_id
    if mlflow_url is None:
        resolve_cfg = dict(cfg)
        resolve_cfg["project_dir"] = str(cfg_pdir)
        mlflow_url = resolve_tracking_uri(resolve_cfg)

    import mlflow

    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(project_name)
    _initialized = True
    logger.info(
        "init_tracing: service=%s mlflow=%s project=%s session=%s",
        service_name,
        mlflow_url,
        project_name,
        _default_session_id or "<none>",
    )


# ===========================================================================
# Live tracer — first-party debug spans, emitted as code runs
# ===========================================================================

# ----- the span primitive -----


@contextmanager
def open_span(name: str, span_type: str | None = None, source: str | None = None):
    """Open a span on the serverless MLflow store.

    Single tracing code path — ``mlflow.start_span`` auto-nests under whatever
    span is already active (its own context model), so child spans Just Work.
    Yields ``None`` when tracing was never initialized (one-shot tools / tests
    outside a project), so the helpers below degrade to no-ops.

    ``source`` is set only on the *categorization root* of a trace — the MCP
    dispatch span and ``tool.execute`` — and tags the whole trace
    ``dsagt.source`` for the debug-view filter (see :func:`_attach_trace_metadata`).
    Inner spans pass ``source=None`` and inherit the root's tag.
    """
    if not _initialized:
        yield None
        return
    import mlflow
    from mlflow.entities import SpanType

    with mlflow.start_span(name=name, span_type=span_type or SpanType.UNKNOWN) as span:
        _attach_trace_metadata(source)
        yield span


# ----- attribute-value helpers -----


def _coerce_attr(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_attr(v) for v in value]
    return str(value)


def truncate(value: str, limit: int = 256) -> str:
    """Truncate a string for span attributes.

    Span backends (and the MLflow trace UI) handle short attribute values
    much better than multi-megabyte stdout/stderr blobs.  The full payload
    lives in ``trace_archive/<record_id>.json``; the span just carries a
    head/tail summary so a human glancing at the UI can tell what happened.
    """
    if value is None:
        return ""
    if len(value) <= limit:
        return value
    head = limit - 32
    return value[:head] + f"... [+{len(value) - head} chars]"


# ----- trace tagging — the dsagt.source debug filter + session grouping -----
#
# dsagt.source names the MCP tool *category* that was invoked — one of
# {memory, skill, knowledge, registry} (the four concern modules of the merged
# dsagt-server), plus ``execution`` for dsagt-run's out-of-process data-tool
# runs.  It is assigned at the *entry point*, not derived from the span name:
# the MCP dispatch shell knows which concern owns each tool and stamps the
# category on the trace's root span, so e.g. ``search_skills`` calling into
# ``kb.search`` is tagged ``skill`` (the tool the agent called), not
# ``knowledge`` (the subsystem that happened to do the work).  Inner spans
# inherit the root's tag; agent traces carry no ``dsagt.source`` at all, which
# is what lets the MLflow UI filter the debug view in or out.


def _attach_trace_metadata(source: str | None) -> None:
    """Tag the current trace as a DSAGT-internal (debug) trace.

    Called from :func:`open_span` on a categorization root (``source`` set):

    - ``dsagt.source`` tag: the MCP tool category / ``execution`` — powers the
      UI's debug-view filter.
    - ``mlflow.trace.session`` metadata: groups this process's internal traces
      under one session (reserved MLflow key, drives the native session filter).

    No-op for inner spans (``source is None``) — they inherit the root's tag.
    """
    if not source:
        return
    metadata = (
        {"mlflow.trace.session": _default_session_id} if _default_session_id else None
    )
    import mlflow

    mlflow.update_current_trace(tags={"dsagt.source": source}, metadata=metadata)


# ----- annotate the active span -----


class _ActiveSpanProxy:
    """The *annotate* verb that complements the *open* verbs (traced/child_span).

    Where ``traced`` / ``child_span`` open a span, this annotates whichever span
    is currently open — ``obs.set("hits", 5)`` from inside a ``@traced`` body
    attaches to that body's span.  It exists so business code (knowledge.py,
    provenance.py, registry_tools.py) never imports MLflow and never branches on
    whether tracing is on: when no span is active (tracing disabled, or call
    site outside any traced block) every method silently does nothing.

    The process-wide singleton is exported as ``obs``.
    """

    def set(self, key: str, value: Any) -> None:
        span = self._current()
        if span is not None and value is not None:
            span.set_attribute(key, _coerce_attr(value))

    def set_many(self, attrs: Mapping[str, Any]) -> None:
        span = self._current()
        if span is None:
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _coerce_attr(v))

    def event(self, name: str, **attrs: Any) -> None:
        span = self._current()
        if span is None:
            return
        from mlflow.entities import SpanEvent

        clean_attrs = {k: v for k, v in attrs.items() if v is not None}
        span.add_event(SpanEvent(name, attributes=clean_attrs))

    def set_inputs(self, inputs: Any) -> None:
        """Populate the trace's ``request`` field for the MLflow trace UI."""
        span = self._current()
        if span is None or inputs is None:
            return
        span.set_inputs(inputs)

    def set_outputs(self, outputs: Any) -> None:
        """Populate the trace's ``response`` field for the MLflow trace UI."""
        span = self._current()
        if span is None or outputs is None:
            return
        span.set_outputs(outputs)

    @staticmethod
    def _current():
        """Return the currently-active MLflow span, or ``None`` if none."""
        if not _initialized:
            return None
        import mlflow

        return mlflow.get_current_active_span()


obs = _ActiveSpanProxy()


# ----- open a span — decorator + context manager -----


def traced(
    span_name: str,
    *,
    capture: Iterable[str] = (),
    extract_return: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Callable:
    """Wrap a function in an MLflow span.

    Parameters
    ----------
    span_name
        Span name. Should be ``"<service>.<operation>"`` (e.g. ``"kb.search"``).
    capture
        Names of arguments to copy into span attributes. Looked up by name
        against the function signature, so positional and keyword args both
        work.
    extract_return
        Optional mapping of attribute name → function applied to the return
        value to extract that attribute (e.g. ``{"hits": lambda r: len(r)}``).

    Behavior
    --------
    * Captures the configured args as attributes (skipping ``None``).
    * Always sets ``duration_ms``.
    * Tags the trace ``dsagt.source`` + session for the debug view.
    * Records exceptions and sets ERROR status (MLflow auto-records the
      exception on context exit); re-raises after recording.
    """
    capture = tuple(capture)
    extract_return = dict(extract_return or {})

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with open_span(span_name) as span:
                if span is None:
                    return fn(*args, **kwargs)

                # Capture configured arguments as span attributes.  Looked up
                # by name against the function signature so positional and
                # keyword args both work.  TypeError on bind_partial means a
                # decorator higher in the stack mangled the signature; in
                # that rare case we just skip arg capture rather than crash.
                if capture:
                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                    except TypeError:
                        bound = None
                    if bound is not None:
                        bound.apply_defaults()
                        for name in capture:
                            if name in bound.arguments:
                                value = bound.arguments[name]
                                if value is not None:
                                    span.set_attribute(name, _coerce_attr(value))

                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    span.set_status("ERROR")
                    raise
                finally:
                    span.set_attribute(
                        "duration_ms", round((time.perf_counter() - start) * 1000, 3)
                    )

                # Extract return-value-derived attributes.  A buggy extractor
                # lambda must NOT crash the instrumented function, but should
                # be visible at DEBUG so a developer running with --verbose
                # can spot a silently-missing span attribute caused by a
                # broken extract_return mapping.
                for attr_name, extractor in extract_return.items():
                    try:
                        value = extractor(result)
                    except Exception as e:
                        logger.debug(
                            "extract_return[%r] failed (%s: %s); "
                            "attribute will be missing from span",
                            attr_name,
                            type(e).__name__,
                            e,
                        )
                        continue
                    if value is not None:
                        span.set_attribute(attr_name, _coerce_attr(value))

                return result

        return wrapper

    return decorator


@contextmanager
def child_span(name: str, *, span_type: str | None = None, **attrs: Any):
    """Open a child span with arbitrary attributes.

    Use this from inside a ``@traced`` method to break a method into sub-phases
    (e.g. embed / index_search / rerank inside kb.search).  Prefer the typed
    factories below when one exists for your operation.
    """
    with open_span(name, span_type=span_type) as span:
        if span is None:
            yield None
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _coerce_attr(v))
        yield span


# ----- named span factories -----
#
# Every span name DSAgt emits has a factory here. Business modules call the
# factory, never mlflow.start_span directly, so span names, attribute schemas,
# and span types stay in one file.

# Knowledge base spans.


def kb_embed_span(backend: str | None, model: str | None, n_texts: int):
    """Span around an embedding call.

    Used for both query embedding (kb.search) and chunk embedding (kb.ingest,
    kb.append, kb.add_entries).  Backend-agnostic: ``backend`` is ``"api"``
    for the HTTP embedder or ``"local"`` for sentence-transformers.
    """
    from mlflow.entities import SpanType

    return child_span(
        "kb.embed",
        span_type=SpanType.EMBEDDING,
        backend=backend,
        model=model,
        n_texts=n_texts,
    )


def kb_index_search_span(vector_db: str | None, k: int, filtered: bool):
    """Span around an underlying vector index search call."""
    from mlflow.entities import SpanType

    return child_span(
        "kb.index_search",
        span_type=SpanType.RETRIEVER,
        vector_db=vector_db,
        k=k,
        filtered=filtered,
    )


def kb_rerank_span(model: str | None, n_pairs: int):
    """Span around the cross-encoder rerank pass."""
    return child_span(
        "kb.rerank",
        model=model,
        n_pairs=n_pairs,
    )


# Registry spans.  Only the deliberate, infrequent registry operations are
# instrumented.  search_registry / search_skills are intentionally NOT — they
# are high-frequency low-information per call, and the agent-side LLM trace
# already records that they were invoked.


def registry_save_code_span(code_name: str | None):
    """Span around ``save_code_spec``."""
    from mlflow.entities import SpanType

    return child_span(
        "registry.save_code_spec", span_type=SpanType.TOOL, code_name=code_name
    )


def registry_install_deps_span(packages: list[str] | None):
    """Span around an ``install_dependencies`` call."""
    from mlflow.entities import SpanType

    return child_span(
        "registry.install_dependencies",
        span_type=SpanType.TOOL,
        package_count=len(packages) if packages else 0,
        # First few package names are useful in the UI for at-a-glance
        # identification; full list is in the LLM call record if needed.
        packages_preview=", ".join(packages[:5]) if packages else "",
    )


def registry_reconstruct_pipeline_span(fmt: str | None):
    """Span around a ``reconstruct_pipeline`` call."""
    from mlflow.entities import SpanType

    return child_span(
        "registry.reconstruct_pipeline", span_type=SpanType.TOOL, format=fmt or "bash"
    )


# Code execution spans (dsagt-run).


def code_execute_span(record_id: str, code_name: str):
    """Span around a single ``dsagt-run`` code execution.

    A *top-level, categorization-root* span: the agent CLI spawns ``dsagt-run``
    in its own process tree, so this trace stands alone in the store rather than
    nesting under any MCP-dispatch span.  It carries ``record_id`` (correlates
    to the ``trace_archive`` record) and ``code_name``, and is tagged
    ``dsagt.source=execution`` — its own bucket, distinct from the four MCP tool
    categories, since these are actual code runs rather than meta-ops.
    """
    from mlflow.entities import SpanType

    @contextmanager
    def _wrapper():
        with open_span(
            "code.execute", span_type=SpanType.TOOL, source="execution"
        ) as span:
            if span is None:
                yield None
                return
            span.set_attribute("record_id", record_id)
            span.set_attribute("code_name", code_name)
            yield span

    return _wrapper()


# ===========================================================================
# Replay sink — finished agent Trace → backdated MLflow spans
# ===========================================================================

_S_PER_NS = 1e9


def _to_ns(epoch_s: float | None) -> int | None:
    return int(epoch_s * _S_PER_NS) if epoch_s is not None else None


class MLflowSink:
    """Render a :class:`~dsagt.traces.Trace` into MLflow spans (a trace consumer).

    The agent half of observability: where ``@traced`` / ``obs`` emit DSAGT's
    own first-party debug spans live, this replays a finished transcript's
    :class:`~dsagt.traces.Trace` after the fact into the *same* store.  It uses
    ``mlflow.start_span_no_context`` — the only API that accepts an explicit
    ``parent_span`` and backdated ``start_time_ns`` — and mirrors the span
    conventions of MLflow's own ``claude_code`` autolog so foreign traces render
    identically in the Chat UI: an AGENT root, ``llm`` children carrying
    ``message.format="anthropic"`` + ``mlflow.chat.tokenUsage``, and
    ``tool_<name>`` children.  Agent traces carry no ``dsagt.source`` tag, so
    they stay in the normal view, separate from the internal debug traces.

    A session ``Trace`` carries one AGENT subtree per turn; the sink emits **one
    MLflow trace per AGENT root**, matching the per-prompt granularity autolog's
    Stop hook produces.  MLflow mints its own trace/span ids, so each trace is
    tagged ``dsagt.trace_id = <trace_id>:<root span_id>`` (a stable per-turn
    idempotency key).

    A *consumer* of :class:`~dsagt.traces.TraceCollector`: ``name`` keys its own
    ack file (``.dsagt/trace_acks_mlflow.json``); ``write`` logs the trace.
    Spans are plain dicts (see ``traces`` module docstring), so this reads them
    directly — no per-object serialization.
    """

    name = "mlflow"

    def __init__(self, tracking_uri: str, experiment: str):
        self._uri = tracking_uri
        self._experiment = experiment

    def write(self, trace) -> list[str]:
        """Log every turn subtree; return the MLflow trace id of each."""
        import mlflow

        mlflow.set_tracking_uri(self._uri)
        mlflow.set_experiment(trace.project or self._experiment)

        children: dict[str, list] = {}
        for span in trace.spans:
            if span["parent_id"] is not None:
                children.setdefault(span["parent_id"], []).append(span)

        trace_ids = []
        for root in trace.spans:
            if root["parent_id"] is None:
                trace_ids.append(
                    self._emit_subtree(root, children.get(root["span_id"], []), trace)
                )
        return trace_ids

    def _emit_subtree(self, root, children, trace) -> str:
        """Emit one MLflow trace for an AGENT ``root`` and its direct children."""
        import mlflow
        from mlflow.entities import SpanType
        from mlflow.tracing.constant import (
            SpanAttributeKey,
            TokenUsageKey,
            TraceMetadataKey,
        )
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        kind_to_type = {
            "AGENT": SpanType.AGENT,
            "LLM": SpanType.LLM,
            "TOOL": SpanType.TOOL,
            "OTHER": SpanType.UNKNOWN,
        }

        ml_root = mlflow.start_span_no_context(
            name=root["name"],
            span_type=kind_to_type[root["kind"]],
            inputs={"prompt": root["attributes"].get("prompt", "")},
            start_time_ns=_to_ns(root["start_time"]),
        )

        for span in children:
            if span["kind"] == "LLM":
                child = mlflow.start_span_no_context(
                    name=span["name"],
                    parent_span=ml_root,
                    span_type=SpanType.LLM,
                    start_time_ns=_to_ns(span["start_time"]),
                    inputs={
                        "model": span["model"] or "unknown",
                        "messages": span["request"],
                    },
                    attributes={
                        "model": span["model"] or "unknown",
                        SpanAttributeKey.MESSAGE_FORMAT: "anthropic",
                    },
                )
                if span["usage"]:
                    inp = span["usage"].get("input_tokens") or 0
                    out = span["usage"].get("output_tokens") or 0
                    child.set_attribute(
                        SpanAttributeKey.CHAT_USAGE,
                        {
                            TokenUsageKey.INPUT_TOKENS: inp,
                            TokenUsageKey.OUTPUT_TOKENS: out,
                            TokenUsageKey.TOTAL_TOKENS: inp + out,
                        },
                    )
                child.set_outputs(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": span["response"],
                    }
                )
            else:  # TOOL / OTHER
                child = mlflow.start_span_no_context(
                    name=span["name"],
                    parent_span=ml_root,
                    span_type=kind_to_type[span["kind"]],
                    start_time_ns=_to_ns(span["start_time"]),
                    inputs=span["attributes"].get("input", {}),
                    attributes={
                        "tool_name": span["attributes"].get("tool_name"),
                        "tool_id": span["attributes"].get("tool_id"),
                    },
                )
                child.set_outputs({"result": span["attributes"].get("result", "")})
            child.end(end_time_ns=_to_ns(span["end_time"]))

        # Trace-level metadata: session correlation + the per-turn canonical id
        # (idempotency key) + request/response previews for the trace list.
        try:
            mgr = InMemoryTraceManager.get_instance()
            with mgr.get_trace(ml_root.trace_id) as in_mem:
                meta = {
                    TraceMetadataKey.TRACE_SESSION: trace.session_id,
                    "dsagt.trace_id": f"{trace.trace_id}:{root['span_id']}",
                    "dsagt.agent": trace.agent,
                }
                in_mem.info.trace_metadata = {**in_mem.info.trace_metadata, **meta}
                if prompt := root["attributes"].get("prompt"):
                    in_mem.info.request_preview = str(prompt)[:1000]
                if response := root["attributes"].get("response"):
                    in_mem.info.response_preview = str(response)[:1000]
        except Exception as e:  # noqa: BLE001
            logger.warning("MLflowSink: could not stamp trace metadata: %s", e)

        ml_root.set_outputs({"response": root["attributes"].get("response", "")})
        ml_root.end(end_time_ns=_to_ns(root["end_time"]))
        return ml_root.trace_id
