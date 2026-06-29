"""
DSAgt observability — span emission via MLflow's native tracer provider.

Business modules (knowledge.py, provenance.py, mcp/registry_tools.py,
run_tool.py) import the small public surface defined here.  ``init_tracing`` installs
MLflow's own ``TracerProvider`` as the OTel global, so every
``trace.get_tracer(...)`` call routes spans into MLflow's native trace store
with full ``mlflow.spanInputs`` / ``mlflow.spanOutputs`` integration and
direct access to ``InMemoryTraceManager`` for trace-metadata stamping.

Public surface
--------------
init_tracing(service_name, *, mlflow_url=None, session_id=None)
    Configure MLflow's tracer provider once per process.  Reads
    ``./.dsagt/config.yaml`` for ``project`` and resolves the tracking URI
    serverlessly (``resolve_tracking_uri``: ``MLFLOW_TRACKING_URI`` env →
    config → default ``sqlite:///<pdir>/mlflow.db``).  Never raises — when cwd
    isn't a project dir it logs and no-ops, so one-shot tools / tests that
    aren't in a project simply run without tracing.

traced(span_name, *, capture=(), extract_return=None)
    Decorator. Opens a span around a function call, captures named arguments
    and (optionally) return-value-derived attributes, records exceptions, and
    sets duration_ms.

child_span(name, **attrs)
    Context manager for nested spans inside a ``@traced`` method.

obs
    Process-wide proxy for the *current* span.  ``obs.set("hits", 5)`` is a
    no-op when no span is active, so business code can annotate without
    branching on whether tracing is on.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

# Module-level state. _initialized guards against double-init in test runs
# and in subprocesses where init_tracing might be called more than once.
#
# NOTE on _default_session_id:
#   This is intentionally a process-global rather than threaded through
#   every span helper.  DSAgt runs one process per project, so the session
#   id is a process-wide constant set at startup by init_tracing(), and
#   propagating it implicitly via this module-level value lets business
#   code in knowledge.py / provenance.py / mcp/registry_tools.py emit spans
#   without ever knowing about session ids.  The cost is that tests have
#   to monkeypatch the global to isolate (see _reset_tracing fixture in
#   test_observability.py).
#
#   If DSAgt ever grows a multi-tenant mode (one server process serving
#   many projects), the right replacement is OTel baggage:
#       from opentelemetry import baggage
#       sid = baggage.get_baggage("session.id")
#   which gives per-request context propagation that works correctly
#   across concurrent requests.  Until then, the global is simpler.
_initialized = False
_tracer_provider = None
_default_session_id: str | None = None
# Strategy pointer bound at init_tracing time. Exists to keep
# backend-specific behavior out of call sites:
#
# _metadata_stamper(dict)
#   Write key/value metadata to the currently-active trace via MLflow's
#   InMemoryTraceManager (so it lands in trace_metadata, queryable in
#   the UI).
_metadata_stamper: "Callable[[dict], None] | None" = None


def init_tracing(
    service_name: str,
    mlflow_url: str | None = None,
    session_id: str | None = None,
) -> None:
    """Install MLflow's tracer provider as the OTel global.

    Every ``trace.get_tracer(...)`` call (from ``@traced`` / ``child_span``
    in MCP servers and dsagt-run) routes spans into MLflow's native trace
    store.  This gives ``mlflow.spanInputs`` / ``mlflow.spanOutputs``
    full integration and direct access to ``InMemoryTraceManager`` for
    trace-metadata stamping (session id, source, agent).

    Serverless: reads ``./.dsagt/config.yaml`` for ``project`` and resolves
    the tracking URI via :func:`resolve_tracking_uri` — defaulting to a
    ``sqlite:///<pdir>/mlflow.db`` store that needs no running server.  The session
    id is passed in by the caller (the MCP server mints it at startup via
    ``session.append_session``); ``None`` when not supplied.

    The ``mlflow_url`` and ``session_id`` keyword args are kept for tests,
    where the caller plants known values directly.

    Never raises.  When cwd isn't a dsagt project dir, logs and no-ops so
    one-shot tools / tests outside a project simply run untraced.
    """
    global _initialized, _default_session_id, _metadata_stamper

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

    _install_mlflow_provider(mlflow_url, project_name)
    _metadata_stamper = _stamp_metadata_mlflow
    _initialized = True
    atexit.register(_shutdown)
    logger.info(
        "init_tracing: service=%s mlflow=%s project=%s session=%s",
        service_name,
        mlflow_url,
        project_name,
        _default_session_id or "<none>",
    )


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
    """Resolve the MLflow tracking URI for DSAGT self-logging.  Never raises.

    Resolution order:
      1. ``MLFLOW_TRACKING_URI`` env — join the user's own store if they
         run one.
      2. ``mlflow.tracking_uri`` in the project config — an explicit
         override.
      3. Default ``sqlite:///<project_dir>/mlflow.db`` — a serverless
         SQLite store that always works with no listener running.

    The MLflow Python client honors a ``sqlite:`` tracking URI directly
    (auto-creating + migrating the DB on first use), so self-logging needs
    no server and the resolver never has to fail.  SQLite is MLflow's
    supported serverless backend — the filesystem store (``file:`` /
    ``./mlruns``) is deprecated as of Feb 2026.  DSAGT emits only traces
    (no runs/models), so the experiment's default artifact dir is never
    materialized.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    cfg = config or {}
    configured = (cfg.get("mlflow") or {}).get("tracking_uri")
    if configured:
        return configured
    pdir = cfg.get("project_dir")
    base = Path(pdir).resolve() if pdir else Path.cwd().resolve()
    return f"sqlite:///{base / 'mlflow.db'}"


def _install_mlflow_provider(mlflow_url: str, project_name: str) -> None:
    """Wire MLflow's tracer provider in as the OTel global.

    Force MLflow's lazy provider to initialize via its private init hook
    so we can hand the resulting TracerProvider to OTel below.  The
    underscore on the init function is MLflow-internal — pinning the
    mlflow version range in pyproject.toml keeps that boundary stable.
    """
    import mlflow
    from mlflow.tracing import provider as mp
    from opentelemetry import trace
    from opentelemetry.util._once import Once

    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(project_name)

    mp._initialize_tracer_provider()

    # OTel guards set_tracer_provider with a one-shot Once flag — the first
    # caller wins.  Reset so installation always takes effect even if
    # something accessed get_tracer earlier and locked in the no-op global.
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    trace.set_tracer_provider(mp.provider.get())


def _stamp_metadata_on_trace(request_id: str, metadata: dict) -> None:
    """Write metadata to a specific MLflow trace by id.

    Used when the caller has the trace_id in hand.  ``stamp_metadata`` is
    the higher-level version that looks up the current trace via the
    active OTel span.
    """
    try:
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        with InMemoryTraceManager.get_instance().get_trace(request_id) as t:
            if t is not None:
                t.info.trace_metadata.update({k: str(v) for k, v in metadata.items()})
    except Exception as e:
        logger.debug("metadata stamp failed for %s: %s", request_id, e)


def _stamp_metadata_mlflow(metadata: dict) -> None:
    """Write metadata to the currently-active trace's trace_metadata."""
    from opentelemetry import trace

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return
    _stamp_metadata_on_trace(f"tr-{ctx.trace_id:032x}", metadata)


def stamp_metadata(metadata: dict) -> None:
    """Stamp arbitrary key/value metadata on the currently-active trace.

    No-op when no backend is configured (tests, standalone tools).
    """
    if _metadata_stamper is not None and metadata:
        _metadata_stamper(metadata)


def _shutdown() -> None:
    """Flush and shut down the tracer provider. Registered via atexit."""
    global _tracer_provider
    if _tracer_provider is None:
        return
    try:
        _tracer_provider.shutdown()
    except Exception as e:
        logger.debug("tracer shutdown raised: %s", e)
    _tracer_provider = None


def get_tracer(name: str):
    """Return an OTel tracer routed through the OTLP provider installed by
    ``init_tracing``."""
    from opentelemetry import trace

    return trace.get_tracer(name)


@contextmanager
def open_span(name: str):
    """Open a span on the installed tracer provider.

    Single OTel code path — provider selection happens once at
    ``init_tracing`` time.  Yields ``None`` if tracing was never initialized
    (test paths that skip init_tracing monkeypatch module state directly).
    """
    if not _initialized:
        yield None
        return
    tracer = get_tracer("dsagt.observability")
    with tracer.start_as_current_span(name) as span:
        yield span


# ---------------------------------------------------------------------------
# obs — process-wide proxy for the current span
# ---------------------------------------------------------------------------


class _Obs:
    """No-op-safe proxy for the active span.

    Business code uses ``obs.set("hits", 5)`` instead of importing OTel. When
    no span is active (tracing disabled, or call site outside any traced
    block), every method silently does nothing.
    """

    def set(self, key: str, value: Any) -> None:
        span = self._current()
        if span is not None and value is not None:
            span.set_attribute(key, value)

    def set_many(self, attrs: Mapping[str, Any]) -> None:
        span = self._current()
        if span is None:
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)

    def event(self, name: str, **attrs: Any) -> None:
        span = self._current()
        if span is None:
            return
        clean_attrs = {k: v for k, v in attrs.items() if v is not None}
        span.add_event(name, attributes=clean_attrs)

    def set_inputs(self, inputs: Any) -> None:
        """Populate the trace's ``request`` field for the MLflow trace UI.

        Stamps ``mlflow.spanInputs`` as a JSON-serialized OTel attribute —
        this is the same key MLflow's ``LiveSpan.set_inputs`` writes, so it
        flows through to MLflow's ``request`` column in ``search_traces``.
        For non-MLflow OTel backends (Jaeger, Tempo) it just shows up as a
        regular span attribute, harmlessly.
        """
        span = self._current()
        if span is None or inputs is None:
            return
        span.set_attribute("mlflow.spanInputs", _to_json(inputs))

    def set_outputs(self, outputs: Any) -> None:
        """Populate the trace's ``response`` field for the MLflow trace UI."""
        span = self._current()
        if span is None or outputs is None:
            return
        span.set_attribute("mlflow.spanOutputs", _to_json(outputs))

    @staticmethod
    def _current():
        """Return the currently-active OTel span, or ``None`` if none."""
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.get_span_context().is_valid:
            return None
        return span


obs = _Obs()


# ---------------------------------------------------------------------------
# traced — decorator for top-level method instrumentation
# ---------------------------------------------------------------------------


def traced(
    span_name: str,
    *,
    capture: Iterable[str] = (),
    extract_return: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Callable:
    """Wrap a function in an OTel span.

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
        value to extract that attribute. Use this for things like
        ``{"hits": lambda r: len(r)}``.

    Behavior
    --------
    * Captures the configured args as attributes (skipping ``None``).
    * Always sets ``duration_ms``.
    * Always sets the DSAgt session id (if init_tracing was given one).
    * Records exceptions via ``span.record_exception`` and sets ERROR status.
    * Re-raises after recording.
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
                _attach_trace_metadata(span_name)

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
                except Exception as exc:
                    span.record_exception(exc)
                    _set_error_status(span, str(exc))
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


_SOURCE_BY_PREFIX = (
    ("tool.", "tool"),
    ("kb.", "knowledge"),
    ("registry.", "registry"),
)


def _derive_source(span_name: str | None) -> str | None:
    """Map a span name to a ``dsagt.source`` value.

    Prefix-based dispatch so every span we open lands with a source tag
    without touching the call site.  Covers the non-LLM spans our own
    instrumentation emits (kb / registry / tool).
    """
    if not span_name:
        return None
    for prefix, source in _SOURCE_BY_PREFIX:
        if span_name.startswith(prefix):
            return source
    return None


def _attach_trace_metadata(span_name: str | None) -> None:
    """Stamp session + source on the currently-active trace.

    - ``mlflow.trace.session``: process-wide session id (reserved MLflow
      key; powers the UI's native session filter).
    - ``dsagt.source``: derived from the span name prefix — so every span
      we open lands in ``dsagt info``'s "by source" bucket.

    No-op when no backend is configured or no span is active; the
    metadata stamper handles both cases.
    """
    md: dict[str, str] = {}
    if _default_session_id:
        md["mlflow.trace.session"] = _default_session_id
    src = _derive_source(span_name)
    if src:
        md["dsagt.source"] = src
    if md:
        stamp_metadata(md)


def _to_json(value: Any) -> str:
    """Serialize *value* to JSON for an OTel attribute.

    OTel attributes only accept primitives + sequences of primitives;
    MLflow's ``mlflow.spanInputs`` / ``mlflow.spanOutputs`` keys expect
    a JSON-encoded payload that the trace UI deserializes for display.
    """
    import json

    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _coerce_attr(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_attr(v) for v in value]
    return str(value)


def _set_error_status(span, message: str) -> None:
    # OTel is mandatory; if Status/StatusCode disappears or moves, we want
    # the resulting ImportError to surface immediately so the dep upgrade
    # gets caught at the test level rather than silently disabling error
    # status on every traced exception path.
    from opentelemetry.trace import Status, StatusCode

    span.set_status(Status(StatusCode.ERROR, message))


# ---------------------------------------------------------------------------
# Typed span helpers — populated incrementally as later stages need them.
#
# The convention is: every span name DSAgt emits has a helper here. Business
# modules call the helper, never tracer.start_as_current_span directly. This
# keeps span names, attribute schemas, and required fields in one file.
# ---------------------------------------------------------------------------


@contextmanager
def child_span(name: str, **attrs: Any):
    """Open a child span with arbitrary attributes.

    Use this from inside a ``@traced`` method when you need to break a method
    into sub-phases (e.g. embed / index_search / rerank inside kb.search).
    Prefer the typed helpers below when one exists for your operation.
    """
    with open_span(name) as span:
        if span is None:
            yield None
            return
        _attach_trace_metadata(name)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _coerce_attr(v))
        yield span


# ----- Knowledge base spans (Stage 1) -----


def kb_embed_span(backend: str | None, model: str | None, n_texts: int):
    """Span around an embedding call.

    Used for both query embedding (kb.search) and chunk embedding (kb.ingest,
    kb.append, kb.add_entries).  Backend-agnostic: ``backend`` is ``"api"``
    for the HTTP embedder or ``"local"`` for sentence-transformers.
    """
    return child_span(
        "kb.embed",
        backend=backend,
        model=model,
        n_texts=n_texts,
    )


def kb_index_search_span(vector_db: str | None, k: int, filtered: bool):
    """Span around an underlying vector index search call."""
    return child_span(
        "kb.index_search",
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


# ----- Registry server spans (Stage 4) -----
#
# Only the deliberate, infrequent registry operations are instrumented.
# search_registry / search_skills are intentionally NOT instrumented — they
# are high-frequency low-information per call, and the agent-side LLM trace
# already records that they were invoked.


def registry_save_tool_span(tool_name: str | None):
    """Span around ``save_tool_spec``."""
    return child_span("registry.save_tool_spec", tool_name=tool_name)


def registry_install_deps_span(packages: list[str] | None):
    """Span around an ``install_dependencies`` call."""
    return child_span(
        "registry.install_dependencies",
        package_count=len(packages) if packages else 0,
        # First few package names are useful in the UI for at-a-glance
        # identification; full list is in the LLM call record if needed.
        packages_preview=", ".join(packages[:5]) if packages else "",
    )


def registry_reconstruct_pipeline_span(fmt: str | None):
    """Span around a ``reconstruct_pipeline`` call."""
    return child_span("registry.reconstruct_pipeline", format=fmt or "bash")


# ----- Tool execution spans (Stage 3) -----


def tool_execute_span(record_id: str, tool_name: str):
    """Span around a single ``dsagt-run`` tool execution.

    This is a *top-level* span, not a child of any LLM call span.  The agent
    CLI (Claude Code, Goose, etc.) spawns ``dsagt-run`` in its own process
    tree without OTel context propagation, so the LLM-call-to-tool-execution
    parent/child linkage cannot be expressed in the trace tree directly.
    Instead, every ``tool.execute`` span carries:

    * ``session.id`` (attached automatically via init_tracing's session_id)
    * ``record_id`` — correlates the execution to its ``trace_archive`` record.
    * ``tool_name`` — for filtering in the MLflow UI.

    Filter by ``session.id`` in the MLflow trace view to see all tool
    executions for a given project alongside the LLM calls that requested
    them.  Cross-reference via ``record_id`` for full intent → execution
    linkage when needed.
    """

    @contextmanager
    def _wrapper():
        with open_span("tool.execute") as span:
            if span is None:
                yield None
                return
            _attach_trace_metadata("tool.execute")
            span.set_attribute("record_id", record_id)
            span.set_attribute("tool_name", tool_name)
            yield span

    return _wrapper()


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


# ---------------------------------------------------------------------------
# Agent-trace sink — CanonicalTrace → MLflow spans (the Phase-2 foreign feed)
# ---------------------------------------------------------------------------

_S_PER_NS = 1e9


def _to_ns(epoch_s: float | None) -> int | None:
    return int(epoch_s * _S_PER_NS) if epoch_s is not None else None


class MLflowSink:
    """Render a :class:`~dsagt.traces.Trace` into MLflow spans (a trace consumer).

    The agent (foreign-trace) half of observability: where ``init_tracing`` +
    ``@traced`` emit DSAGT's own first-party spans live, this replays a finished
    transcript's :class:`~dsagt.traces.Trace` after the fact.  It uses
    ``mlflow.start_span_no_context`` — the only API that accepts an explicit
    ``parent_span`` and backdated ``start_time_ns`` — and mirrors the span
    conventions of MLflow's own ``claude_code`` autolog so foreign traces render
    identically in the Chat UI: an AGENT root, ``llm`` children carrying
    ``message.format="anthropic"`` + ``mlflow.chat.tokenUsage``, and
    ``tool_<name>`` children.

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
