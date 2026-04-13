"""
DSAgt observability — OpenTelemetry tracing wired to MLflow.

This module is the *only* place in DSAgt that imports OpenTelemetry. Business
modules (knowledge.py, provenance.py, registry_server.py, run_tool.py) import
the small public surface defined here and remain otel-agnostic.

Public surface
--------------
init_tracing(service_name, otel_endpoint=None, session_id=None)
    Configure the SDK once per process. No-op if no endpoint is set, so unit
    tests and offline use are unaffected.

traced(span_name, *, capture=(), extract_return=None)
    Decorator. Opens a span around a function call, captures named arguments
    and (optionally) return-value-derived attributes, records exceptions, and
    sets duration_ms.

obs
    Process-wide proxy for the *current* span. ``obs.set("hits", 5)`` is a
    no-op when no span is active, so business code can annotate without
    branching on whether tracing is on.

The typed span helpers (kb_search_span, kb_embed_span, etc.) live in this
module too — they will be added in Stage 1 alongside the call sites that use
them.

Design notes
------------
* The OTLP endpoint is the same one MLflow's tracing server exposes for the
  LiteLLM proxy. We don't run a separate collector.
* OTel exporters are registered with an ``atexit`` shutdown hook so that
  short-lived commands (e.g. ``dsagt-setup-kb``) flush spans before exit.
* If init_tracing has never been called, ``get_tracer`` returns the OTel
  no-op tracer and every helper short-circuits cleanly.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import logging
import os
import time
from contextlib import contextmanager
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
#   code in knowledge.py / provenance.py / registry_server.py emit spans
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


def init_tracing(
    service_name: str,
    otel_endpoint: str | None = None,
    session_id: str | None = None,
) -> None:
    """Configure the OTel SDK for this process.

    Parameters
    ----------
    service_name
        Name reported in span ``service.name``. Pick something meaningful per
        process: ``dsagt-knowledge-server``, ``dsagt-run``, etc.
    otel_endpoint
        OTLP HTTP base URL (e.g. ``http://localhost:5001``). Spans are POSTed
        to ``<endpoint>/v1/traces``. Falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT``
        if not provided. If neither is set, tracing is silently disabled.
    session_id
        Optional DSAgt session id. Attached as the ``session.id`` attribute to
        every span emitted from this process so siblings across MCP servers
        can be correlated even without true hierarchical traces.
    """
    global _initialized, _tracer_provider, _default_session_id

    if _initialized:
        # Allow updating session id without re-initializing the SDK.
        if session_id:
            _default_session_id = session_id
        return

    endpoint = otel_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    _default_session_id = session_id

    if not endpoint:
        # No endpoint, no tracing. The OTel API still works (no-op tracer),
        # so business code never has to branch on this.
        _initialized = True
        logger.debug("init_tracing: no endpoint configured, tracing disabled")
        return

    # OpenTelemetry is a hard dependency in pyproject.toml.  If these
    # imports fail, the install is broken and we want a real ImportError
    # immediately, not a silently-disabled tracing layer.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer_provider = provider
    _initialized = True

    # Flush spans on process exit. Short-lived commands like dsagt-setup-kb
    # otherwise lose their final batch.
    atexit.register(_shutdown)

    logger.info(
        "init_tracing: service=%s endpoint=%s session=%s",
        service_name, endpoint, session_id or "<none>",
    )


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
    """Return an OTel tracer. No-op tracer if init_tracing was not called."""
    from opentelemetry import trace
    return trace.get_tracer(name)


def configure_litellm_retries(
    num_retries: int = 5,
    request_timeout: float = 300.0,
) -> None:
    """Configure LiteLLM module-level retry / timeout knobs and quiet its
    stdout chatter.

    These are independent of tracing — even ``dsagt-setup-kb`` running before
    any project exists (and therefore with no MLflow endpoint) gets the
    rate-limit-resilient embedding path.

    LiteLLM's ``num_retries`` retries on transient failures including 429s
    with exponential backoff.  We also retry inside APIEmbeddingClient with
    finer control, but the litellm-level setting acts as a backstop for
    other code paths (e.g. the proxy).

    LiteLLM is also chatty by default: every error prints a "Give Feedback /
    Get Help" footer to stdout, and every successful call duplicates a log
    line through the ``LiteLLM`` logger.  Both make ``dsagt-setup-kb`` output
    nearly unreadable during long ingests with retries.  We suppress them
    here so DSAgt's own progress logs are visible.
    """
    # litellm[proxy] is a hard dependency in pyproject.toml — failing this
    # import means a broken install, not "tracing optional".
    import litellm

    litellm.num_retries = num_retries
    litellm.request_timeout = request_timeout

    # Kill the "Give Feedback / Get Help" + "Provider List" stdout footers
    # that LiteLLM prints on every exception.  These are aimed at first-time
    # users; for a long-running embed job they make the output unreadable.
    litellm.suppress_debug_info = True

    # Stop the duplicate INFO log lines.  LiteLLM emits its own
    # "Wrapper: Completed Call" line on every embedding call, and propagation
    # to the root logger doubles each one (once via LiteLLM's namespace, once
    # via root).  Mute the namespace and stop propagation.
    _llm_log = logging.getLogger("LiteLLM")
    _llm_log.setLevel(logging.WARNING)
    _llm_log.propagate = False

    logger.info(
        "configure_litellm_retries: num_retries=%d timeout=%.0fs",
        num_retries, request_timeout,
    )


def install_litellm_otel_callback(
    num_retries: int = 5,
    request_timeout: float = 300.0,
) -> None:
    """Configure LiteLLM to emit OTel spans into our tracer provider.

    Spans created by ``litellm.embedding(...)`` (and ``litellm.completion(...)``)
    will nest under whatever span is active when the call is made — so the
    LiteLLM span automatically becomes a child of any ``kb.embed`` span the
    knowledge base opens around the call.

    Also calls :func:`configure_litellm_retries` so the rate-limit-resilient
    embedding path is set up regardless of whether tracing is enabled.

    The OTel-callback registration is a no-op if ``init_tracing`` has not
    installed a real provider, but the retry knobs are still applied.
    """
    # Always set retries — they don't depend on tracing being on.
    configure_litellm_retries(num_retries=num_retries, request_timeout=request_timeout)

    if not _initialized or _tracer_provider is None:
        logger.debug("install_litellm_otel_callback: tracing not initialized")
        return

    # litellm[proxy] is a hard dependency — broken install if these fail.
    import litellm
    from litellm.integrations.opentelemetry import (
        OpenTelemetry,
        OpenTelemetryConfig,
    )

    # Hand LiteLLM our existing provider so its spans share the same OTLP
    # exporter and the same trace context as everything else DSAgt emits.
    # exporter="otlp_http" is required by OpenTelemetryConfig but the tracer
    # provider override means the exporter setting itself is unused — the
    # provider already has its own configured exporter.
    config = OpenTelemetryConfig(exporter="otlp_http")
    callback = OpenTelemetry(config=config, tracer_provider=_tracer_provider)

    if callback not in litellm.callbacks:
        litellm.callbacks = (litellm.callbacks or []) + [callback]

    logger.info("install_litellm_otel_callback: registered with tracer provider")


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
        span.add_event(name, attributes={k: v for k, v in attrs.items() if v is not None})

    @staticmethod
    def _current():
        from opentelemetry import trace
        span = trace.get_current_span()
        # The no-op tracer returns INVALID_SPAN; treat that as "no active span".
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
            tracer = get_tracer(fn.__module__)
            with tracer.start_as_current_span(span_name) as span:
                _attach_session_id(span)

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
                            attr_name, type(e).__name__, e,
                        )
                        continue
                    if value is not None:
                        span.set_attribute(attr_name, _coerce_attr(value))

                return result

        return wrapper

    return decorator


def _attach_session_id(span) -> None:
    if _default_session_id:
        span.set_attribute("session.id", _default_session_id)


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
    tracer = get_tracer("dsagt.observability")
    with tracer.start_as_current_span(name) as span:
        _attach_session_id(span)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, _coerce_attr(v))
        yield span


# ----- Knowledge base spans (Stage 1) -----


def kb_embed_span(backend: str | None, model: str | None, n_texts: int):
    """Span around an embedding call.

    Used for both query embedding (kb.search) and chunk embedding (kb.ingest,
    kb.append, kb.add_entries).  Backend-agnostic: ``backend`` is ``"api"``
    for LiteLLM/HTTP embedders or ``"local"`` for sentence-transformers.
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
    * ``record_id`` — matches the ``tool_use_id`` in the proxy's intent record,
      which is how trace_archive correlates execution to LLM intent today.
    * ``tool_name`` — for filtering in the MLflow UI.

    Filter by ``session.id`` in the MLflow trace view to see all tool
    executions for a given project alongside the LLM calls that requested
    them.  Cross-reference via ``record_id`` for full intent → execution
    linkage when needed.
    """
    tracer = get_tracer("dsagt.observability")
    cm = tracer.start_as_current_span("tool.execute")

    @contextmanager
    def _wrapper():
        with cm as span:
            _attach_session_id(span)
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
