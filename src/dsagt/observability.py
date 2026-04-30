"""
DSAgt observability — span emission to MLflow or any OTel-compatible backend.

Business modules (knowledge.py, provenance.py, registry_server.py, run_tool.py)
import the small public surface defined here and remain backend-agnostic.
``open_span()`` dispatches to either ``mlflow.start_span`` or
``opentelemetry.trace.get_tracer().start_as_current_span`` based on which
endpoint ``init_tracing`` was given.  Both return span objects with the same
OTel-compatible interface (``set_attribute``, ``add_event``, ``record_exception``,
``set_status``), so the rest of the module is one code path.

Public surface
--------------
init_tracing(service_name, *, mlflow_url=None, otel_endpoint=None, session_id=None)
    Configure the backend once per process.  MLflow is preferred when both
    endpoints are available.  No-op if neither is configured.

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

import asyncio
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
# Strategy pointers bound at init_tracing time.  Both exist to keep
# backend-specific behavior out of call sites:
#
# _metadata_stamper(dict)
#   Write key/value metadata to the currently-active trace.  MLflow: goes
#   to trace_metadata via InMemoryTraceManager (queryable in the UI).
#   OTel: goes to active span attributes (queryable in Jaeger/Tempo).
#
# _llm_context_factory(dict) → context manager
#   Tag every trace created inside the returned context.  MLflow uses
#   mlflow.tracing.context which stamps at trace-creation time.  OTel
#   has no direct analog and no-ops — OTel backend would need baggage
#   propagation wired in if source tagging becomes concrete there.
_metadata_stamper: "Callable[[dict], None] | None" = None
_llm_context_factory: "Callable[[dict], Any] | None" = None


def init_tracing(
    service_name: str,
    mlflow_url: str | None = None,
    otel_endpoint: str | None = None,
    session_id: str | None = None,
) -> None:
    """Install a tracer provider and bind the session-stamping strategy.

    Picks one of two provider flavors:

    * **MLflow** — ``mlflow_url`` arg or ``MLFLOW_TRACKING_URI`` env var.
      MLflow's own ``TracerProvider`` is installed as the OTel global, so
      every ``trace.get_tracer(...)`` call routes spans into MLflow's store.
      Session id is stamped as the reserved ``mlflow.trace.session`` trace
      metadata key (powers the UI's native session filter).
    * **OTel** — ``otel_endpoint`` arg or ``OTEL_EXPORTER_OTLP_ENDPOINT``.
      A fresh ``TracerProvider`` with an OTLP exporter is installed.  Spans
      go to any OTLP-compatible collector (Jaeger, Tempo, Honeycomb).
      Session id is attached as a ``session.id`` span attribute.

    Neither configured → ``RuntimeError``.  Processes that legitimately run
    without observability (one-shot setup tools, tests) should not call this
    function; tests install a test provider directly.
    """
    global _initialized, _tracer_provider, _default_session_id
    global _metadata_stamper, _llm_context_factory

    if _initialized:
        if session_id:
            _default_session_id = session_id
        return

    _default_session_id = session_id or os.environ.get("DSAGT_SESSION_ID")
    mlflow_url = mlflow_url or os.environ.get("MLFLOW_TRACKING_URI")
    otel_endpoint = otel_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if mlflow_url:
        _install_mlflow_provider(mlflow_url)
        _metadata_stamper = _stamp_metadata_mlflow
        _llm_context_factory = _mlflow_llm_context
        # NOTE: we deliberately do NOT call ``mlflow.litellm.autolog()`` here.
        # Post-Option-A all litellm calls from MCP servers go through the
        # local proxy at localhost:<proxy_port>, where ``_DSAGTMlflowLogger``
        # autologs them with full request/response, tokens, cost, and cache
        # stats.  Enabling autolog here too would produce a second MLflow
        # trace per call (the MCP-server-side one) carrying only thin span
        # metadata — a duplicate that's strictly less informative than the
        # proxy-side trace.  The MCP server's ``kb.*`` / ``registry.*`` /
        # ``tool.execute`` spans (created by @traced/@llm_source decorators)
        # remain as attestation that the MCP tool ran.
        _initialized = True
        logger.info(
            "init_tracing: service=%s backend=mlflow url=%s session=%s",
            service_name, mlflow_url, _default_session_id or "<none>",
        )
        return

    if otel_endpoint:
        _install_otlp_provider(service_name, otel_endpoint)
        _metadata_stamper = _stamp_metadata_otel
        _llm_context_factory = _noop_llm_context
        _initialized = True
        atexit.register(_shutdown)
        logger.info(
            "init_tracing: service=%s backend=otel endpoint=%s session=%s",
            service_name, otel_endpoint, _default_session_id or "<none>",
        )
        return

    raise RuntimeError(
        f"{service_name}: no observability backend configured. "
        f"Expected MLFLOW_TRACKING_URI or OTEL_EXPORTER_OTLP_ENDPOINT in the "
        f"environment. Processes that legitimately run without tracing "
        f"(e.g. one-shot setup tools) should not call init_tracing at all."
    )


def _install_mlflow_provider(mlflow_url: str) -> None:
    """Wire MLflow's tracer provider in as the OTel global."""
    import mlflow
    from mlflow.tracing import provider as mp
    from opentelemetry import trace
    from opentelemetry.util._once import Once

    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(os.environ.get("DSAGT_PROJECT", "dsagt"))

    # Force MLflow's lazy provider to initialize via its private init hook
    # so we can hand the resulting TracerProvider to OTel below.  Earlier
    # versions of this code used ``with mlflow.start_span(name="_bootstrap")``
    # for the same purpose — but that emitted a clutter ``_bootstrap`` trace
    # in the MLflow UI from every dsagt-run / MCP-server subprocess.  The
    # underscore on the init function is MLflow-internal — pinning the
    # mlflow version range in pyproject.toml keeps that boundary stable.
    mp._initialize_tracer_provider()

    # OTel guards set_tracer_provider with a one-shot Once flag.  We reset
    # it so installing MLflow's provider actually takes effect (necessary in
    # long-running processes that may have touched the no-op global first).
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    trace.set_tracer_provider(mp.provider.get())


def _install_otlp_provider(service_name: str, otel_endpoint: str) -> None:
    """Stand up a fresh TracerProvider with an OTLP exporter."""
    global _tracer_provider
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{otel_endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def extract_cache_stats(usage: dict) -> tuple[int, int]:
    """Extract (cache_read, cache_write) tokens from a LiteLLM usage dict.

    LiteLLM's ``Usage`` object doesn't backfill cache fields across
    providers; whichever shape the upstream returned is the one populated.
    The configured ``LLM_PROVIDER`` (passed to ``dsagt-proxy --provider X``
    and used as the LiteLLM provider prefix in the model_list) decides
    which response format we get back.  The four field-name conventions
    we've seen in practice are all checked here so this function works
    regardless of provider config:

    Cache read (tokens served from cache at the lower rate):
        - ``cache_read_input_tokens``           — Anthropic, Bedrock-Claude
        - ``prompt_tokens_details.cached_tokens`` — OpenAI, Azure-OpenAI
        - ``cached_content_token_count``        — Gemini
        - ``prompt_cache_hit_tokens``           — DeepSeek

    Cache write (only Anthropic-family bills cache writes separately at
    the 1.25× premium; OpenAI/Gemini/DeepSeek auto-cache without a
    distinct write fee, so the field doesn't exist):
        - ``cache_creation_input_tokens``       — Anthropic, Bedrock-Claude
    """
    if not isinstance(usage, dict):
        return (0, 0)

    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}

    read = (
        usage.get("cache_read_input_tokens")
        or details.get("cached_tokens")
        or usage.get("cached_content_token_count")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    write = usage.get("cache_creation_input_tokens") or 0
    return (int(read), int(write))


def stamp_metadata(metadata: dict) -> None:
    """Stamp arbitrary key/value metadata on the currently-active trace.

    No-op when no backend is configured (tests, standalone tools).  See the
    ``_metadata_stamper`` comment up top for backend-specific behavior.
    """
    if _metadata_stamper is not None and metadata:
        _metadata_stamper(metadata)


def _stamp_metadata_on_trace(request_id: str, metadata: dict) -> None:
    """Write metadata to a specific MLflow trace by id.

    Used when the caller has the trace_id in hand (e.g. inside LiteLLM's
    MlflowLogger subclass right after ``start_trace``).  ``stamp_metadata``
    is the higher-level version that looks up the current trace via the
    active OTel span.
    """
    try:
        from mlflow.tracing.trace_manager import InMemoryTraceManager
        with InMemoryTraceManager.get_instance().get_trace(request_id) as t:
            if t is not None:
                t.info.trace_metadata.update(
                    {k: str(v) for k, v in metadata.items()}
                )
    except Exception as e:
        logger.debug("metadata stamp failed for %s: %s", request_id, e)


def _stamp_metadata_mlflow(metadata: dict) -> None:
    """MLflow backend: write to trace_metadata via the trace manager."""
    from opentelemetry import trace
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return
    _stamp_metadata_on_trace(f"tr-{ctx.trace_id:032x}", metadata)


def _stamp_metadata_otel(metadata: dict) -> None:
    """OTel backend: write to active span attributes."""
    from opentelemetry import trace
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    for k, v in metadata.items():
        span.set_attribute(k, str(v))


def _mlflow_llm_context(metadata: dict):
    """MLflow: native tracing.context stamps at trace-creation time."""
    import mlflow
    return mlflow.tracing.context(metadata=metadata)


@contextmanager
def _noop_llm_context(metadata: dict):
    """OTel backend or no backend: nothing to do.

    A real OTel implementation would use baggage propagation so traces
    created inside inherit the keys, but DSAGT's OTel path is preserved
    for flexibility rather than actively exercised — extend here when it
    becomes concrete.
    """
    yield


@contextmanager
def llm_call_context(source: str):
    """Stamp ``dsagt.source`` (+ ``dsagt.agent`` when set) on traces created
    inside this block.

    MLflow backend: metadata is attached at trace-creation time via MLflow's
    native tracing.context API, so the UI can filter by it.  OTel backend
    currently no-ops — see ``_noop_llm_context``.
    """
    metadata = {"dsagt.source": source}
    if agent := os.environ.get("DSAGT_AGENT"):
        metadata["dsagt.agent"] = agent
    if _default_session_id:
        metadata["mlflow.trace.session"] = _default_session_id
    if _llm_context_factory is not None:
        with _llm_context_factory(metadata):
            yield
    else:
        yield


def llm_source(source: str):
    """Decorator form of ``llm_call_context`` for tidy call sites.

    Handles both sync and async.  Every LLM call made inside the decorated
    function lands in MLflow with ``dsagt.source = <source>`` metadata,
    letting the UI distinguish extraction / embedding / agent-turn origins
    at a glance.
    """
    def dec(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a, **kw):
                with llm_call_context(source):
                    return await fn(*a, **kw)
            return aw
        @functools.wraps(fn)
        def sw(*a, **kw):
            with llm_call_context(source):
                return fn(*a, **kw)
        return sw
    return dec


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
    """Return an OTel tracer. Routes through whichever provider init_tracing
    installed (MLflow's or an OTLP one)."""
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
                            attr_name, type(e).__name__, e,
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

    Prefix-based dispatch so every span we open (now or later) lands with
    a source tag without touching the call site.  LLM traces get their
    source from ``_DSAGTMlflowLogger`` (agent) or ``@llm_source``
    (extraction, embedding) instead — this helper covers the non-LLM
    spans our own instrumentation emits.
    """
    if not span_name:
        return None
    for prefix, source in _SOURCE_BY_PREFIX:
        if span_name.startswith(prefix):
            return source
    return None


def _attach_trace_metadata(span_name: str | None) -> None:
    """Stamp session + source on the currently-active trace.

    - ``mlflow.trace.session``: process-wide session id (reserved MLflow key;
      powers the UI's native session filter).
    - ``dsagt.source``: derived from the span name prefix — so every span we
      open lands in ``dsagt info``'s "by source" bucket without extra work
      at the call site.

    No-op when no backend is configured or no span is active; the metadata
    stamper handles both cases.
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
# Sidechannel model-call handling
# ---------------------------------------------------------------------------
#
# Every agent platform hardcodes a small/fast model for internal features
# (goose → gpt-4o-mini session-namer; claude → claude-haiku-4-5... title
# generator; the next agent will pick its own).  When the user's gateway
# doesn't carry that exact bare name — which is the norm for lab gateways
# that alias every model — those requests would 400 and clutter MLflow.
#
# Rather than maintain a per-vendor list of known hardcoded names (which
# rots as vendors rename their sidechannel models), DSAGT catches all of
# them with one wildcard LiteLLM route, records which names fired, and
# surfaces a single yellow warning at session teardown so the user can
# distinguish a harmless sidechannel from a typo in their own config.
#
# Callers import the specific thing they need:
#   commands/proxy_server._generate_config → SIDECHANNEL_WILDCARD_ROUTE_YAML
#   provenance._handle_success             → record_sidechannel_call()
#   commands/cli._cmd_start (teardown)     → print_sidechannel_warning()


import json as _json  # local alias keeps the section self-contained
import sys as _sys
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path

#: Env var the parent process exports so the proxy's DSAGT callback can tell
#: "this is the configured primary model" from "this is a sidechannel hit".
SIDECHANNEL_PRIMARY_MODEL_ENV = "DSAGT_PRIMARY_MODEL"

#: JSONL file (one entry per intercepted call) that the proxy subprocess
#: appends to and the parent reads at teardown.  Lives at the project
#: directory root, adjacent to ``trace_archive/``.
SIDECHANNEL_LOG_FILENAME = "sidechannel.jsonl"

#: Canned reply the wildcard returns.  Short enough that goose's
#: session-namer (expects ≤4 words) and claude's title generator both
#: accept it without error.
_SIDECHANNEL_CANNED_RESPONSE = "session"

#: Post-routing model name LiteLLM resolves the wildcard catchall to.  Only
#: true mock hits end up here; explicit primary entries and aliases route
#: to the real upstream regardless of what name the client requested.  The
#: sidechannel detector uses this to avoid flagging alias hits as
#: sidechannel — see ``record_sidechannel_call``.
SIDECHANNEL_CATCHALL_MODEL = "openai/dsagt-sidechannel-catchall"

#: Where the user can read the longer explanation.  Printed in the warning.
SIDECHANNEL_DOC_LOCATION = "README.md § Sidechannel model calls"

#: YAML fragment appended to the proxy's ``model_list`` after the primary
#: route.  LiteLLM prefers exact matches over wildcards, so the configured
#: model still routes normally; everything else falls through to the mock.
#:
#: ``api_base`` has to be a syntactically valid URL but is never dialed —
#: ``mock_response`` short-circuits upstream entirely.
SIDECHANNEL_WILDCARD_ROUTE_YAML = f"""\
  - model_name: "*"
    litellm_params:
      model: {SIDECHANNEL_CATCHALL_MODEL}
      api_base: http://invalid.local
      api_key: unused
      mock_response: "{_SIDECHANNEL_CANNED_RESPONSE}"
"""


def _sidechannel_client_requested_model(kwargs: dict) -> str | None:
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


def record_sidechannel_call(records_dir: _Path, kwargs: dict) -> None:
    """Append a JSONL entry when a request hit the wildcard catchall.

    Detection rule: ``kwargs["model"]`` (the post-routing target LiteLLM
    selected) equals ``SIDECHANNEL_CATCHALL_MODEL``.  This is the only
    reliable discriminator: name-based comparison against the primary
    misclassifies alias hits (e.g. ``claude-sonnet-4-5`` aliased to a
    longer lab-gateway model name) as sidechannel even though they
    actually routed to the real upstream.

    Called from the DSAGT callback's success handler, so only successful
    calls get logged — failures land in MLflow as errors regardless.
    No-ops when ``SIDECHANNEL_PRIMARY_MODEL_ENV`` isn't set.
    """
    primary = os.environ.get(SIDECHANNEL_PRIMARY_MODEL_ENV)
    if not primary:
        return

    routed_to = kwargs.get("model") or ""
    if routed_to != SIDECHANNEL_CATCHALL_MODEL:
        return  # real upstream call (primary entry or alias) — not a sidechannel

    requested = _sidechannel_client_requested_model(kwargs)
    if not requested:
        return

    entry = {
        "timestamp": _datetime.now(_timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": requested,
        "agent": os.environ.get("DSAGT_AGENT", ""),
        "session": os.environ.get("DSAGT_SESSION_ID", ""),
    }
    path = _Path(records_dir).parent / SIDECHANNEL_LOG_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug("sidechannel log append failed: %s", e)


def print_sidechannel_warning(project_dir: _Path, session_id: str | None) -> None:
    """Read ``SIDECHANNEL_LOG_FILENAME`` under *project_dir*, dedup within
    *session_id*, and print a yellow warning to stdout.

    No-op when nothing was logged for this session (common case).  The
    warning lists each unique model that hit the wildcard along with the
    call count, and points the user at ``SIDECHANNEL_DOC_LOCATION`` for the
    two possible causes (harmless sidechannel vs config typo).

    ANSI colors are only emitted when stdout is a TTY — CI logs stay clean.
    """
    log_path = _Path(project_dir) / SIDECHANNEL_LOG_FILENAME
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
            entry = _json.loads(line)
            if session_id and entry.get("session") != session_id:
                continue
            model = entry.get("model") or "<unknown>"
            seen[model] = seen.get(model, 0) + 1
    except (OSError, ValueError):
        return

    if not seen:
        return

    tty = _sys.stdout.isatty()
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
    print(f"{yellow}    See: {SIDECHANNEL_DOC_LOCATION}{reset}")
