"""
DSAgt observability — OTLP span emission to MLflow (or any OTel collector).

Business modules (knowledge.py, provenance.py, registry_server.py, run_tool.py)
import the small public surface defined here.  Spans are emitted via the
OpenTelemetry SDK with an ``OTLPSpanExporter`` pointed at MLflow's
``/v1/traces`` endpoint.  Any OTel-compatible collector (Jaeger, Tempo,
Honeycomb) also works — the only MLflow-specific piece is the
``x-mlflow-experiment-id`` HTTP header that routes the trace to the right
experiment in the receiver.

Public surface
--------------
init_tracing(service_name, *, mlflow_url=None, session_id=None)
    Configure the OTLP exporter once per process.  Reads
    ``MLFLOW_TRACKING_URI`` / ``DSAGT_PROJECT`` / ``DSAGT_SESSION_ID`` from
    the env when not passed.  Raises ``RuntimeError`` when no MLflow URL is
    available — processes that legitimately run without tracing (one-shot
    setup tools, tests) should not call this function.

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

llm_source(source) / llm_call_context(source)
    Marker decorator/context for LLM-call origin tagging.  Currently no-ops:
    LLM-call traces are emitted by the proxy's ``_DSAGTMlflowLogger`` which
    hardcodes ``dsagt.source`` itself.  When the proxy is removed and MCP
    servers enable ``mlflow.litellm.autolog()`` directly, attribute stamping
    for LLM-call traces will live here.
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
    mlflow_url: str | None = None,
    session_id: str | None = None,
) -> None:
    """Install an OTLP-over-HTTP tracer provider routed at MLflow's /v1/traces.

    Reads ``MLFLOW_TRACKING_URI`` and ``DSAGT_PROJECT`` from env when not
    passed.  Resolves the MLflow experiment id once at startup (creating the
    experiment if absent) and bakes it into the OTLP exporter as the
    ``x-mlflow-experiment-id`` HTTP header — MLflow's OTLP receiver requires
    that header and has no name-based fallback.

    Session correlation: ``DSAGT_SESSION_ID`` (or the ``session_id`` arg)
    is stamped on every span as the OTel-semconv ``session.id`` attribute.
    MLflow's OTLP receiver promotes that into ``mlflow.trace.session``
    trace_metadata, which powers the UI's native session-filter widget.

    Raises ``RuntimeError`` when no MLflow URL is available.  Processes
    that legitimately run without tracing (one-shot setup tools, tests)
    should not call this function; tests install a test provider directly.
    """
    global _initialized, _default_session_id

    if _initialized:
        if session_id:
            _default_session_id = session_id
        return

    _default_session_id = session_id or os.environ.get("DSAGT_SESSION_ID")
    mlflow_url = mlflow_url or os.environ.get("MLFLOW_TRACKING_URI")

    if not mlflow_url:
        raise RuntimeError(
            f"{service_name}: no observability backend configured. "
            f"Expected MLFLOW_TRACKING_URI in the environment. Processes "
            f"that legitimately run without tracing (e.g. one-shot setup "
            f"tools) should not call init_tracing at all."
        )

    experiment_id = _resolve_experiment_id(mlflow_url)
    _install_provider(service_name, mlflow_url, experiment_id)
    # Autolog every LiteLLM completion/embedding call from this process into
    # MLflow.  This is what feeds memory extraction (it queries MLflow for
    # session traces) and ``dsagt info`` after the proxy was removed.
    # ``dsagt-run`` doesn't make litellm calls, so skip the import there.
    if service_name != "dsagt-run":
        import mlflow
        mlflow.litellm.autolog()
    _initialized = True
    atexit.register(_shutdown)
    logger.info(
        "init_tracing: service=%s mlflow=%s experiment=%s session=%s",
        service_name, mlflow_url, experiment_id,
        _default_session_id or "<none>",
    )


def _resolve_experiment_id(mlflow_url: str) -> str:
    """Get-or-create the MLflow experiment for ``DSAGT_PROJECT``; return its numeric id.

    MLflow's OTLP receiver requires the numeric experiment id in the
    ``x-mlflow-experiment-id`` header — no name-based fallback exists.
    ``mlflow.set_experiment`` does get-or-create in one call and returns
    the resolved Experiment, so we use that.
    """
    import mlflow
    mlflow.set_tracking_uri(mlflow_url)
    name = os.environ.get("DSAGT_PROJECT", "dsagt")
    return str(mlflow.set_experiment(name).experiment_id)


def _install_provider(
    service_name: str, mlflow_url: str, experiment_id: str
) -> None:
    """Stand up a TracerProvider whose OTLPSpanExporter posts to MLflow."""
    global _tracer_provider
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.util._once import Once

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=f"{mlflow_url.rstrip('/')}/v1/traces",
        headers={"x-mlflow-experiment-id": experiment_id},
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # OTel guards set_tracer_provider with a one-shot Once flag — the first
    # caller wins.  Reset so installation always takes effect even if
    # something accessed get_tracer earlier and locked in the no-op global.
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
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

    logger.debug(
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


def _attach_trace_metadata(span_name: str | None) -> None:
    """Stamp the OTel-semconv ``session.id`` attribute on the active span.

    MLflow's OTLP receiver promotes this attribute into ``mlflow.trace.session``
    trace_metadata server-side, powering the UI's native session-filter
    widget.  No SDK call needed from our side.

    Source bucketing (kb / registry / tool / agent) used to live alongside
    via a ``dsagt.source`` attribute, but post-proxy ``dsagt info`` slices
    by the OTel ``service.name`` resource attribute instead — each emitting
    process (knowledge-server, registry-server, dsagt-run, claude-code,
    goose) has a distinct ``service.name`` and that's the natural bucket.

    No-op when no span is active.  ``span_name`` is unused but kept in the
    signature to avoid touching every call site.
    """
    del span_name  # reserved for future per-span attribute stamping
    from opentelemetry import trace
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    if _default_session_id:
        span.set_attribute("session.id", _default_session_id)


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
# Phase 2: dsagt-proxy callback (cache breakpoints + sidechannel detection)
# ---------------------------------------------------------------------------
#
# When ``dsagt start --enable-proxy`` is set, the proxy intercepts every
# LLM call and emits OTLP spans to MLflow's /v1/traces endpoint via
# LiteLLM's built-in "otel" callback.  Two extra concerns can't be done
# by pure observation — they require request mutation / response logging
# at the proxy hot path:
#
#   1. Anthropic prompt-cache breakpoint injection (saves money on
#      multi-turn sessions; anthropic + bedrock-anthropic honor it,
#      others ignore the marker).
#   2. Sidechannel-call detection (agents' hardcoded title-gen / session-
#      namer models would 400 against lab gateways; we mock them and
#      surface a single warning at teardown so the user can spot a typo
#      in their primary llm.model vs. a harmless sidechannel hit).
#
# These live here (not in provenance.py) because they're observation-time
# concerns, gated by proxy mode, and consumed by the proxy entry point
# in commands/proxy_server.py via :func:`init_proxy_tracing`.

_CACHE_MARKER = {"type": "ephemeral"}


def _inject_cache_breakpoints(messages: list, kwargs: dict) -> None:
    """Mark the largest stable request prefix as cacheable.

    Anthropic prompt caching keys on the prefix UP TO each marked block, so
    one marker at the end of the tools array caches "system + tools", and a
    second on the system message itself ensures the system block is cached
    even on requests with no tools.  Subsequent turns within the 5-minute
    TTL pay 10% on the cached prefix instead of 100%.

    Mutates ``messages`` and ``kwargs`` in place — the proxy reads the same
    objects on its way to the upstream call.

    Marker is ``{"type": "ephemeral"}``; LiteLLM forwards it as
    ``cache_control`` to anthropic-family providers (Anthropic direct,
    Bedrock-Claude).  Providers without prompt caching (current OpenAI
    automatic caching ignores explicit markers, Cohere/Mistral/etc just
    drop them) treat it as a no-op, so this is safe to set unconditionally.
    """
    # 1) Tools: stamp the last tool definition.  In LiteLLM's OpenAI-shape
    #    the tool dict carries cache_control as a top-level key and the
    #    Anthropic translator picks it up.
    tools = kwargs.get("tools") or []
    if tools and isinstance(tools[-1], dict):
        tools[-1]["cache_control"] = _CACHE_MARKER

    # 2) System message: stamp the last text block.  For string content we
    #    promote to block format because cache_control lives on the block,
    #    not the message itself.
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": _CACHE_MARKER,
            }]
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = _CACHE_MARKER
        break


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
#   DSAGTCallback.log_success_event        → record_sidechannel_call()
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


def record_sidechannel_call(records_dir, kwargs: dict) -> None:
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


def print_sidechannel_warning(project_dir, session_id: str | None) -> None:
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


# ---------------------------------------------------------------------------
# DSAGTCallback — LiteLLM CustomLogger for proxy-mode cache + sidechannel
# ---------------------------------------------------------------------------

def _make_dsagt_callback(records_dir):
    """Construct a LiteLLM CustomLogger with cache injection + sidechannel.

    Lazy-imports LiteLLM so the rest of observability.py is testable
    without it on the import path.  Trace transport is via
    ``litellm.callbacks = ["otel"]``, configured by :func:`init_proxy_tracing`
    — this callback is only for the two intercept-time concerns LiteLLM
    autolog can't cover.
    """
    from litellm.integrations.custom_logger import CustomLogger

    class DSAGTCallback(CustomLogger):
        def __init__(self, records_dir):
            self.records_dir = records_dir

        def log_pre_api_call(self, model, messages, kwargs):
            _inject_cache_breakpoints(messages, kwargs)

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_sidechannel_call(self.records_dir, kwargs)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            record_sidechannel_call(self.records_dir, kwargs)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            logger.warning("LLM call failed: model=%s", kwargs.get("model"))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            logger.warning("LLM call failed: model=%s", kwargs.get("model"))

    return DSAGTCallback(records_dir)


def init_proxy_tracing(mlflow_url: str, project: str, session_id: str | None,
                        records_dir) -> None:
    """Configure LiteLLM proxy → MLflow OTel transport, plus install the
    DSAGT cache + sidechannel callback.

    Replaces old_code's ``install_mlflow_logger_with_session_tag`` +
    ``litellm.success_callback = [..., "mlflow"]`` dance.  All trace
    transport is now standard OTLP-over-HTTP to MLflow's
    ``/v1/traces`` — same shape as MCP-server / dsagt-run / agent-native
    OTel emission, just from a different ``service.name``.

    Sets:
      * ``OTEL_EXPORTER_OTLP_ENDPOINT`` → MLflow's /v1/traces
      * ``OTEL_EXPORTER_OTLP_HEADERS`` → ``x-mlflow-experiment-id`` so traces
        bucket into this project's experiment
      * ``OTEL_RESOURCE_ATTRIBUTES`` → ``service.name=dsagt-proxy`` (which
        ``memory.drain_session_traces`` filters on) plus ``session.id``
      * ``litellm.callbacks`` → ``[DSAGTCallback, "otel"]``

    Caller is ``commands/proxy_server.py:main``.  ``records_dir`` is the
    project's ``trace_archive/`` so the sidechannel jsonl lands adjacent.
    """
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{mlflow_url.rstrip('/')}/v1/traces"
    # _resolve_experiment_id reads DSAGT_PROJECT from env; ensure it's set
    # so the proxy's traces bucket into this project's experiment.
    os.environ["DSAGT_PROJECT"] = project
    try:
        exp_id = _resolve_experiment_id(mlflow_url)
    except Exception as e:
        logger.debug("could not resolve experiment id at proxy init: %s", e)
        exp_id = None
    if exp_id:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"x-mlflow-experiment-id={exp_id}"
    resource_attrs = ["service.name=dsagt-proxy"]
    if session_id:
        resource_attrs.append(f"session.id={session_id}")
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)

    import litellm
    callback = _make_dsagt_callback(records_dir)
    # Two callbacks: ours for cache + sidechannel, LiteLLM's "otel" for
    # trace transport.  Order doesn't matter — they touch different things.
    litellm.callbacks = [callback, "otel"]
    logger.info(
        "init_proxy_tracing: service=dsagt-proxy mlflow=%s session=%s",
        mlflow_url, session_id,
    )




