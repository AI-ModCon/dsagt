"""
Tests for dsagt.observability — Stage 0.

These tests use OTel's InMemorySpanExporter so they don't require a real
collector. They cover:

* init_tracing is a no-op without an endpoint
* init_tracing with an endpoint installs a tracer provider
* @traced opens a span, captures args, sets duration_ms, records exceptions
* obs.set / obs.event are no-ops outside a span and write attributes inside
* child_span nests under the active span
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from dsagt import observability as obs_module
from dsagt.observability import child_span, init_tracing, obs, traced


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch):
    """Reset module state and install an in-memory exporter for each test.

    We bypass ``init_tracing`` (no MLflow / OTLP endpoint to talk to in tests)
    and install our own TracerProvider with an InMemorySpanExporter directly.
    The session-stamp strategy is bound to the OTel one so ``_attach_session_id``
    behaves the way production OTel callers would.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(obs_module, "_tracer_provider", None)
    monkeypatch.setattr(obs_module, "_default_session_id", None)
    monkeypatch.setattr(obs_module, "_metadata_stamper", obs_module._stamp_metadata_otel)
    monkeypatch.setattr(obs_module, "_llm_context_factory", obs_module._noop_llm_context)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    from opentelemetry.util._once import Once
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    monkeypatch.setattr(obs_module, "_initialized", True)

    yield exporter

    exporter.clear()


@pytest.mark.parametrize("span_name,expected", [
    ("tool.execute", "tool"),
    ("kb.search", "knowledge"),
    ("kb.embed", "knowledge"),
    ("registry.save_tool_spec", "registry"),
    ("litellm-acompletion", None),  # LLM traces use _DSAGTMlflowLogger, not this
    ("", None),
    (None, None),
])
def test_derive_source_from_span_name(span_name, expected):
    assert obs_module._derive_source(span_name) == expected


def test_traced_stamps_source_derived_from_span_name(_reset_tracing):
    """Every traced function should emit a span tagged with dsagt.source
    derived from its name prefix — so dsagt info can bucket them without
    a decorator at every call site."""
    exporter = _reset_tracing

    @traced("tool.execute")
    def fake_tool():
        return None
    fake_tool()

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("dsagt.source") == "tool"


def test_init_tracing_no_endpoint_raises(monkeypatch):
    """init_tracing must fail loudly when no backend is configured — silent
    no-op behavior would let a misconfigured subprocess run with tracing
    silently dropped, which is exactly the kind of silent-fallback bug
    DSAGT's design principles prohibit."""
    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(obs_module, "_metadata_stamper", None)
    monkeypatch.setattr(obs_module, "_llm_context_factory", None)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="no observability backend"):
        init_tracing("test-service")


def test_traced_emits_span_with_args(_reset_tracing):
    exporter = _reset_tracing

    @traced("test.op", capture=["a", "b"])
    def f(a, b, c=10):
        return a + b + c

    result = f(1, 2, c=3)
    assert result == 6

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "test.op"
    assert span.attributes["a"] == 1
    assert span.attributes["b"] == 2
    assert "c" not in span.attributes  # not in capture list
    assert "duration_ms" in span.attributes


def test_traced_extract_return(_reset_tracing):
    exporter = _reset_tracing

    @traced("test.op", extract_return={"hits": len})
    def search():
        return ["a", "b", "c"]

    search()
    span = exporter.get_finished_spans()[0]
    assert span.attributes["hits"] == 3


def test_traced_records_exception(_reset_tracing):
    exporter = _reset_tracing

    @traced("test.boom")
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    # exception is recorded as a span event
    assert any(e.name == "exception" for e in span.events)
    # duration is still set even on error path
    assert "duration_ms" in span.attributes


def test_obs_set_outside_span_is_noop():
    """obs.set must not raise when called with no active span."""
    obs.set("nothing", 1)  # must be silent
    obs.set_many({"a": 1, "b": 2})
    obs.event("ping", x=1)


def test_obs_set_inside_span_writes_attribute(_reset_tracing):
    exporter = _reset_tracing

    @traced("test.op")
    def f():
        obs.set("hits", 7)
        obs.set_many({"foo": "bar", "skipped": None})
        obs.event("milestone", phase="middle")
        return None

    f()
    span = exporter.get_finished_spans()[0]
    assert span.attributes["hits"] == 7
    assert span.attributes["foo"] == "bar"
    assert "skipped" not in span.attributes
    assert any(e.name == "milestone" for e in span.events)


def test_child_span_nests(_reset_tracing):
    exporter = _reset_tracing

    @traced("test.parent")
    def parent():
        with child_span("test.child", phase="embed"):
            pass

    parent()
    spans = exporter.get_finished_spans()
    # Children finish first, then the parent.
    by_name = {s.name: s for s in spans}
    child = by_name["test.child"]
    parent_span = by_name["test.parent"]
    assert child.parent is not None
    assert child.parent.span_id == parent_span.context.span_id
    assert child.attributes["phase"] == "embed"


def test_session_id_attached(_reset_tracing, monkeypatch):
    exporter = _reset_tracing
    monkeypatch.setattr(obs_module, "_default_session_id", "proj-xyz")

    @traced("test.op")
    def f():
        pass

    f()
    span = exporter.get_finished_spans()[0]
    # The metadata stamper writes the MLflow-reserved key on both backends
    # (MLflow: trace_metadata; OTel: span attribute with the same name).
    assert span.attributes["mlflow.trace.session"] == "proj-xyz"


def test_init_tracing_double_call_only_updates_session(monkeypatch):
    """Re-calling init_tracing should not raise; should update session id."""
    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", "old")
    init_tracing("test", session_id="new")
    assert obs_module._default_session_id == "new"


# ---------------------------------------------------------------------------
# Safety nets for the remaining defensive catches in observability.py.
#
# After the fallback-purge pass, three "soft" catches remain in the
# observability layer.  Two of them now live inline inside traced()'s
# wrapper (previously _attach_captured_args and _attach_return_attrs):
#
#   1. _shutdown wraps tracer_provider.shutdown() in try/except so that
#      a shutdown failure during process exit can't escape into the
#      atexit chain and corrupt the parent process exit.
#   2. traced() wraps sig.bind_partial in except TypeError so that a
#      function whose signature was mangled by another decorator doesn't
#      crash on every traced call.
#   3. traced() wraps each user-supplied extractor lambda in except
#      Exception so a buggy lambda doesn't crash the instrumented
#      function.
#
# These tests pin the HAPPY PATH so that if those catches ever fire in
# normal use the test suite fails immediately.  Without them, the
# silent-degradation behavior of the catches would hide real bugs.
# ---------------------------------------------------------------------------


def test_shutdown_runs_cleanly_against_real_provider(monkeypatch):
    """The _shutdown helper must successfully flush and shut down a real
    BatchSpanProcessor-backed provider on the happy path.

    If the catch in _shutdown ever fires during dsagt-setup-kb's atexit
    pass, the user loses any spans still buffered in the BatchSpanProcessor.
    This test catches "shutdown raised an exception we silently swallowed"
    by exercising the real shutdown path against a real (in-memory) provider.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "shutdown-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Plug it into the module so _shutdown reaches the real object.
    monkeypatch.setattr(obs_module, "_tracer_provider", provider)

    # Emit a span first so there's actually something to flush.
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("pre-shutdown"):
        pass

    # Run shutdown — must not raise, and must clear the module-level handle.
    obs_module._shutdown()
    assert obs_module._tracer_provider is None

    # The exporter should still hold the span we emitted before shutdown.
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "pre-shutdown"

    # Calling shutdown again is a no-op (idempotent atexit safety).
    obs_module._shutdown()


def test_extract_return_failure_logs_at_debug_and_does_not_crash(_reset_tracing, caplog):
    """A buggy extract_return lambda must NOT crash the instrumented function,
    must NOT crash the span emission, and must produce a DEBUG log line so
    the developer can spot the silently-missing attribute when running with
    --verbose.
    """
    import logging as _logging

    @traced(
        "test.extractor_bug",
        extract_return={
            "good": lambda r: len(r),
            "bad": lambda r: r["nonexistent_key"],  # KeyError on every call
            "another_good": lambda r: r[0] if r else None,
        },
    )
    def f():
        return ["a", "b", "c"]

    with caplog.at_level(_logging.DEBUG, logger="dsagt.observability"):
        result = f()

    # Function still returned its value normally.
    assert result == ["a", "b", "c"]

    # Span was emitted with the good attributes set, bad one missing.
    spans = _reset_tracing.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["good"] == 3
    assert span.attributes["another_good"] == "a"
    assert "bad" not in span.attributes

    # The failure was logged at DEBUG with the attribute name.  This is
    # what makes the silent skip visible to a developer running with
    # --verbose / DEBUG logging.
    debug_messages = [r.message for r in caplog.records if r.levelno == _logging.DEBUG]
    assert any("extract_return['bad']" in m or "'bad'" in m for m in debug_messages), (
        f"Expected a DEBUG log mentioning the broken 'bad' extractor, "
        f"got: {debug_messages}"
    )


def test_attach_captured_args_happy_path_protects_bind_partial_catch(_reset_tracing):
    """Pin the happy path for _attach_captured_args so the silent
    'except TypeError: return' catch can't hide a regression where args
    stop being captured due to a signature-introspection bug.

    If sig.bind_partial silently failed for any reason, this test would
    fail because the captured 'a' and 'b' attributes would be missing
    from the emitted span.
    """
    @traced("test.signature_capture", capture=["a", "b", "c"])
    def f(a, b, c=42, *, d=None):
        return None

    f(1, b=2, d="ignored")
    span = _reset_tracing.get_finished_spans()[0]
    assert span.attributes["a"] == 1
    assert span.attributes["b"] == 2
    assert span.attributes["c"] == 42  # default value still captured
    assert "d" not in span.attributes


def test_litellm_imports_at_observability_init_no_fallback():
    """Regression test for the deletion of `except ImportError: return` in
    configure_litellm_retries.

    litellm is a hard dependency in pyproject.toml.  If anyone re-introduces
    a try/except around the import (turning litellm "optional" again), the
    function would silently no-op and the rate-limit retry/backoff knobs
    would never be configured — exactly the silent-degradation pattern
    we're trying to eliminate.

    This test asserts the import succeeds AND the side-effects of
    configure_litellm_retries actually happened.
    """
    import litellm
    from dsagt.observability import configure_litellm_retries

    # Mutate litellm state to known-bad values, then call configure and
    # verify it stomped them.  If configure ever silently no-ops on a
    # caught ImportError, the asserts below would fail.
    litellm.num_retries = -999
    litellm.request_timeout = -999.0

    configure_litellm_retries(num_retries=7, request_timeout=42.0)

    assert litellm.num_retries == 7
    assert litellm.request_timeout == 42.0


# ---------------------------------------------------------------------------
# Stage 1: KnowledgeBase instrumentation
# ---------------------------------------------------------------------------


def _spans_by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


@contextmanager
def _kb_with_mocked_embedder(tmp_path, backend: str = "api", model: str = "test-model"):
    """Build a KnowledgeBase with a mocked embedder for the given backend.

    The mock patch lives for the whole context so cache misses on later
    embed() calls still resolve to the fake.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np

    from dsagt.knowledge import KnowledgeBase

    def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    mock_client = MagicMock()
    mock_client.embed = fake_embed

    with patch("dsagt.knowledge._make_embedder", return_value=mock_client):
        kb = KnowledgeBase(
            index_dir=tmp_path / f"kb_{backend}",
            default_embedder=backend,
            embedder_kwargs={"model": model},
        )
        try:
            yield kb
        finally:
            kb.close()


def test_kb_search_emits_three_child_spans(_reset_tracing, tmp_path):
    """kb.search should produce kb.search → {kb.embed, kb.index_search}."""
    exporter = _reset_tracing
    with _kb_with_mocked_embedder(tmp_path) as kb:
        # Seed a collection so search has something to load.
        kb.add_entries(texts=["hello world", "goodbye"], collection="tcoll")
        exporter.clear()  # ignore add_entries spans for this assertion

        results = kb.search("hello", collection="tcoll", top_k=2, rerank=False)
        assert isinstance(results, list)

    spans = _spans_by_name(exporter)
    assert "kb.search" in spans
    assert "kb.embed" in spans
    assert "kb.index_search" in spans
    # No rerank requested, so no rerank span.
    assert "kb.rerank" not in spans

    parent = spans["kb.search"]
    embed = spans["kb.embed"]
    index = spans["kb.index_search"]

    assert embed.parent.span_id == parent.context.span_id
    assert index.parent.span_id == parent.context.span_id

    # Captured args + obs.set('hits', ...) on the parent.
    assert parent.attributes["collection"] == "tcoll"
    assert parent.attributes["top_k"] == 2
    assert parent.attributes["rerank"] is False
    assert "hits" in parent.attributes
    assert "duration_ms" in parent.attributes

    assert embed.attributes["backend"] == "api"
    assert embed.attributes["model"] == "test-model"
    assert embed.attributes["n_texts"] == 1

    assert index.attributes["k"] >= 1
    assert index.attributes["filtered"] is False


def test_kb_search_local_backend_same_span_shape(_reset_tracing, tmp_path):
    """The local embedding backend should emit the same span tree."""
    exporter = _reset_tracing
    with _kb_with_mocked_embedder(tmp_path, backend="local", model="bge-base") as kb:
        kb.add_entries(texts=["hello world"], collection="tcoll")
        exporter.clear()
        kb.search("hello", collection="tcoll", top_k=1, rerank=False)

    spans = _spans_by_name(exporter)
    assert "kb.search" in spans
    assert "kb.embed" in spans
    assert spans["kb.embed"].attributes["backend"] == "local"
    assert spans["kb.embed"].attributes["model"] == "bge-base"


def test_kb_ingest_emits_embed_child(_reset_tracing, tmp_path):
    """kb.ingest opens an outer span and one child kb.embed span."""
    exporter = _reset_tracing
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.txt").write_text("alpha beta gamma")
    (src / "b.txt").write_text("delta epsilon zeta")

    with _kb_with_mocked_embedder(tmp_path) as kb:
        kb.ingest(src)

    spans = _spans_by_name(exporter)
    assert "kb.ingest" in spans
    assert "kb.embed" in spans

    ingest = spans["kb.ingest"]
    embed = spans["kb.embed"]
    assert embed.parent.span_id == ingest.context.span_id
    assert ingest.attributes["n_files"] == 2
    assert ingest.attributes["n_chunks"] >= 2
    assert embed.attributes["n_texts"] == ingest.attributes["n_chunks"]


def test_kb_add_entries_emits_span(_reset_tracing, tmp_path):
    """kb.add_entries should emit a top-level span with n_entries."""
    exporter = _reset_tracing
    with _kb_with_mocked_embedder(tmp_path) as kb:
        kb.add_entries(texts=["one", "two", "three"], collection="epis")

    spans = _spans_by_name(exporter)
    assert "kb.add_entries" in spans
    assert spans["kb.add_entries"].attributes["collection"] == "epis"
    assert spans["kb.add_entries"].attributes["n_entries"] == 3


# ---------------------------------------------------------------------------
# Stage 2: LiteLLM retry wiring
# ---------------------------------------------------------------------------


def test_configure_litellm_retries_sets_module_globals(monkeypatch):
    """configure_litellm_retries must set litellm.num_retries / request_timeout."""
    import litellm

    from dsagt.observability import configure_litellm_retries

    # Save and clear so the test owns the values.
    monkeypatch.setattr(litellm, "num_retries", None, raising=False)
    monkeypatch.setattr(litellm, "request_timeout", 0.0, raising=False)

    configure_litellm_retries(num_retries=7, request_timeout=42.0)

    assert litellm.num_retries == 7
    assert litellm.request_timeout == 42.0


def test_configure_litellm_retries_works_without_tracing(monkeypatch):
    """The retry knobs must be applied even when tracing is not initialized.

    This is the dsagt-setup-kb path: the long-running embed job that has to
    survive rate limits also runs before any MLflow endpoint exists.
    """
    import litellm

    from dsagt.observability import configure_litellm_retries

    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(obs_module, "_tracer_provider", None)
    monkeypatch.setattr(litellm, "num_retries", None, raising=False)

    configure_litellm_retries(num_retries=3, request_timeout=120.0)

    assert litellm.num_retries == 3
    assert litellm.request_timeout == 120.0


def test_kb_search_does_not_call_real_litellm(_reset_tracing, tmp_path):
    """Sanity check: with a mocked embedder, no litellm.embedding call escapes."""
    from unittest.mock import patch as _patch

    exporter = _reset_tracing
    with _patch("litellm.embedding") as mock_embed:
        with _kb_with_mocked_embedder(tmp_path) as kb:
            kb.add_entries(texts=["hello"], collection="tcoll")
            kb.search("hello", collection="tcoll", top_k=1)

    # _kb_with_mocked_embedder patches _make_embedder, so litellm.embedding
    # should never be called even though APIEmbeddingClient now uses it.
    assert mock_embed.call_count == 0
    assert "kb.search" in _spans_by_name(exporter)


# ---------------------------------------------------------------------------
# Stage 3: tool execution spans
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    from dsagt.observability import truncate
    assert truncate("hello", 256) == "hello"


def test_truncate_long_string_appends_suffix():
    from dsagt.observability import truncate
    s = "x" * 500
    result = truncate(s, 64)
    assert len(result) < 100  # truncated, not full length
    assert result.startswith("x" * 32)
    assert "[+" in result and "chars]" in result


def test_truncate_handles_none():
    from dsagt.observability import truncate
    assert truncate(None, 256) == ""


def test_tool_execute_span_attributes(_reset_tracing):
    """tool_execute_span sets record_id and tool_name on the span."""
    exporter = _reset_tracing

    from dsagt.observability import obs, tool_execute_span

    with tool_execute_span(record_id="abc123", tool_name="fastp"):
        obs.set("exit_code", 0)
        obs.set("duration_ms", 42.5)

    spans = _spans_by_name(exporter)
    assert "tool.execute" in spans
    span = spans["tool.execute"]
    assert span.attributes["record_id"] == "abc123"
    assert span.attributes["tool_name"] == "fastp"
    assert span.attributes["exit_code"] == 0
    assert span.attributes["duration_ms"] == 42.5


def test_run_and_record_emits_tool_execute_span(_reset_tracing, tmp_path):
    """run_and_record() should produce a tool.execute span with the
    expected execution attributes."""
    exporter = _reset_tracing

    from dsagt.provenance import run_and_record

    rc = run_and_record(
        tool_name="echo",
        command=["echo", "hello world"],
        records_dir=tmp_path / "records",
        session_id="test-session",
        record_id="rec-001",
        input_files=["in.txt"],
        output_files=["out.txt", "out2.txt"],
    )
    assert rc == 0

    spans = _spans_by_name(exporter)
    assert "tool.execute" in spans
    span = spans["tool.execute"]

    assert span.attributes["record_id"] == "rec-001"
    assert span.attributes["tool_name"] == "echo"
    assert span.attributes["exit_code"] == 0
    assert span.attributes["duration_ms"] >= 0
    assert span.attributes["n_input_files"] == 1
    assert span.attributes["n_output_files"] == 2
    assert span.attributes["command"].startswith("echo")
    assert span.attributes["stdout_len"] > 0


def test_run_and_record_failed_tool_records_event_and_status(_reset_tracing, tmp_path):
    """A failed tool call should emit a tool_failed event with the exit code
    and the truncated stderr should be attached."""
    exporter = _reset_tracing

    from dsagt.provenance import run_and_record

    rc = run_and_record(
        tool_name="missing_tool",
        command=["this-binary-does-not-exist-anywhere"],
        records_dir=tmp_path / "records",
        session_id="test-session",
    )
    assert rc == 127

    spans = _spans_by_name(exporter)
    span = spans["tool.execute"]

    assert span.attributes["exit_code"] == 127
    # Stderr was set by the FileNotFoundError branch — should be on the span.
    assert "stderr_truncated" in span.attributes
    # tool_failed event was added.
    assert any(e.name == "tool_failed" for e in span.events)


def test_run_and_record_long_stderr_is_truncated(_reset_tracing, tmp_path):
    """Span attributes should never carry multi-megabyte stderr blobs."""
    exporter = _reset_tracing

    from dsagt.provenance import run_and_record

    # Use python -c to emit a large stderr deterministically.
    big = "x" * 5000
    rc = run_and_record(
        tool_name="echo_err",
        command=["python", "-c", f"import sys; sys.stderr.write('{big}'); sys.exit(0)"],
        records_dir=tmp_path / "records",
        session_id="s",
    )
    assert rc == 0

    spans = _spans_by_name(exporter)
    span = spans["tool.execute"]
    truncated = span.attributes["stderr_truncated"]
    # Truncated to ~256 chars even though stderr was 5000.
    assert len(truncated) < 300
    assert "chars]" in truncated


# ---------------------------------------------------------------------------
# Stage 4: registry server event spans
# ---------------------------------------------------------------------------


def _make_registry_server(tmp_path):
    """Build an in-process registry MCP server with no KB.

    Mirrors the pattern used by tests/test_registry_server.py so we exercise
    the real call_tool dispatcher rather than reaching into private state.
    """
    from dsagt.commands.registry_server import create_registry_server
    from dsagt.registry import ToolRegistry

    source_dir = tmp_path / "source_skills"
    source_dir.mkdir()
    reg = ToolRegistry(
        source_tools_dir=str(source_dir),
        runtime_dir=str(tmp_path / "runtime"),
    )
    return create_registry_server(reg)


def _minimal_spec(name: str, **extras) -> dict:
    spec = {
        "name": name,
        "description": "test",
        "executable": "echo hi",
        "parameters": {"x": {"type": "string", "required": True, "description": "x"}},
    }
    spec.update(extras)
    return spec


def test_save_tool_spec_emits_registry_save_span(_reset_tracing, tmp_path):
    """save_tool_spec should produce a registry.save_tool_spec span with
    tool_name, language, n_dependencies, n_tags, action, and registry_size."""
    from mcp_helpers import call_tool_sync as call_tool

    exporter = _reset_tracing
    server = _make_registry_server(tmp_path)

    spec = _minimal_spec("alpha", language="python", tags=["genomics", "qc"])
    call_tool(server, "save_tool_spec", {"spec": spec})

    spans = _spans_by_name(exporter)
    assert "registry.save_tool_spec" in spans
    span = spans["registry.save_tool_spec"]
    assert span.attributes["tool_name"] == "alpha"
    assert span.attributes["language"] == "python"
    assert span.attributes["n_dependencies"] == 0
    assert span.attributes["n_tags"] == 2
    assert span.attributes["action"] == "added"
    assert span.attributes["registry_size"] == 1


def test_save_tool_spec_with_deps_nests_install_span(_reset_tracing, tmp_path, monkeypatch):
    """When a spec carries dependencies, save_tool_spec should open a
    nested registry.install_dependencies span as a child."""
    from mcp_helpers import call_tool_sync as call_tool

    # Stub the actual uv install so the test doesn't hit the network.
    import dsagt.commands.registry_server as rs_mod
    monkeypatch.setattr(
        rs_mod, "_install_dependencies",
        lambda packages, timeout=120: f"Successfully installed: {', '.join(packages)}",
    )

    exporter = _reset_tracing
    server = _make_registry_server(tmp_path)

    spec = _minimal_spec("beta", dependencies=["numpy", "pandas"])
    call_tool(server, "save_tool_spec", {"spec": spec})

    spans = _spans_by_name(exporter)
    save_span = spans["registry.save_tool_spec"]
    install_span = spans["registry.install_dependencies"]

    assert install_span.parent.span_id == save_span.context.span_id
    assert install_span.attributes["package_count"] == 2
    assert install_span.attributes["status"] == "ok"
    assert "numpy" in install_span.attributes["packages_preview"]


def test_install_dependencies_failed_records_event(_reset_tracing, tmp_path, monkeypatch):
    """A failing _install_dependencies should set status=failed and emit
    an install_failed event with the error message truncated."""
    from mcp_helpers import call_tool_sync as call_tool

    import dsagt.commands.registry_server as rs_mod
    monkeypatch.setattr(
        rs_mod, "_install_dependencies",
        lambda packages, timeout=120: "Installation failed (exit code 1):\nresolution failure",
    )

    exporter = _reset_tracing
    server = _make_registry_server(tmp_path)

    # First register a tool with deps so install_dependencies has something
    # to operate on, then call install_dependencies directly.
    spec = _minimal_spec("gamma", dependencies=["broken-package"])
    call_tool(server, "save_tool_spec", {"spec": spec})
    exporter.clear()
    call_tool(server, "install_dependencies", {})

    spans = _spans_by_name(exporter)
    assert "registry.install_dependencies" in spans
    span = spans["registry.install_dependencies"]
    assert span.attributes["package_count"] == 1
    assert span.attributes["status"] == "failed"
    assert any(e.name == "install_failed" for e in span.events)


def test_reconstruct_pipeline_emits_span(_reset_tracing, tmp_path):
    """reconstruct_pipeline should produce a span with format and output_chars."""
    from mcp_helpers import call_tool_sync as call_tool

    exporter = _reset_tracing
    server = _make_registry_server(tmp_path)

    # Empty trace_archive — reconstruct_pipeline should still emit a span,
    # even though the script body will be empty / minimal.
    (tmp_path / "runtime" / "trace_archive").mkdir(parents=True, exist_ok=True)

    call_tool(server, "reconstruct_pipeline", {"format": "bash"})

    spans = _spans_by_name(exporter)
    assert "registry.reconstruct_pipeline" in spans
    span = spans["registry.reconstruct_pipeline"]
    assert span.attributes["format"] == "bash"
    assert "output_chars" in span.attributes


def test_search_registry_does_not_emit_span(_reset_tracing, tmp_path):
    """High-frequency search calls are deliberately not instrumented."""
    from mcp_helpers import call_tool_sync as call_tool

    exporter = _reset_tracing
    server = _make_registry_server(tmp_path)
    call_tool(server, "search_registry", {"query": "anything"})

    spans = _spans_by_name(exporter)
    assert "registry.search" not in spans
    assert "registry.search_registry" not in spans
