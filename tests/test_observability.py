"""
Tests for dsagt.observability — Stage 0.

These tests read spans back from the serverless MLflow trace store (a
per-test ``sqlite:///<tmp>/mlflow.db``) rather than from OTel's
InMemorySpanExporter — the live-span path now uses ``mlflow.start_span``
directly and installs no OTel TracerProvider.  They cover:

* init_tracing is a no-op outside a dsagt project dir
* init_tracing points MLflow at the resolved store + experiment
* @traced opens a span, captures args, sets duration_ms, records exceptions
* obs.set / obs.event are no-ops outside a span and write attributes inside
* child_span / typed helpers nest under the active span
* internal traces are tagged ``dsagt.source`` + ``mlflow.trace.session``

Each test gets its own fresh store, so the LAST trace in the store is the
operation under test — no clear-then-read dance is needed.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from dsagt import observability as obs_module
from dsagt.observability import child_span, init_tracing, obs, traced


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch, tmp_path):
    """Point MLflow at a fresh per-test sqlite store and mark tracing live.

    Each test gets its own store, so reading the LAST active trace always
    yields the operation under test — no exporter to clear between calls.
    """
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")

    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", None)

    yield


def _last_trace():
    import mlflow
    from mlflow import MlflowClient

    return MlflowClient().get_trace(mlflow.get_last_active_trace_id())


def _spans_by_name(_ignored=None):
    return {s.name: s for s in _last_trace().data.spans}


def test_init_tracing_outside_project_is_noop(monkeypatch):
    """Serverless + never-raise: when cwd isn't a dsagt project dir (no
    ``.dsagt/config.yaml`` with a ``project``), ``init_tracing`` logs and
    no-ops rather than raising — one-shot tools / tests outside a project
    simply run untraced.  The store itself never needs a server, so the
    only reason to skip is "not in a project", which must not be fatal."""
    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    # Repo root has no .dsagt/config.yaml → find_project_config returns None.
    init_tracing("test-service")  # must not raise
    assert obs_module._initialized is False


def test_traced_emits_span_with_args(_reset_tracing):
    @traced("test.op", capture=["a", "b"])
    def f(a, b, c=10):
        return a + b + c

    result = f(1, 2, c=3)
    assert result == 6

    spans = _spans_by_name()
    assert "test.op" in spans
    span = spans["test.op"]
    assert span.attributes["a"] == 1
    assert span.attributes["b"] == 2
    assert "c" not in span.attributes  # not in capture list
    assert "duration_ms" in span.attributes


def test_traced_extract_return(_reset_tracing):
    @traced("test.op", extract_return={"hits": len})
    def search():
        return ["a", "b", "c"]

    search()
    span = _spans_by_name()["test.op"]
    assert span.attributes["hits"] == 3


def test_traced_records_exception(_reset_tracing):
    @traced("test.boom")
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()

    span = _spans_by_name()["test.boom"]
    assert span.status.status_code.name == "ERROR"
    # exception is recorded as a span event (MLflow auto-records it)
    assert any(e.name == "exception" for e in span.events)
    # duration is still set even on error path
    assert "duration_ms" in span.attributes


def test_obs_set_outside_span_is_noop():
    """obs.set must not raise when called with no active span."""
    obs.set("nothing", 1)  # must be silent
    obs.set_many({"a": 1, "b": 2})
    obs.event("ping", x=1)


def test_obs_set_inside_span_writes_attribute(_reset_tracing):
    @traced("test.op")
    def f():
        obs.set("hits", 7)
        obs.set_many({"foo": "bar", "skipped": None})
        obs.event("milestone", phase="middle")
        return None

    f()
    span = _spans_by_name()["test.op"]
    assert span.attributes["hits"] == 7
    assert span.attributes["foo"] == "bar"
    assert "skipped" not in span.attributes
    assert any(e.name == "milestone" for e in span.events)


def test_child_span_nests(_reset_tracing):
    @traced("test.parent")
    def parent():
        with child_span("test.child", phase="embed"):
            pass

    parent()
    spans = _spans_by_name()
    child = spans["test.child"]
    parent_span = spans["test.parent"]
    assert child.parent_id is not None
    assert child.parent_id == parent_span.span_id
    assert child.attributes["phase"] == "embed"


def test_root_span_source_tags_trace_and_session(_reset_tracing, monkeypatch):
    """A categorization root (``open_span(source=...)``) tags the trace's
    ``dsagt.source`` and, when a session id is set, the reserved
    ``mlflow.trace.session`` metadata key (the native session filter).
    """
    monkeypatch.setattr(obs_module, "_default_session_id", "proj-xyz")

    with obs_module.open_span("search_knowledge", source="knowledge"):
        pass

    trace = _last_trace()
    assert trace.info.tags["dsagt.source"] == "knowledge"
    assert trace.info.trace_metadata["mlflow.trace.session"] == "proj-xyz"


def test_inner_spans_inherit_root_source(_reset_tracing):
    """Source is set at the entry point, not derived from the span name: a
    child span opened under a ``skill`` root inherits ``skill`` even when the
    child is a ``kb.*`` (knowledge-subsystem) span.  This is what makes
    ``search_skills`` → ``kb.search`` tag as ``skill``, not ``knowledge``.
    """
    with obs_module.open_span("search_skills", source="skill"):

        @traced("kb.search")
        def inner():
            pass

        inner()

    trace = _last_trace()
    assert trace.info.tags["dsagt.source"] == "skill"
    # Both the root and the kb.search child live in the one trace.
    names = {s.name for s in trace.data.spans}
    assert {"search_skills", "kb.search"} <= names


def test_uncategorized_span_has_no_source(_reset_tracing):
    """A span opened with no source (inner span outside any root, e.g. a
    background ``kb.*`` write) carries no ``dsagt.source`` — it doesn't leak
    into the debug-view filter as a miscategorized concern.
    """

    @traced("kb.add_entries")
    def f():
        pass

    f()
    assert "dsagt.source" not in _last_trace().info.tags


def test_init_tracing_double_call_only_updates_session(monkeypatch):
    """Re-calling init_tracing should not raise; should update session id."""
    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", "old")
    init_tracing("test", session_id="new")
    assert obs_module._default_session_id == "new"


def test_init_tracing_points_mlflow_at_store_and_experiment(monkeypatch):
    """init_tracing resolves the project name from ``.dsagt/config.yaml``,
    points MLflow's tracking URI at the passed store, sets the experiment to
    the project name, and flips ``_initialized`` true.
    """
    captured: dict = {}

    def _fake_set_experiment(name):
        captured["experiment"] = name

    def _fake_set_tracking_uri(uri):
        captured["tracking_uri"] = uri

    import mlflow

    monkeypatch.setattr(mlflow, "set_experiment", _fake_set_experiment)
    monkeypatch.setattr(mlflow, "set_tracking_uri", _fake_set_tracking_uri)

    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(
        obs_module,
        "find_project_config",
        lambda: (None, {"project": "my-project"}),
    )

    try:
        init_tracing("dsagt-run", mlflow_url="sqlite:///x.db")
        assert captured["tracking_uri"] == "sqlite:///x.db"
        assert captured["experiment"] == "my-project"
        assert obs_module._initialized is True
    finally:
        monkeypatch.setattr(obs_module, "_initialized", False)


# ---------------------------------------------------------------------------
# Safety nets for the remaining defensive catches in observability.py.
#
# After the fallback-purge pass, two "soft" catches remain in the
# observability layer, both inline inside traced()'s wrapper (previously
# _attach_captured_args and _attach_return_attrs):
#
#   1. traced() wraps sig.bind_partial in except TypeError so that a
#      function whose signature was mangled by another decorator doesn't
#      crash on every traced call.
#   2. traced() wraps each user-supplied extractor lambda in except
#      Exception so a buggy lambda doesn't crash the instrumented
#      function.
#
# These tests pin the HAPPY PATH so that if those catches ever fire in
# normal use the test suite fails immediately.  Without them, the
# silent-degradation behavior of the catches would hide real bugs.
# ---------------------------------------------------------------------------


def test_extract_return_failure_logs_at_debug_and_does_not_crash(
    _reset_tracing, caplog
):
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
    span = _spans_by_name()["test.extractor_bug"]
    assert span.attributes["good"] == 3
    assert span.attributes["another_good"] == "a"
    assert "bad" not in span.attributes

    # The failure was logged at DEBUG with the attribute name.  This is
    # what makes the silent skip visible to a developer running with
    # --verbose / DEBUG logging.
    debug_messages = [r.message for r in caplog.records if r.levelno == _logging.DEBUG]
    assert any("extract_return['bad']" in m for m in debug_messages), (
        f"Expected a DEBUG log mentioning the broken 'bad' extractor, "
        f"got: {debug_messages}"
    )


def test_attach_captured_args_happy_path_protects_bind_partial_catch(_reset_tracing):
    """Pin the happy path for arg capture so the silent 'except TypeError:
    skip' catch can't hide a regression where args stop being captured due to
    a signature-introspection bug.

    If sig.bind_partial silently failed for any reason, this test would
    fail because the captured 'a' and 'b' attributes would be missing
    from the emitted span.
    """

    @traced("test.signature_capture", capture=["a", "b", "c"])
    def f(a, b, c=42, *, d=None):
        return None

    f(1, b=2, d="ignored")
    span = _spans_by_name()["test.signature_capture"]
    assert span.attributes["a"] == 1
    assert span.attributes["b"] == 2
    assert span.attributes["c"] == 42  # default value still captured
    assert "d" not in span.attributes


# ---------------------------------------------------------------------------
# Stage 1: KnowledgeBase instrumentation
# ---------------------------------------------------------------------------


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

    with patch("dsagt.knowledge.Embedder.create", return_value=mock_client):
        kb = KnowledgeBase(
            index_dir=tmp_path / f"kb_{backend}",
            default_embedder=backend,
            model=model,
        )
        try:
            yield kb
        finally:
            kb.close()


def test_kb_search_emits_three_child_spans(_reset_tracing, tmp_path):
    """kb.search should produce kb.search → {kb.embed, kb.index_search}."""
    with _kb_with_mocked_embedder(tmp_path) as kb:
        # Seed a collection so search has something to load.
        kb.add_entries(texts=["hello world", "goodbye"], collection="tcoll")

        results = kb.search("hello", collection="tcoll", top_k=2, rerank=False)
        assert isinstance(results, list)

    # The last trace is the search (add_entries is an earlier trace).
    spans = _spans_by_name()
    assert "kb.search" in spans
    assert "kb.embed" in spans
    assert "kb.index_search" in spans
    # No rerank requested, so no rerank span.
    assert "kb.rerank" not in spans

    parent = spans["kb.search"]
    embed = spans["kb.embed"]
    index = spans["kb.index_search"]

    assert embed.parent_id == parent.span_id
    assert index.parent_id == parent.span_id

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
    with _kb_with_mocked_embedder(tmp_path, backend="local", model="bge-base") as kb:
        kb.add_entries(texts=["hello world"], collection="tcoll")
        kb.search("hello", collection="tcoll", top_k=1, rerank=False)

    spans = _spans_by_name()
    assert "kb.search" in spans
    assert "kb.embed" in spans
    assert spans["kb.embed"].attributes["backend"] == "local"
    assert spans["kb.embed"].attributes["model"] == "bge-base"


def test_kb_ingest_emits_embed_child(_reset_tracing, tmp_path):
    """kb.ingest opens an outer span and one child kb.embed span."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.txt").write_text("alpha beta gamma")
    (src / "b.txt").write_text("delta epsilon zeta")

    with _kb_with_mocked_embedder(tmp_path) as kb:
        kb.ingest(src)

    spans = _spans_by_name()
    assert "kb.ingest" in spans
    assert "kb.embed" in spans

    ingest = spans["kb.ingest"]
    embed = spans["kb.embed"]
    assert embed.parent_id == ingest.span_id
    assert ingest.attributes["n_files"] == 2
    assert ingest.attributes["n_chunks"] >= 2
    assert embed.attributes["n_texts"] == ingest.attributes["n_chunks"]


def test_kb_add_entries_emits_span(_reset_tracing, tmp_path):
    """kb.add_entries should emit a top-level span with n_entries."""
    with _kb_with_mocked_embedder(tmp_path) as kb:
        kb.add_entries(texts=["one", "two", "three"], collection="epis")

    spans = _spans_by_name()
    assert "kb.add_entries" in spans
    assert spans["kb.add_entries"].attributes["collection"] == "epis"
    assert spans["kb.add_entries"].attributes["n_entries"] == 3


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


def test_code_execute_span_attributes(_reset_tracing):
    """code_execute_span sets record_id and code_name on the span."""
    from dsagt.observability import obs, code_execute_span

    with code_execute_span(record_id="abc123", code_name="fastp"):
        obs.set("exit_code", 0)
        obs.set("duration_ms", 42.5)

    spans = _spans_by_name()
    assert "code.execute" in spans
    span = spans["code.execute"]
    assert span.attributes["record_id"] == "abc123"
    assert span.attributes["code_name"] == "fastp"
    assert span.attributes["exit_code"] == 0
    assert span.attributes["duration_ms"] == 42.5


def test_run_and_record_emits_code_execute_span(_reset_tracing, tmp_path):
    """run_and_record() should produce a tool.execute span with the
    expected execution attributes."""
    from dsagt.provenance import run_and_record

    rc = run_and_record(
        code_name="echo",
        command=["echo", "hello world"],
        records_dir=tmp_path / "records",
        session_id="test-session",
        record_id="rec-001",
        input_files=["in.txt"],
        output_files=["out.txt", "out2.txt"],
    )
    assert rc == 0

    spans = _spans_by_name()
    assert "code.execute" in spans
    span = spans["code.execute"]

    assert span.attributes["record_id"] == "rec-001"
    assert span.attributes["code_name"] == "echo"
    assert span.attributes["exit_code"] == 0
    assert span.attributes["duration_ms"] >= 0
    assert span.attributes["n_input_files"] == 1
    assert span.attributes["n_output_files"] == 2
    assert span.attributes["command"].startswith("echo")
    assert span.attributes["stdout_len"] > 0


def test_run_and_record_failed_tool_records_event_and_status(_reset_tracing, tmp_path):
    """A failed tool call should emit a code_failed event with the exit code
    and the truncated stderr should be attached."""
    from dsagt.provenance import run_and_record

    rc = run_and_record(
        code_name="missing_tool",
        command=["this-binary-does-not-exist-anywhere"],
        records_dir=tmp_path / "records",
        session_id="test-session",
    )
    assert rc == 127

    span = _spans_by_name()["code.execute"]

    assert span.attributes["exit_code"] == 127
    # Stderr was set by the FileNotFoundError branch — should be on the span.
    assert "stderr_truncated" in span.attributes
    # code_failed event was added.
    assert any(e.name == "code_failed" for e in span.events)


def test_run_and_record_long_stderr_is_truncated(_reset_tracing, tmp_path):
    """Span attributes should never carry multi-megabyte stderr blobs."""
    from dsagt.provenance import run_and_record

    # Use python -c to emit a large stderr deterministically.
    big = "x" * 5000
    rc = run_and_record(
        code_name="echo_err",
        command=["python", "-c", f"import sys; sys.stderr.write('{big}'); sys.exit(0)"],
        records_dir=tmp_path / "records",
        session_id="s",
    )
    assert rc == 0

    span = _spans_by_name()["code.execute"]
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
    from dsagt.mcp.registry_tools import create_registry_server
    from dsagt.registry import CodeRegistry

    source_dir = tmp_path / "source_skills"
    source_dir.mkdir()
    reg = CodeRegistry(
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


def test_save_code_spec_emits_registry_save_span(_reset_tracing, tmp_path):
    """save_code_spec should produce a registry.save_code_spec span with
    code_name, language, n_dependencies, n_tags, action, and registry_size."""
    from mcp_helpers import call_tool_sync as call_tool

    server = _make_registry_server(tmp_path)

    spec = _minimal_spec("alpha", language="python", tags=["genomics", "qc"])
    call_tool(server, "save_code_spec", {"spec": spec})

    spans = _spans_by_name()
    assert "registry.save_code_spec" in spans
    span = spans["registry.save_code_spec"]
    assert span.attributes["code_name"] == "alpha"
    assert span.attributes["language"] == "python"
    assert span.attributes["n_dependencies"] == 0
    assert span.attributes["n_tags"] == 2
    assert span.attributes["action"] == "added"
    assert span.attributes["registry_size"] == 1
    # The dispatch root tags the whole trace with the tool's concern category.
    assert _last_trace().info.tags["dsagt.source"] == "registry"


def test_save_code_spec_with_deps_nests_install_span(
    _reset_tracing, tmp_path, monkeypatch
):
    """When a spec carries dependencies, save_code_spec should open a
    nested registry.install_dependencies span as a child."""
    from mcp_helpers import call_tool_sync as call_tool

    # Stub the actual uv install so the test doesn't hit the network.
    import dsagt.mcp.registry_tools as rs_mod

    monkeypatch.setattr(
        rs_mod,
        "_install_dependencies",
        lambda packages, timeout=120: f"Successfully installed: {', '.join(packages)}",
    )

    server = _make_registry_server(tmp_path)

    spec = _minimal_spec("beta", dependencies=["numpy", "pandas"])
    call_tool(server, "save_code_spec", {"spec": spec})

    spans = _spans_by_name()
    save_span = spans["registry.save_code_spec"]
    install_span = spans["registry.install_dependencies"]

    assert install_span.parent_id == save_span.span_id
    assert install_span.attributes["package_count"] == 2
    assert install_span.attributes["status"] == "ok"
    assert "numpy" in install_span.attributes["packages_preview"]


def test_install_dependencies_failed_records_event(
    _reset_tracing, tmp_path, monkeypatch
):
    """A failing _install_dependencies should set status=failed and emit
    an install_failed event with the error message truncated."""
    from mcp_helpers import call_tool_sync as call_tool

    import dsagt.mcp.registry_tools as rs_mod

    monkeypatch.setattr(
        rs_mod,
        "_install_dependencies",
        lambda packages, timeout=120: "Installation failed (exit code 1):\nresolution failure",
    )

    server = _make_registry_server(tmp_path)

    # First register a tool with deps so install_dependencies has something
    # to operate on, then call install_dependencies directly.
    spec = _minimal_spec("gamma", dependencies=["broken-package"])
    call_tool(server, "save_code_spec", {"spec": spec})
    call_tool(server, "install_dependencies", {})

    # The last trace is the install_dependencies call.
    spans = _spans_by_name()
    assert "registry.install_dependencies" in spans
    span = spans["registry.install_dependencies"]
    assert span.attributes["package_count"] == 1
    assert span.attributes["status"] == "failed"
    assert any(e.name == "install_failed" for e in span.events)


def test_reconstruct_pipeline_emits_span(_reset_tracing, tmp_path):
    """reconstruct_pipeline should produce a span with format and output_chars."""
    from mcp_helpers import call_tool_sync as call_tool

    server = _make_registry_server(tmp_path)

    # Empty trace_archive — reconstruct_pipeline should still emit a span,
    # even though the script body will be empty / minimal.
    (tmp_path / "runtime" / "trace_archive").mkdir(parents=True, exist_ok=True)

    call_tool(server, "reconstruct_pipeline", {"format": "bash"})

    spans = _spans_by_name()
    assert "registry.reconstruct_pipeline" in spans
    span = spans["registry.reconstruct_pipeline"]
    assert span.attributes["format"] == "bash"
    assert "output_chars" in span.attributes


def test_search_registry_categorized_but_no_internal_span(_reset_tracing, tmp_path):
    """Every MCP call gets a categorized dispatch root span (so the concern
    shows up in the debug view), but high-frequency search still adds no
    internal subsystem span — the trace is just the root, tagged ``registry``,
    with no ``registry.*`` child."""
    from mcp_helpers import call_tool_sync as call_tool

    server = _make_registry_server(tmp_path)
    call_tool(server, "search_registry", {"query": "anything"})

    trace = _last_trace()
    names = {s.name for s in trace.data.spans}
    # The dispatch root exists and is categorized...
    assert "search_registry" in names
    assert trace.info.tags["dsagt.source"] == "registry"
    # ...but search opens no internal subsystem span.
    assert not any(n.startswith("registry.") for n in names)
