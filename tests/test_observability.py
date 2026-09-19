"""
Tests for dsagt.observability.

These tests read spans back from the serverless MLflow trace store (a
per-test ``sqlite:///<tmp>/mlflow.db``); the live-span path uses
``mlflow.start_span`` directly.  They cover:

* init_tracing is a no-op outside a dsagt project dir
* init_tracing points MLflow at the resolved store + experiment
* @traced opens a span, captures args, sets duration_ms, records exceptions
* obs.set / obs.event are no-ops outside a span and write attributes inside
* child_span / typed helpers nest under the active span
* internal traces are tagged ``dsagt.source`` + ``mlflow.trace.session``

Each test gets its own fresh store, so the last trace in the store is the
operation under test.
"""

from __future__ import annotations

from contextlib import contextmanager

import os

import pytest

from dsagt import observability as obs_module
from dsagt.observability import experiment_name, child_span, init_tracing, obs, traced


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch, tmp_path):
    """Point MLflow at a fresh per-test sqlite store and mark tracing live.

    Each test gets its own store, so the last active trace is always the
    operation under test.
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
    """When cwd is not a dsagt project dir (no ``.dsagt/config.yaml`` with a
    ``project``), ``init_tracing`` logs and returns without raising; one-shot
    tools and tests outside a project run untraced.  The store needs no
    server, so the one reason to skip is "not in a project", which is not
    fatal."""
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
    from dsagt import __version__

    monkeypatch.setattr(obs_module, "_default_session_id", "proj-xyz")
    monkeypatch.setattr(obs_module, "_default_agent", "goose")

    with obs_module.open_span("search_knowledge", source="knowledge"):
        pass

    trace = _last_trace()
    assert trace.info.tags["dsagt.source"] == "knowledge"
    # agent + version are metadata on internal and agent traces alike, the
    # one place `dsagt info` reads them from.
    assert trace.info.trace_metadata["dsagt.agent"] == "goose"
    assert trace.info.trace_metadata["mlflow.trace.session"] == "proj-xyz"
    assert trace.info.trace_metadata["dsagt.version"] == __version__
    # The reserved key behind the trace table's User column.
    import getpass

    assert trace.info.trace_metadata["mlflow.trace.user"] == getpass.getuser()


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
    # Both the root and the kb.search child are in the one trace.
    names = {s.name for s in trace.data.spans}
    assert {"search_skills", "kb.search"} <= names


def test_uncategorized_span_has_no_source(_reset_tracing):
    """A span opened with no source (inner span outside any root, e.g. a
    background ``kb.*`` write) carries no ``dsagt.source``, so it does not
    appear in the debug-view filter as a miscategorized concern.
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
        return type("Exp", (), {"tags": {}})()  # fresh experiment: no tags yet

    def _fake_set_experiment_tag(key, value):
        captured.setdefault("tags", {})[key] = value

    def _fake_set_tracking_uri(uri):
        captured["tracking_uri"] = uri

    import mlflow

    monkeypatch.setattr(mlflow, "set_experiment", _fake_set_experiment)
    monkeypatch.setattr(mlflow, "set_experiment_tag", _fake_set_experiment_tag)
    monkeypatch.setattr(mlflow, "set_tracking_uri", _fake_set_tracking_uri)

    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(
        obs_module,
        "find_project_config",
        lambda: ("/proj", {"project": "my-project"}),
    )

    try:
        init_tracing("dsagt-run", mlflow_url="sqlite:///x.db")
        assert captured["tracking_uri"] == "sqlite:///x.db"
        # The experiment is the resolved hash name, not the project name; the
        # project name rides on the description and tag instead.
        assert captured["experiment"] == experiment_name({"project_dir": "/proj"})
        assert captured["experiment"].startswith("dsagt-")
        assert captured["tags"]["dsagt.project"] == "my-project"
        assert "DSAgt (DataSmith Agent)" in captured["tags"]["mlflow.note.content"]
        assert "my-project" in captured["tags"]["mlflow.note.content"]
        assert obs_module._initialized is True
    finally:
        monkeypatch.setattr(obs_module, "_initialized", False)


# ---------------------------------------------------------------------------
# Safety nets for the remaining defensive catches in observability.py.
#
# Two "soft" catches exist in the observability layer, both inline inside
# traced()'s wrapper:
#
#   1. traced() wraps sig.bind_partial in except TypeError so that a
#      function whose signature was mangled by another decorator does not
#      crash on every traced call.
#   2. traced() wraps each user-supplied extractor lambda in except
#      Exception so a buggy lambda does not crash the instrumented
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
    skip' catch cannot hide a regression where args stop being captured due to
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

    The mock patch is active for the whole context so cache misses on later
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

        results = kb.search("hello", collection="tcoll", top_k=2)
        assert isinstance(results, list)

    # The last trace is the search (add_entries is an earlier trace).
    spans = _spans_by_name()
    assert "kb.search" in spans
    assert "kb.embed" in spans
    assert "kb.index_search" in spans

    parent = spans["kb.search"]
    embed = spans["kb.embed"]
    index = spans["kb.index_search"]

    assert embed.parent_id == parent.span_id
    assert index.parent_id == parent.span_id

    # Captured args + obs.set('hits', ...) on the parent.
    assert parent.attributes["collection"] == "tcoll"
    assert parent.attributes["top_k"] == 2
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
        kb.search("hello", collection="tcoll", top_k=1)

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
    # A write child reports what it wrote, the way kb.search reports what it read.
    assert spans["kb.add_entries"].outputs == {"entries_added": 3}


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


def test_log_execution_trace_is_backdated_to_the_run(_reset_tracing):
    """The trace is logged after the run, from the record, and its span
    carries the run's own start and end times."""
    from dsagt.observability import log_execution_trace

    record = {
        "record_id": "abc123",
        "code_name": "fastp",
        "execution": {
            "exact_command": "fastp -i a.fq",
            "timestamp_start": "2026-01-01T00:00:00+00:00",
            "timestamp_end": "2026-01-01T00:00:42+00:00",
            "duration_ms": 42000.0,
            "return_code": 3,
            "stdout": "",
            "stderr": "boom",
            "input_files": ["a.fq"],
            "output_files": [],
        },
    }
    assert log_execution_trace(record) is not None

    span = _spans_by_name()["code.execute"]
    assert span.attributes["record_id"] == "abc123"
    assert span.attributes["code_name"] == "fastp"
    assert span.attributes["exit_code"] == 3
    assert span.attributes["n_input_files"] == 1
    assert span.end_time_ns - span.start_time_ns == 42 * 10**9
    assert span.status.status_code.name == "ERROR"
    assert [event.name for event in span.events] == ["code_failed"]


def test_mlflow_agent_hint_is_off_by_default():
    """Importing dsagt sets the mlflow hint switch, so dsagt-run under an
    agent prints no hint; a value the user exported is kept."""
    import os

    import dsagt  # noqa: F401  (the import is the effect under test)

    assert os.environ["MLFLOW_DISABLE_AGENT_HINT"] == "1"


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
    # Stderr was set by the FileNotFoundError branch: it is on the span.
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

    reg = CodeRegistry(runtime_dir=str(tmp_path / "runtime"))
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

    # Stub the uv install so the test does not reach the network.
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

    # Empty trace_archive: reconstruct_pipeline still emits a span, even
    # though the script body is empty or minimal.
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
    internal subsystem span: the trace is the root alone, tagged ``registry``,
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


def test_bound_redacts_credential_keys_at_any_depth():
    from dsagt.observability import bound

    args = {
        "url": "https://api.example.com",
        "headers": {"Authorization": "Bearer sk-live-abc"},
        "nested": {"api_key": "k", "keep": "v"},
    }
    out = bound(args)
    assert out["headers"] == "[redacted]"
    assert out["nested"]["api_key"] == "[redacted]"
    assert out["nested"]["keep"] == "v"
    assert out["url"] == "https://api.example.com"
    assert "sk-live-abc" not in str(out)


def test_bound_truncates_string_leaves_and_keeps_structure():
    from dsagt.observability import bound

    result = {"stdout": "x" * 10_000, "files": ["y" * 10_000, "short"], "code": 0}
    out = bound(result, limit=64)
    assert len(out["stdout"]) < 100 and "[+" in out["stdout"]
    assert len(out["files"][0]) < 100
    assert out["files"][1] == "short"
    assert out["code"] == 0  # non-strings pass through


def test_bound_handles_plain_string_result():
    """Registry handlers return a bare ``str``, not a dict."""
    from dsagt.observability import bound

    assert bound("short") == "short"
    assert "[+" in bound("z" * 10_000, limit=64)


def test_resolve_tracking_uri_env_overrides_sqlite(monkeypatch):
    from dsagt.observability import resolve_tracking_uri

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri({"project_dir": "/p"}) == "sqlite:////p/mlflow.db"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.org")
    assert resolve_tracking_uri({"project_dir": "/p"}) == "https://mlflow.example.org"


def test_api_key_header_provider_sends_x_api_key_only_when_set(monkeypatch):
    from importlib.metadata import entry_points

    from dsagt.observability import ApiKeyHeaderProvider

    # Registered where MLflow reads it, so every process picks it up.
    eps = {
        e.name: e.value for e in entry_points(group="mlflow.request_header_provider")
    }
    assert eps["dsagt_api_key"] == "dsagt.observability:ApiKeyHeaderProvider"

    p = ApiKeyHeaderProvider()
    monkeypatch.delenv("MLFLOW_TRACKING_API_KEY", raising=False)
    assert p.in_context() is False
    monkeypatch.setenv("MLFLOW_TRACKING_API_KEY", "k-123")
    assert p.in_context() is True
    assert p.request_headers() == {"X-API-Key": "k-123"}


def test_bound_masks_credential_shapes_inside_strings():
    """Key-name redaction cannot match a bearer inside a recorded argv or an
    API key in a URL query string; the value shape has to be masked."""
    from dsagt.observability import bound

    argv = {
        "command": [
            "curl",
            "-H",
            "Authorization: Bearer sk-live-abcdefghijklmnop",
            "https://x",
        ]
    }
    assert bound(argv)["command"][2] == "Authorization: Bearer [redacted]"

    url = bound({"url": "https://api.x/v1?api_key=sk-live-abcdefghijklmnop&page=2"})[
        "url"
    ]
    assert url == "https://api.x/v1?api_key=[redacted]&page=2"

    for key in ("X-API-Key", "access_token", "apikey", "auth"):
        assert bound({key: "sk-live-3"})[key] == "[redacted]"


def test_experiment_name_defaults_to_project_dir_hash_and_honors_config():
    from dsagt.observability import experiment_name

    a = experiment_name({"project_dir": "/home/a/dsagt-projects/demo"})
    b = experiment_name({"project_dir": "/home/b/dsagt-projects/demo"})
    assert a.startswith("dsagt-") and len(a) == len("dsagt-") + 8
    assert a == experiment_name(
        {"project_dir": "/home/a/dsagt-projects/demo"}
    )  # stable
    assert a != b  # same project name, different users: no collision on a shared server
    assert (
        experiment_name({"project_dir": "/x", "mlflow": {"experiment": "team/demo"}})
        == "team/demo"
    )


def test_ensure_experiment_tags_only_on_first_creation(monkeypatch):
    """A description edited by hand on the server must not be overwritten on
    every periodic pass: tags are written only when the experiment has none."""
    import mlflow

    from dsagt.observability import _ensure_experiment

    calls = []
    existing = type(
        "Exp", (), {"tags": {"dsagt.project": "p", "mlflow.note.content": "edited"}}
    )()
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: existing)
    monkeypatch.setattr(mlflow, "set_experiment_tag", lambda k, v: calls.append(k))
    _ensure_experiment("dsagt-abc", "p")
    assert calls == []


def test_code_execute_nonzero_exit_is_an_error_trace(_reset_tracing, tmp_path):
    """A failed code run is a failure in the store, not an OK span with an
    event tucked inside it."""
    import mlflow

    from dsagt.provenance import run_and_record

    run_and_record(code_name="t", command=["false"], records_dir=tmp_path)
    trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
    assert str(trace.info.state).endswith("ERROR")


def test_init_tracing_survives_a_deleted_experiment(tmp_path, monkeypatch, caplog):
    """A deleted experiment on the store must not take the server down: the
    name is deterministic, so `set_experiment` would refuse it on every start
    and the project could never run again.  Tracing is turned off and the
    failure is logged at ERROR."""
    import logging

    import mlflow

    from dsagt.observability import experiment_name, init_tracing

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    name = experiment_name({"project_dir": "/proj"})
    mlflow.MlflowClient().delete_experiment(mlflow.create_experiment(name))

    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(
        obs_module, "find_project_config", lambda: ("/proj", {"project": "p"})
    )

    with caplog.at_level(logging.ERROR):
        init_tracing("dsagt-server", mlflow_url=uri)  # must not raise

    assert obs_module._initialized is False
    msg = caplog.text
    assert (
        "tracing disabled" in msg
        and "deleted state" in msg
        and "mlflow.experiment" in msg
    )


def test_init_tracing_activates_the_version_model(tmp_path, monkeypatch):
    """`mlflow.modelId` must reference a LoggedModel named for the dsagt
    release, created once per experiment; that is what the UI's Version
    column shows."""
    import mlflow

    from dsagt import __version__
    from dsagt.observability import init_tracing

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(
        obs_module, "find_project_config", lambda: ("/proj", {"project": "p"})
    )
    init_tracing("dsagt-server", mlflow_url=uri)
    with obs_module.open_span("demo", source="knowledge"):
        pass
    md = _last_trace().info.trace_metadata
    model = mlflow.get_logged_model(md["mlflow.modelId"])
    assert model.name == "dsagt-" + __version__.replace(".", "_")


def test_bound_leaves_ordinary_prose_alone_and_catches_json_keys():
    """The value-shape sweep is anchored: `Bearer`/`Basic` only after an
    `Authorization:` label, key labels only before a token-shaped value;
    otherwise a recorded document with "basic " or "bearer " in it
    loses the next word in the stored preview.  JSON-quoted keys, the shape of
    a printed config, are caught."""
    from dsagt.observability import bound

    prose = "A basic example of the bearer of bad news; max_token: 5 items"
    assert bound(prose) == prose
    assert (
        bound({"cfg": '{"api_key": "sk-live-abcdefghijklmnop"}'})["cfg"]
        == '{"api_key": "[redacted]"}'
    )
    assert (
        bound("OPENAI_API_KEY=sk-live-abcdefghijklmnop") == "OPENAI_API_KEY=[redacted]"
    )
    assert bound("token=abc") == "token=abc"  # too short to be a credential


def test_remote_store_retry_budget_is_bounded_but_overridable(monkeypatch):
    from dsagt.observability import _bound_remote_retries

    for v in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
        monkeypatch.delenv(v, raising=False)
    _bound_remote_retries("sqlite:///x.db")
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ  # local: untouched
    _bound_remote_retries("https://mlflow.example.org")
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "9")
    _bound_remote_retries("https://mlflow.example.org")
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"  # explicit wins


def test_init_tracing_quiets_mlflow_info_chatter(tmp_path, monkeypatch, capsys):
    """MLflow logs set_experiment / set_active_model at INFO on stderr
    ("Active model is set to …") on every dsagt-run.  An agent capturing a
    code's stderr would read that as the code's output."""
    import logging

    from dsagt.observability import init_tracing

    monkeypatch.setattr(obs_module, "_initialized", False)
    monkeypatch.setattr(
        obs_module, "find_project_config", lambda: ("/proj", {"project": "p"})
    )
    init_tracing("dsagt-run", mlflow_url=f"sqlite:///{tmp_path}/mlflow.db")
    assert logging.getLogger("mlflow.tracking.fluent").level == logging.WARNING
    assert "Active model is set" not in capsys.readouterr().err
