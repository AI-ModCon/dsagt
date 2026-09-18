"""Tests for the merged ``dsagt-server`` (all four concern modules under one Server).

These verify the *composition* contract: every tool from the registry / knowledge
/ memory / skill modules is exposed under one MCP ``Server``, and the single
``call_tool`` wrapper preserves both return-type contracts (registry + skill
handlers may return a plain string; knowledge / memory handlers return a dict
that gets JSON-encoded).  Also covers ``_build_kb_from_config`` credential
validation in-process (the full subprocess boot needs a live MLflow — see
``test_server_startup.py``).
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import mcp.types as types
import pytest
from mcp_helpers import call_tool_sync

from dsagt.mcp.server import _build_kb_from_config, create_dsagt_server
from dsagt.registry import CodeRegistry, SkillRegistry


def _make_merged_server(tmp_path: Path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.collections = []
    runtime = str(tmp_path / "runtime")
    reg = CodeRegistry(runtime_dir=runtime, kb=None)
    reg.ensure_bundled_copies()
    sreg = SkillRegistry(runtime_dir=runtime, kb=None)
    return create_dsagt_server(reg, kb, sreg, runtime_dir=runtime)


def _list_tools(server) -> list[str]:
    handler = server.get_request_handler("tools/list").handler
    res = asyncio.run(handler(None, None))
    return sorted(t.name for t in res.tools)


def _call(server, name: str, arguments: dict) -> str:
    return call_tool_sync(server, name, arguments)


def test_merged_server_exposes_all_tools(tmp_path):
    """Both concern modules' tools land under one server with no collision."""
    server = _make_merged_server(tmp_path)
    names = _list_tools(server)
    # 8 registry + 5 knowledge + 2 memory + 5 skill = 20 distinct tools.
    assert set(names) == {
        # registry / provenance (8)
        "get_registry",
        "search_registry",
        "save_code_spec",
        "install_dependencies",
        "run_command",
        "read_file",
        "http_request",
        "reconstruct_pipeline",
        # knowledge (5)
        "kb_search",
        "kb_ingest",
        "kb_list_collections",
        "kb_job_status",
        "kb_append",
        # memory (2)
        "kb_remember",
        "kb_get_memories",
        # skills (5)
        "search_skills",
        "install_skill",
        "save_skill",
        "add_skill_source",
        "list_skill_sources",
    }
    assert len(set(names)) == len(names)  # no name collision


def test_dispatch_root_span_records_tool_inputs_and_outputs(tmp_path, monkeypatch):
    """The categorization-root span must carry the tool arguments as its inputs
    and the handler result as its outputs.

    Without this, every MCP tool trace shows a null Request and empty
    Inputs/Outputs in the MLflow UI, because the trace-level fields are read
    from the root span and the dispatch wrapper is the root.
    """
    import mlflow

    import dsagt.observability as obs_module
    from dsagt.mcp.server import build_dispatch_server

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")
    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", None)

    async def echo(args):
        return {"echoed": args["q"]}

    tools = [types.Tool(name="demo", description="d", inputSchema={"type": "object"})]
    server = build_dispatch_server("test", tools, {"demo": echo}, {"demo": "knowledge"})

    out = _call(server, "demo", {"q": "hello"})
    assert json.loads(out) == {"echoed": "hello"}

    trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
    root = next(s for s in trace.data.spans if s.name == "demo")
    assert root.inputs == {"q": "hello"}
    assert root.outputs == {"echoed": "hello"}


def test_dispatch_root_span_never_stores_credentials_or_payloads(tmp_path, monkeypatch):
    """What lands on the root span is written verbatim into ``mlflow.db`` and
    served by ``dsagt traces`` — so an ``http_request``-style ``headers`` arg
    must be redacted and a ``read_file``-sized result must be cut to a preview.
    """
    import mlflow

    import dsagt.observability as obs_module
    from dsagt.mcp.server import build_dispatch_server

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")
    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", None)

    async def fetch(args):
        return {"body": "x" * 100_000}

    tools = [types.Tool(name="fetch", description="d", inputSchema={"type": "object"})]
    server = build_dispatch_server(
        "test", tools, {"fetch": fetch}, {"fetch": "registry"}
    )

    out = _call(
        server,
        "fetch",
        {"url": "https://x", "headers": {"Authorization": "Bearer sk-secret"}},
    )
    assert (
        len(json.loads(out)["body"]) == 100_000
    )  # the agent still gets the full result

    trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
    root = next(s for s in trace.data.spans if s.name == "fetch")
    assert root.inputs["headers"] == "[redacted]"
    assert "sk-secret" not in json.dumps(root.inputs)
    assert len(root.outputs["body"]) < 5_000
    assert "[+" in root.outputs["body"]


def test_registry_tool_returns_plain_string(tmp_path):
    """Registry handlers return a bare string — passed through unchanged."""
    server = _make_merged_server(tmp_path)
    out = _call(server, "get_registry", {})
    # Not JSON — the registry contract is a human-readable string.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    assert "codes:" in out


def test_dict_returning_handler_is_json_encoded(tmp_path):
    """Dict-returning handlers (knowledge/memory/skill) are JSON-encoded by the wrapper."""
    server = _make_merged_server(tmp_path)
    out = _call(server, "list_skill_sources", {})
    parsed = json.loads(out)
    assert "sources" in parsed


class TestInputValidation:
    """The dispatch shell validates arguments against the tool's input schema.

    The mcp 2.x server invokes ``on_call_tool`` without validating arguments
    (and ``params.arguments`` is None when omitted), so the shell must reject
    malformed calls before they reach a handler.
    """

    def _server(self):
        from dsagt.mcp.server import build_dispatch_server

        async def echo(args):
            return {"echoed": args["q"]}

        tools = [
            types.Tool(
                name="demo",
                description="d",
                inputSchema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]
        return build_dispatch_server("test", tools, {"demo": echo})

    def test_missing_required_argument_rejected(self):
        out = json.loads(_call(self._server(), "demo", {}))
        assert out["status"] == "error"
        assert "Input validation error" in out["error"]
        assert "'q' is a required property" in out["error"]

    def test_omitted_arguments_rejected(self):
        server = self._server()
        handler = server.get_request_handler("tools/call").handler
        params = types.CallToolRequestParams(name="demo")  # arguments is None
        res = asyncio.run(handler(None, params))
        out = json.loads(res.content[0].text)
        assert out["status"] == "error"
        assert "'q' is a required property" in out["error"]

    def test_wrong_type_rejected(self):
        out = json.loads(_call(self._server(), "demo", {"q": 7}))
        assert out["status"] == "error"
        assert "Input validation error" in out["error"]

    def test_valid_arguments_dispatch(self):
        out = json.loads(_call(self._server(), "demo", {"q": "hello"}))
        assert out == {"echoed": "hello"}

    def _call_raw(self, server, name, arguments):
        handler = server.get_request_handler("tools/call").handler
        params = types.CallToolRequestParams(name=name, arguments=arguments)
        return asyncio.run(handler(None, params))

    def test_rejection_is_flagged_is_error(self):
        """``is_error`` is the only signal a client has that a call failed.

        Without it a rejected call renders as a successful tool result and the
        agent has nothing to correct itself from.
        """
        res = self._call_raw(self._server(), "demo", {})
        assert res.is_error is True

    def test_successful_call_is_not_flagged(self):
        res = self._call_raw(self._server(), "demo", {"q": "hello"})
        assert res.is_error is False

    def test_unknown_tool_is_rejected_not_raised(self):
        """The tool name is client-controlled, so an unknown one must come back
        as a readable rejection — an escaping KeyError becomes a JSON-RPC
        protocol error that tears down the request instead."""
        res = self._call_raw(self._server(), "no_such_tool", {})
        assert res.is_error is True
        out = json.loads(res.content[0].text)
        assert out["status"] == "error"
        assert "Unknown tool: no_such_tool" in out["error"]

    def test_unserializable_result_is_rejected_not_raised(self):
        """A handler returning non-JSON data must not escape as a protocol
        error either — ``json.dumps`` runs after the handler's own guard."""
        from dsagt.mcp.server import build_dispatch_server

        tools = [
            types.Tool(name="bad", description="d", inputSchema={"type": "object"})
        ]

        async def bad(args):
            return {"obj": object()}

        server = build_dispatch_server("test", tools, {"bad": bad})
        res = self._call_raw(server, "bad", {})
        assert res.is_error is True
        assert "Unserializable result" in json.loads(res.content[0].text)["error"]


class TestBuildKbFromConfig:
    """``_build_kb_from_config`` validates embedding config before building a KB.

    These raise paths fire before any embedder / ChromaDB construction, so they
    need no real backend.
    """

    def _cfg(self, **embedding):
        return {
            "embedding": embedding,
            "knowledge": {"chunk_size": 1024},
        }

    def test_invalid_backend_raises(self, tmp_path):
        cfg = self._cfg(backend="not-a-backend")
        with pytest.raises(ValueError, match="backend must be"):
            _build_kb_from_config(cfg, tmp_path)

    def test_api_backend_without_base_url_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EMBEDDING_API_KEY", "k")
        cfg = self._cfg(backend="api", model="m", base_url="")
        with pytest.raises(ValueError, match="requires embedding.base_url"):
            _build_kb_from_config(cfg, tmp_path)

    def test_api_backend_without_api_key_raises(self, tmp_path, monkeypatch):
        # Credentials come from the EMBEDDING_API_KEY env var, never on disk.
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        cfg = self._cfg(backend="api", model="m", base_url="http://x")
        with pytest.raises(ValueError, match="requires the EMBEDDING_API_KEY"):
            _build_kb_from_config(cfg, tmp_path)


def test_returned_tool_error_marks_the_trace_as_error(tmp_path, monkeypatch):
    """A handler that *returns* ``{"status": "error"}`` must produce an ERROR
    trace — otherwise a failed call is indistinguishable from a successful one
    in the store, and ``dsagt info`` reports ``Errors: 0`` after failures."""
    import mlflow

    import dsagt.observability as obs_module
    from dsagt.mcp.server import build_dispatch_server

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")
    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", None)

    async def boom(args):
        raise ValueError("nope")

    tools = [types.Tool(name="boom", description="d", inputSchema={"type": "object"})]
    server = build_dispatch_server("test", tools, {"boom": boom}, {"boom": "registry"})
    assert json.loads(_call(server, "boom", {}))["status"] == "error"

    trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
    assert str(trace.info.state).endswith("ERROR")


class TestPinTraceSource:
    """The trace-source token must be pinned as soon as *this* session's
    transcript exists — not on the first periodic pass (~50 s in), which a
    scripted session never reaches, and never to the previous session's
    transcript, which is what "newest file" resolves to before the agent's
    first message."""

    def _state(self, tmp_path, previous=None):
        from dsagt.session import append_session, record_trace_source

        (tmp_path / ".dsagt").mkdir(parents=True, exist_ok=True)
        append_session(tmp_path)  # the previous session
        if previous:
            record_trace_source(tmp_path, previous)
        append_session(tmp_path)  # this session
        return tmp_path

    def _run(self, collector, pdir, ticks=5):
        import asyncio

        from dsagt.mcp.server import _pin_trace_source

        async def go():
            await asyncio.wait_for(
                _pin_trace_source(collector, pdir, interval=0.01), timeout=1.0
            )

        asyncio.run(go())

    def test_skips_the_previous_sessions_transcript_and_pins_the_new_one(
        self, tmp_path, monkeypatch
    ):
        import os

        from dsagt.mcp import server as server_mod
        from dsagt.session import read_state

        old = tmp_path / "old.jsonl"
        old.write_text("{}\n")
        stale = os.path.getmtime(old) - 100
        os.utime(old, (stale, stale))  # written before this server started
        monkeypatch.setattr(server_mod, "_SERVER_STARTED_AT", stale + 50)
        pdir = self._state(tmp_path, previous=None)  # a too-short previous session

        new = tmp_path / "new.jsonl"
        seen = []

        class Collector:
            def active_source(self):
                seen.append(1)
                if len(seen) >= 3:  # the agent's first message lands the new file
                    new.write_text("{}\n")
                    return str(new)
                return str(old)

        self._run(Collector(), pdir)
        assert read_state(pdir)["sessions"][-1]["trace_source"] == str(new)

    def test_non_path_token_is_pinned_once_it_differs_from_the_previous(self, tmp_path):
        from dsagt.session import read_state

        pdir = self._state(tmp_path, previous="sess-old")
        seen = []

        class Collector:
            def active_source(self):
                seen.append(1)
                return "sess-old" if len(seen) < 3 else "sess-new"

        self._run(Collector(), pdir)
        assert read_state(pdir)["sessions"][-1]["trace_source"] == "sess-new"


def test_rejected_call_is_traced_as_an_error(tmp_path, monkeypatch):
    """A validation rejection must leave a trace — an agent looping on bad
    arguments is the case the debug view exists for — and still be flagged
    ``is_error`` on the wire."""
    import asyncio

    import mlflow

    import dsagt.observability as obs_module
    from dsagt.mcp.server import build_dispatch_server

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test")
    monkeypatch.setattr(obs_module, "_initialized", True)
    monkeypatch.setattr(obs_module, "_default_session_id", None)

    async def echo(args):
        return {"echoed": args["q"]}

    tools = [
        types.Tool(
            name="demo",
            description="d",
            inputSchema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
    ]
    server = build_dispatch_server("test", tools, {"demo": echo}, {"demo": "knowledge"})

    handler = server.get_request_handler("tools/call").handler
    res = asyncio.run(
        handler(None, types.CallToolRequestParams(name="demo", arguments={"q": 7}))
    )
    assert res.is_error is True

    trace = mlflow.MlflowClient().get_trace(mlflow.get_last_active_trace_id())
    assert str(trace.info.state).endswith("ERROR")
    assert trace.info.tags["dsagt.source"] == "knowledge"
    root = next(s for s in trace.data.spans if s.name == "demo")
    assert root.inputs == {"q": 7}
    assert "Input validation error" in root.outputs["error"]


class TestPinTraceSourceResume:
    def test_resumed_session_pins_the_previous_transcript_when_it_is_being_written(
        self, tmp_path, monkeypatch
    ):
        """``claude --resume`` keeps writing the previous session's file; a fresh
        mtime makes it this session's source even though the token repeats."""
        import os

        from dsagt.mcp import server as server_mod
        from dsagt.session import append_session, read_state, record_trace_source

        (tmp_path / ".dsagt").mkdir()
        append_session(tmp_path)
        same = tmp_path / "same.jsonl"
        same.write_text("{}\n")
        record_trace_source(tmp_path, str(same))
        append_session(tmp_path)
        monkeypatch.setattr(
            server_mod, "_SERVER_STARTED_AT", os.path.getmtime(same) - 1
        )

        class Collector:
            def active_source(self):
                return str(same)

        TestPinTraceSource()._run(Collector(), tmp_path)
        assert read_state(tmp_path)["sessions"][-1]["trace_source"] == str(same)


def test_reconstruct_pipeline_saves_to_a_project_path(tmp_path):
    """``output`` writes the script under the project and the reply names it."""
    server = _make_merged_server(tmp_path)
    trace_dir = tmp_path / "runtime" / "trace_archive"
    trace_dir.mkdir(parents=True)
    (trace_dir / "echo_r1.json").write_text(
        json.dumps(
            {
                "record_id": "r1",
                "code_name": "echo",
                "session_id": "s1",
                "execution": {
                    "exact_command": ["echo", "hi"],
                    "return_code": 0,
                    "stdout": "hi\n",
                    "stderr": "",
                    "timestamp_start": "2026-01-01T00:00:00Z",
                    "timestamp_end": "2026-01-01T00:00:01Z",
                    "input_files": [],
                    "output_files": [],
                },
            }
        )
    )
    out = _call(server, "reconstruct_pipeline", {"output": "audit/pipeline.sh"})
    assert out.startswith("Saved to audit/pipeline.sh")
    saved = (tmp_path / "runtime" / "audit" / "pipeline.sh").read_text()
    assert "echo hi" in saved
    outside = _call(server, "reconstruct_pipeline", {"output": "../escape.sh"})
    assert outside.startswith("Error: output must be a path under the project")
