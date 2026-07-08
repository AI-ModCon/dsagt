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

from dsagt.mcp.server import _build_kb_from_config, create_dsagt_server
from dsagt.registry import SkillRegistry, CodeRegistry


def _make_merged_server(tmp_path: Path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.default_rerank = True
    kb.collections = []
    runtime = str(tmp_path / "runtime")
    reg = CodeRegistry(runtime_dir=runtime, kb=None)
    reg.ensure_bundled_copies()
    sreg = SkillRegistry(source_skills_dir=None, runtime_dir=runtime, kb=None)
    return create_dsagt_server(reg, kb, sreg, runtime_dir=runtime)


def _list_tools(server) -> list[str]:
    handler = server.request_handlers[types.ListToolsRequest]
    res = asyncio.run(handler(types.ListToolsRequest(method="tools/list")))
    return sorted(t.name for t in res.root.tools)


def _call(server, name: str, arguments: dict) -> str:
    handler = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    res = asyncio.run(handler(req))
    return res.root.content[0].text


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


class TestBuildKbFromConfig:
    """``_build_kb_from_config`` validates embedding config before building a KB.

    These raise paths fire before any embedder / ChromaDB construction, so they
    need no real backend.
    """

    def _cfg(self, **embedding):
        return {
            "embedding": embedding,
            "knowledge": {"chunk_size": 1024, "rerank": False},
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
