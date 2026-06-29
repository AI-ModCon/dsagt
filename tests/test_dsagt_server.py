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
from dsagt.registry import SkillRegistry, ToolRegistry


def _make_merged_server(tmp_path: Path):
    kb = MagicMock()
    kb.index_dir = tmp_path / "kb_index"
    kb.index_dir.mkdir()
    kb.default_rerank = True
    kb.collections = []
    runtime = str(tmp_path / "runtime")
    reg = ToolRegistry(source_tools_dir=None, runtime_dir=runtime, kb=None)
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
    # 8 registry + 5 knowledge + 4 memory + 5 skill = 22 distinct tools.
    # (kb_add_vector_db is gated off until BYO lands — see knowledge_tools.py.)
    assert len(names) == 22
    assert len(set(names)) == len(names)  # no name collision
    # Representative tools from each side.
    for expected in (
        "get_registry",
        "search_skills",
        "kb_search",
        "list_skill_sources",
    ):
        assert expected in names


def test_registry_tool_returns_plain_string(tmp_path):
    """Registry handlers return a bare string — passed through unchanged."""
    server = _make_merged_server(tmp_path)
    out = _call(server, "get_registry", {})
    # Not JSON — the registry contract is a human-readable string.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    assert "tools:" in out


def test_knowledge_tool_returns_json(tmp_path):
    """Knowledge handlers return a dict — JSON-encoded by the wrapper."""
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
