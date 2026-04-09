"""
Consolidated integration tests for DSAGT.

Validates that real endpoints are reachable and that the config chain
produces working configurations. Requires test_site_config.yaml with
valid credentials — tests FAIL if missing or incomplete.

Usage:
    uv run pytest tests/test_integration.py -v
    uv run pytest tests/test_integration.py -v -k TestLLMEndpoint
    uv run pytest -m integration          # all integration tests across files
    uv run pytest -m "not integration"    # unit tests only
"""

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from dsagt.knowledge import APIEmbeddingClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Embedding endpoint
# ---------------------------------------------------------------------------

class TestEmbeddingEndpoint:
    """Validates the embedding API is reachable and returns valid vectors."""

    def test_embedding_returns_vector(self, embedding_config):
        """Single text produces a numpy array with positive dimension."""
        client = APIEmbeddingClient(
            model=embedding_config["model"],
            base_url=embedding_config["base_url"],
            api_key=embedding_config["api_key"],
        )
        try:
            result = client.embed(["test sentence"])
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert result.shape[0] == 1
            assert result.shape[1] > 0
        finally:
            client.close()

    def test_embedding_dimension_consistent(self, embedding_config):
        """Two different texts produce vectors of the same dimension."""
        client = APIEmbeddingClient(
            model=embedding_config["model"],
            base_url=embedding_config["base_url"],
            api_key=embedding_config["api_key"],
        )
        try:
            result = client.embed(["hello world", "genome assembly pipeline"])
            assert result.shape[0] == 2
            assert result.shape[1] > 0
        finally:
            client.close()

    def test_embedding_batch(self, embedding_config):
        """Five texts produce correct batch shape."""
        client = APIEmbeddingClient(
            model=embedding_config["model"],
            base_url=embedding_config["base_url"],
            api_key=embedding_config["api_key"],
        )
        texts = [
            "data quality assessment",
            "genome assembly metrics",
            "tool registration workflow",
            "semantic search retrieval",
            "pipeline provenance tracking",
        ]
        try:
            result = client.embed(texts)
            assert result.shape[0] == 5
        finally:
            client.close()


# ---------------------------------------------------------------------------
# LLM endpoint
# ---------------------------------------------------------------------------

class TestLLMEndpoint:
    """Validates the LLM API is reachable and returns valid responses."""

    def test_llm_returns_response(self, llm_config):
        """Minimal LLM request returns 200 with non-empty content."""
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{llm_config['base_url'].rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": llm_config["api_key"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": llm_config["model"],
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Say hello."}],
                },
            )
            response.raise_for_status()
            data = response.json()

        # Anthropic format: content[0].text
        texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        assert len(texts) > 0, f"No text content in response: {data}"
        assert len(texts[0]) > 0

    def test_llm_model_matches_config(self, llm_config):
        """Response model field matches the requested model."""
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{llm_config['base_url'].rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": llm_config["api_key"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": llm_config["model"],
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Say hello."}],
                },
            )
            response.raise_for_status()
            data = response.json()

        assert data.get("model") == llm_config["model"], (
            f"Expected model {llm_config['model']}, got {data.get('model')}"
        )


# ---------------------------------------------------------------------------
# MCP server handshake
# ---------------------------------------------------------------------------

_uv_available = shutil.which("uv") is not None


@pytest.mark.skipif(not _uv_available, reason="uv not available")
class TestMCPServerHandshake:
    """Validates MCP servers start and respond to handshakes with real config."""

    def test_registry_server_handshake(self, tmp_path):
        """Registry server starts and completes MCP handshake."""
        from mcp_helpers import mcp_initialize, mcp_list_tools, start_server

        proc = start_server([
            sys.executable, "-m", "dsagt.commands.registry_server",
            "--runtime-dir", str(tmp_path / "runtime"),
        ])
        try:
            resp = mcp_initialize(proc)
            assert "result" in resp

            tools_resp = mcp_list_tools(proc)
            tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
            assert "save_tool_spec" in tool_names
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_knowledge_server_handshake(self, tmp_path, embedding_config):
        """Knowledge server starts with real credentials and lists tools."""
        from mcp_helpers import mcp_initialize, mcp_list_tools, start_server

        proc = start_server(
            [
                sys.executable, "-m", "dsagt.commands.knowledge_server",
                "--base-index-dir", str(tmp_path / "kb_index"),
                "--runtime-dir", str(tmp_path / "runtime"),
            ],
            env={
                "LLM_API_KEY": embedding_config["api_key"],
                "OPENAI_BASE_URL": embedding_config["base_url"],
                "EMBEDDING_MODEL": embedding_config["model"],
            },
        )
        try:
            resp = mcp_initialize(proc)
            assert "result" in resp

            tools_resp = mcp_list_tools(proc)
            tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
            assert "kb_search" in tool_names
            assert "kb_ingest" in tool_names
        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Config chain
# ---------------------------------------------------------------------------

class TestConfigChain:
    """Validates the init -> config generation -> env var chain."""

    def test_init_generates_valid_config(self, tmp_path, monkeypatch):
        """init_project creates a valid dsagt_config.yaml."""
        import yaml
        from dsagt.session import init_project

        monkeypatch.setattr("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path)
        monkeypatch.setattr("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path)
        # Patch registry to store in memory
        reg = {}
        monkeypatch.setattr("dsagt.session._load_registry", lambda: dict(reg))
        monkeypatch.setattr("dsagt.session._save_registry", lambda r: reg.update(r))
        monkeypatch.setattr("dsagt.session.register_project", lambda n, p: reg.update({n: str(p)}))

        pdir = init_project("test-proj", "claude-code")

        config_path = pdir / "dsagt_config.yaml"
        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        assert config["project"] == "test-proj"
        assert config["agent"] == "claude-code"
        assert "llm" in config
        assert "embedding" in config

    def test_agent_configs_have_correct_ports(self, integration_config, tmp_path):
        """generate_agent_configs writes the correct proxy URL."""
        from dsagt.agents import generate_agent_configs

        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        generate_agent_configs(integration_config, working_dir)

        env_content = (working_dir / ".dsagt_env").read_text()
        proxy_port = integration_config["proxy"]["port"]
        assert f"localhost:{proxy_port}" in env_content

    def test_env_vars_chain_complete(self, integration_config):
        """agent_env() produces all critical environment variables."""
        from dsagt.agents import agent_env

        env = agent_env(integration_config)

        assert "DSAGT_PROJECT" in env
        assert "DSAGT_PROJECT_DIR" in env
        assert env["DSAGT_PROJECT"] == integration_config["project"]

        # Claude Code should get ANTHROPIC_BASE_URL
        if integration_config["agent"] == "claude-code":
            assert "ANTHROPIC_BASE_URL" in env
            proxy_port = integration_config["proxy"]["port"]
            assert str(proxy_port) in env["ANTHROPIC_BASE_URL"]

    def test_mcp_env_block_passes_api_key(self, integration_config):
        """Resolved API key appears in the MCP server env block."""
        from dsagt.agents import _mcp_env_block

        env_block = _mcp_env_block(integration_config)

        api_key = integration_config.get("embedding", {}).get("api_key", "")
        if api_key and not api_key.startswith("${"):
            assert env_block.get("LLM_API_KEY") == api_key


# ---------------------------------------------------------------------------
# Full stack: ingest and search with real embedding
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _uv_available, reason="uv not available")
class TestFullStack:
    """End-to-end: knowledge server subprocess with real embedding API."""

    def test_ingest_and_search(self, tmp_path, embedding_config):
        """Ingest docs via MCP, search, assert results."""
        from mcp_helpers import mcp_call_tool, mcp_initialize, start_server
        import time

        docs = tmp_path / "test_docs"
        docs.mkdir()
        (docs / "test.md").write_text(
            "# Integration Test\n\n"
            "This document validates the full DSAGT knowledge pipeline.\n"
        )

        proc = start_server(
            [
                sys.executable, "-m", "dsagt.commands.knowledge_server",
                "--base-index-dir", str(tmp_path / "kb_index"),
                "--runtime-dir", str(tmp_path / "runtime"),
            ],
            env={
                "LLM_API_KEY": embedding_config["api_key"],
                "OPENAI_BASE_URL": embedding_config["base_url"],
                "EMBEDDING_MODEL": embedding_config["model"],
            },
        )
        try:
            resp = mcp_initialize(proc)
            assert "result" in resp

            # Ingest
            ingest_resp = mcp_call_tool(proc, "kb_ingest", {
                "folder_path": str(docs),
            }, msg_id=10, timeout=60.0)
            ingest_data = json.loads(ingest_resp["result"]["content"][0]["text"])
            assert ingest_data["status"] == "started"
            job_id = ingest_data["job_id"]

            # Poll
            for i in range(60):
                status_resp = mcp_call_tool(
                    proc, "kb_job_status", {"job_id": job_id}, msg_id=20 + i,
                )
                status_data = json.loads(status_resp["result"]["content"][0]["text"])
                if status_data["status"] != "running":
                    break
                time.sleep(1)

            assert status_data["status"] == "complete", f"Ingest failed: {status_data}"

            # Search
            search_resp = mcp_call_tool(proc, "kb_search", {
                "query": "knowledge pipeline",
                "collection": "test_docs",
            }, msg_id=100, timeout=30.0)
            search_data = json.loads(search_resp["result"]["content"][0]["text"])
            assert search_data["status"] == "ok"
            assert search_data["result_count"] > 0
        finally:
            proc.terminate()
            proc.wait(timeout=5)
