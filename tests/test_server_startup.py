"""
Process-level MCP server tests.

Two categories:

1. **Startup tests** — verify each server starts and completes the MCP
   handshake. No API key or network access needed.

2. **Transport-closed reproduction tests** — exercise the exact agent flow
   (ingest -> search) that triggers 'transport closed'. Require valid
   credentials in test_site_config.yaml.

Usage:
    # Startup tests only (no config needed):
    uv run pytest test_server_startup.py -v -k 'not Search'

    # Full suite including embedding API tests:
    uv run pytest test_server_startup.py -v
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from mcp_helpers import (
    mcp_call_tool,
    mcp_initialize,
    mcp_list_tools,
    start_server,
)


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_uv_available = shutil.which("uv") is not None

pytestmark = pytest.mark.skipif(not _uv_available, reason="uv not available")


# ---------------------------------------------------------------------------
# Startup tests (no API key needed)
# ---------------------------------------------------------------------------

class TestRegistryServerStartup:

    def test_starts_and_lists_tools(self, tmp_path):
        """Registry server starts, completes MCP handshake, and lists tools."""
        proc = start_server([
            sys.executable, "-m", "dsagt.commands.registry_server",
            "--runtime-dir", str(tmp_path / "runtime"),
        ])
        try:
            resp = mcp_initialize(proc)
            assert "result" in resp, f"Init failed: {resp}"

            tools_resp = mcp_list_tools(proc)
            assert "result" in tools_resp, f"list_tools failed: {tools_resp}"

            tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
            assert "save_tool_spec" in tool_names
            assert "install_dependencies" in tool_names
        finally:
            proc.terminate()
            proc.wait(timeout=5)


class TestKnowledgeServerStartup:

    def test_starts_and_lists_tools(self, tmp_path):
        """Knowledge server starts, completes MCP handshake, and lists tools.

        Uses a fake API key — server accepts it at init time since it only
        validates the key is non-empty. Actual API calls would fail, but we
        don't make any here.
        """
        proc = start_server(
            [
                sys.executable, "-m", "dsagt.commands.knowledge_server",
                "--base-index-dir", str(tmp_path / "kb_index"),
                "--runtime-dir", str(tmp_path / "runtime"),
            ],
            env={"LLM_API_KEY": "test-fake-key-for-startup"},
        )
        try:
            resp = mcp_initialize(proc)
            assert "result" in resp, f"Init failed: {resp}"

            tools_resp = mcp_list_tools(proc)
            assert "result" in tools_resp, f"list_tools failed: {tools_resp}"

            tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
            assert "kb_search" in tool_names
            assert "kb_list_collections" in tool_names
            assert "kb_ingest" in tool_names
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_starts_without_api_key(self, tmp_path):
        """Knowledge server starts even without an API key (lazy embedder init)."""
        stripped = {"LLM_API_KEY", "OPENAI_API_KEY", "DSAGT_EMBEDDING_BACKEND"}
        clean_env = {k: v for k, v in os.environ.items() if k not in stripped}

        proc = start_server(
            [
                sys.executable, "-m", "dsagt.commands.knowledge_server",
                "--base-index-dir", str(tmp_path / "kb_index"),
                "--runtime-dir", str(tmp_path / "runtime"),
                "--embedding-backend", "api",
            ],
            env=clean_env,
        )
        try:
            resp = mcp_initialize(proc)
            assert "result" in resp, f"Init failed: {resp}"

            tools_resp = mcp_list_tools(proc)
            assert "result" in tools_resp, f"list_tools failed: {tools_resp}"

            list_resp = mcp_call_tool(proc, "kb_list_collections", {}, msg_id=10)
            assert "result" in list_resp, f"Server crashed: {list_resp}"
        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Transport-closed reproduction tests (require valid credentials)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestKnowledgeServerSearch:
    """End-to-end MCP search through a real knowledge server subprocess.

    Reproduces the exact agent flow that causes 'transport closed':
    the server calls the embedding API during both ingest and search.
    If either call crashes the server process, MCP reports transport closed.
    """

    @pytest.fixture
    def kb_server(self, tmp_path, embedding_config):
        """Start knowledge server, complete MCP handshake, yield process."""
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
            assert "result" in resp, f"Server failed to initialize: {resp}"
            yield proc
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_search_nonexistent_collection(self, kb_server):
        """Searching a missing collection returns error, not transport closed."""
        resp = mcp_call_tool(kb_server, "kb_search", {
            "query": "test query",
            "collection": "nonexistent",
        }, msg_id=10)

        assert "result" in resp, f"Server crashed: {resp}"
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["status"] == "error"

    def test_list_collections(self, kb_server):
        """kb_list_collections works on empty server."""
        resp = mcp_call_tool(kb_server, "kb_list_collections", {}, msg_id=10)

        assert "result" in resp, f"Server crashed: {resp}"
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["status"] == "ok"

    def test_ingest_and_search(self, kb_server, tmp_path):
        """Full agent flow: ingest docs -> search -> verify response."""
        docs = tmp_path / "test_docs"
        docs.mkdir()
        (docs / "test.md").write_text(
            "# DSAGT Knowledge Base Test\n\n"
            "This document tests semantic search over knowledge bases.\n"
            "It covers document ingestion, embedding, and retrieval.\n"
        )

        ingest_resp = mcp_call_tool(kb_server, "kb_ingest", {
            "folder_path": str(docs),
        }, msg_id=10, timeout=60.0)

        assert "result" in ingest_resp, f"Ingest crashed server: {ingest_resp}"
        ingest_data = json.loads(ingest_resp["result"]["content"][0]["text"])
        assert ingest_data["status"] == "started", f"Ingest failed: {ingest_data}"
        job_id = ingest_data["job_id"]

        for i in range(60):
            status_resp = mcp_call_tool(
                kb_server, "kb_job_status", {"job_id": job_id},
                msg_id=20 + i,
            )
            status_data = json.loads(status_resp["result"]["content"][0]["text"])
            if status_data["status"] != "running":
                break
            time.sleep(1)

        assert status_data["status"] == "complete", (
            f"Ingest failed (embedding API error?): {status_data}"
        )

        search_resp = mcp_call_tool(kb_server, "kb_search", {
            "query": "knowledge base search",
            "collection": "test_docs",
        }, msg_id=100, timeout=30.0)

        assert "result" in search_resp, (
            f"Search crashed server (transport closed): {search_resp}"
        )
        search_data = json.loads(search_resp["result"]["content"][0]["text"])
        assert search_data["status"] == "ok", f"Search error: {search_data}"
        assert search_data["result_count"] > 0

    def test_server_alive_after_search_error(self, kb_server):
        """Server stays alive after a failed search (no transport closed)."""
        mcp_call_tool(kb_server, "kb_search", {
            "query": "test",
            "collection": "nonexistent",
        }, msg_id=10)

        resp = mcp_call_tool(kb_server, "kb_list_collections", {}, msg_id=11)
        assert "result" in resp, (
            f"Server died after search error (transport closed): {resp}"
        )
