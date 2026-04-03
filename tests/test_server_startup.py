"""
Process-level MCP server tests.

Two categories of tests:

1. **Startup tests** — verify each server starts and completes the MCP
   handshake. No API key or network access needed.

2. **Transport-closed reproduction tests** — exercise the exact agent flow
   (ingest → search) that triggers 'transport closed'. These call the real
   embedding API and require LLM_API_KEY and EMBEDDING_MODEL in the environment.

Usage:
    # Startup tests only (no API key needed):
    uv run pytest test_server_startup.py -v -k 'not Search'

    # Full suite including embedding API tests:
    source ~/.bashrc && uv run pytest test_server_startup.py -v
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_mcp_message(proc, message: dict):
    """Send a JSON-RPC message as newline-delimited JSON (MCP stdio framing)."""
    line = json.dumps(message) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def _read_line_timeout(proc, timeout: float) -> str:
    """Read one line from proc.stdout with a timeout using a daemon thread."""
    import threading

    result = [None]

    def reader():
        result[0] = proc.stdout.readline()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError("Timed out reading line from server stdout")

    line = result[0]
    if line == "":
        raise ConnectionError(
            f"Server process exited (rc={proc.poll()}). "
            f"stderr: {proc.stderr.read()}"
        )
    return line


def _read_mcp_message(proc, timeout: float = 10.0, expect_id=None) -> dict:
    """Read one JSON-RPC message from stdout (newline-delimited JSON)."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out reading MCP message (expect_id={expect_id})."
            )

        line = _read_line_timeout(proc, remaining).strip()
        if not line:
            continue

        msg = json.loads(line)
        if expect_id is not None and msg.get("id") != expect_id:
            continue
        return msg


def _mcp_initialize(proc) -> dict:
    """Send MCP initialize handshake and return the server's response."""
    _send_mcp_message(proc, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    })
    response = _read_mcp_message(proc, expect_id=1)

    _send_mcp_message(proc, {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })

    return response


def _mcp_list_tools(proc) -> dict:
    """Request tools/list and return the response."""
    _send_mcp_message(proc, {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    return _read_mcp_message(proc, expect_id=2)


def _mcp_call_tool(proc, tool_name: str, arguments: dict,
                   msg_id: int = 3, timeout: float = 30.0) -> dict:
    """Call an MCP tool and return the JSON-RPC response."""
    _send_mcp_message(proc, {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    })
    return _read_mcp_message(proc, timeout=timeout, expect_id=msg_id)


def _start_server(cmd: list[str], env: dict = None) -> subprocess.Popen:
    """Start a server subprocess with stdio pipes."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
    )


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_uv_available = shutil.which("uv") is not None

pytestmark = pytest.mark.skipif(not _uv_available, reason="uv not available")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistryServerStartup:

    def test_starts_and_lists_tools(self, tmp_path):
        """Registry server starts, completes MCP handshake, and lists tools."""
        proc = _start_server([
            sys.executable, "-m", "dsagt.registry_server",
            "--runtime-dir", str(tmp_path / "runtime"),
        ])
        try:
            resp = _mcp_initialize(proc)
            assert "result" in resp, f"Init failed: {resp}"

            tools_resp = _mcp_list_tools(proc)
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
        proc = _start_server(
            [
                sys.executable, "-m", "dsagt.knowledge_server",
                "--base-index-dir", str(tmp_path / "kb_index"),
                "--runtime-dir", str(tmp_path / "runtime"),
            ],
            env={"LLM_API_KEY": "test-fake-key-for-startup"},
        )
        try:
            resp = _mcp_initialize(proc)
            assert "result" in resp, f"Init failed: {resp}"

            tools_resp = _mcp_list_tools(proc)
            assert "result" in tools_resp, f"list_tools failed: {tools_resp}"

            tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
            assert "kb_search" in tool_names
            assert "kb_list_collections" in tool_names
            assert "kb_ingest" in tool_names
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_starts_without_api_key(self, tmp_path):
        """Knowledge server starts even without an API key (lazy embedder init).

        With the route-based architecture, embedders are created lazily when
        first needed (during ingest or search), not at server startup. The
        server should start, complete the handshake, and list tools. Embedding
        operations will fail later with a clear error.
        """
        stripped = {"LLM_API_KEY", "OPENAI_API_KEY", "DSAGT_EMBEDDING_BACKEND"}
        clean_env = {k: v for k, v in os.environ.items() if k not in stripped}

        proc = _start_server(
            [
                sys.executable, "-m", "dsagt.knowledge_server",
                "--base-index-dir", str(tmp_path / "kb_index"),
                "--runtime-dir", str(tmp_path / "runtime"),
                "--embedding-backend", "api",
            ],
            env=clean_env,
        )
        try:
            resp = _mcp_initialize(proc)
            assert "result" in resp, f"Init failed: {resp}"

            tools_resp = _mcp_list_tools(proc)
            assert "result" in tools_resp, f"list_tools failed: {tools_resp}"

            # Server is up and responsive — listing collections should work
            list_resp = _mcp_call_tool(proc, "kb_list_collections", {}, msg_id=10)
            assert "result" in list_resp, f"Server crashed: {list_resp}"
        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Transport-closed reproduction tests
#
# These exercise the exact agent flow that triggers 'transport closed':
# start server → ingest documents (embedding API) → search (embedding API).
# Requires LLM_API_KEY and EMBEDDING_MODEL — skips if not set.
#
# To exclude these when you intentionally don't have an API key:
#   uv run pytest test_server_startup.py -k 'not Search'
# ---------------------------------------------------------------------------


class TestKnowledgeServerSearch:
    """End-to-end MCP search through a real knowledge server subprocess.

    Reproduces the exact agent flow that causes 'transport closed':
    the server calls the embedding API during both ingest and search.
    If either call crashes the server process, MCP reports transport closed.
    """

    @pytest.fixture
    def kb_server(self, tmp_path):
        """Start knowledge server, complete MCP handshake, yield process."""
        api_key = os.environ.get("LLM_API_KEY")
        embedding_model = os.environ.get("EMBEDDING_MODEL")

        if not api_key:
            pytest.skip(
                "LLM_API_KEY not set. These tests call the real embedding API. "
                "Fix: export LLM_API_KEY=... && uv run pytest"
            )
        if not embedding_model:
            pytest.skip(
                "EMBEDDING_MODEL not set. Set to your institution's model, e.g. "
                "export EMBEDDING_MODEL=text-embedding-3-small-project"
            )

        proc = _start_server([
            sys.executable, "-m", "dsagt.knowledge_server",
            "--base-index-dir", str(tmp_path / "kb_index"),
            "--runtime-dir", str(tmp_path / "runtime"),
        ])
        try:
            resp = _mcp_initialize(proc)
            assert "result" in resp, f"Server failed to initialize: {resp}"
            yield proc
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_search_nonexistent_collection(self, kb_server):
        """Searching a missing collection returns error, not transport closed."""
        resp = _mcp_call_tool(kb_server, "kb_search", {
            "query": "test query",
            "collection": "nonexistent",
        }, msg_id=10)

        assert "result" in resp, f"Server crashed: {resp}"
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["status"] == "error"

    def test_list_collections(self, kb_server):
        """kb_list_collections works on empty server."""
        resp = _mcp_call_tool(kb_server, "kb_list_collections", {}, msg_id=10)

        assert "result" in resp, f"Server crashed: {resp}"
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["status"] == "ok"

    def test_ingest_and_search(self, kb_server, tmp_path):
        """Full agent flow: ingest docs → search → verify response.

        This is the exact scenario that causes 'transport closed':
        1. kb_ingest calls embedding API to index documents
        2. kb_search calls embedding API to embed the query
        If either API call crashes the server, we get transport closed.
        """
        docs = tmp_path / "test_docs"
        docs.mkdir()
        (docs / "test.md").write_text(
            "# DSAGT Knowledge Base Test\n\n"
            "This document tests semantic search over knowledge bases.\n"
            "It covers document ingestion, embedding, and retrieval.\n"
        )

        # Ingest — calls embedding API
        ingest_resp = _mcp_call_tool(kb_server, "kb_ingest", {
            "folder_path": str(docs),
        }, msg_id=10, timeout=60.0)

        assert "result" in ingest_resp, f"Ingest crashed server: {ingest_resp}"
        ingest_data = json.loads(ingest_resp["result"]["content"][0]["text"])
        assert ingest_data["status"] == "started", f"Ingest failed: {ingest_data}"
        job_id = ingest_data["job_id"]

        # Poll until ingest completes
        for i in range(60):
            status_resp = _mcp_call_tool(
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

        # Search — calls embedding API to embed query, then FAISS lookup
        search_resp = _mcp_call_tool(kb_server, "kb_search", {
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
        _mcp_call_tool(kb_server, "kb_search", {
            "query": "test",
            "collection": "nonexistent",
        }, msg_id=10)

        resp = _mcp_call_tool(kb_server, "kb_list_collections", {}, msg_id=11)
        assert "result" in resp, (
            f"Server died after search error (transport closed): {resp}"
        )
