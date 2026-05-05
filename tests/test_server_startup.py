"""
Process-level MCP server startup tests.

Each server is spawned as a subprocess and must complete the MCP handshake
without an API key or network access.  End-to-end ingest/search flows
belong in smoke_test (which drives them through the real agent), not here.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

from mcp_helpers import mcp_initialize, mcp_list_tools, start_server


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_uv_available = shutil.which("uv") is not None

pytestmark = pytest.mark.skipif(not _uv_available, reason="uv not available")


def _write_minimal_config(
    project_dir: Path,
    project_name: str = "test",
    backend: str = "api",
) -> None:
    """Write the minimum dsagt_config.yaml the servers need to start.

    Default backend is ``"api"`` here (with fake credentials) so startup
    doesn't trigger a sentence-transformers model load — keeps these
    tests fast.  Tests that need to exercise the local backend pass
    ``backend="local"`` explicitly.
    """
    import yaml
    config = {
        "project": project_name,
        "agent": "claude",
        "embedding": {
            "backend": backend,
            "model": "test-model",
            "base_url": "http://localhost:9999",
            "api_key": "test-fake-key",
        },
        "knowledge": {
            "chunk_size": 1024,
            "vector_db": "chroma",
            "rerank": False,
        },
    }
    (project_dir / "dsagt_config.yaml").write_text(
        yaml.dump(config, default_flow_style=False)
    )


# ---------------------------------------------------------------------------
# Startup tests (no API key needed)
# ---------------------------------------------------------------------------

class TestRegistryServerStartup:

    def test_starts_and_lists_tools(self, tmp_path):
        """Registry server starts without embedding credentials (kb=None)
        and completes the MCP handshake."""
        import yaml as _yaml
        project = tmp_path / "runtime"
        project.mkdir()
        # backend="api" with empty credentials → kb_available=False →
        # registry runs with kb=None.  This is the fast path the test
        # cares about (no ChromaDB / sentence-transformers init).  With
        # backend="local" (the default for new projects) registry would
        # eagerly load the local embedder, which is the slow path
        # exercised by other tests.
        (project / "dsagt_config.yaml").write_text(_yaml.dump({
            "project": "test",
            "agent": "claude",
            "embedding": {
                "backend": "api",
                "model": "", "base_url": "", "api_key": "",
            },
            "knowledge": {"chunk_size": 1024, "vector_db": "chroma", "rerank": False},
        }))
        proc = start_server(
            [sys.executable, "-m", "dsagt.commands.registry_server"],
            env={
                "DSAGT_PROJECT_DIR": str(project),
                # init_tracing raises without a backend; give it a local
                # file store so the handshake proceeds without needing an
                # MLflow server.
                "MLFLOW_TRACKING_URI": f"file://{tmp_path}/mlruns",
            },
        )
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
        project = tmp_path / "runtime"
        project.mkdir()
        _write_minimal_config(project)
        proc = start_server(
            [sys.executable, "-m", "dsagt.commands.knowledge_server"],
            env={
                "DSAGT_PROJECT_DIR": str(project),
                "LLM_API_KEY": "test-fake-key-for-startup",
                "MLFLOW_TRACKING_URI": f"file://{tmp_path}/mlruns",
            },
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

    def test_api_backend_fails_fast_without_api_key(self, tmp_path):
        """When the user explicitly opts into backend='api', the knowledge
        server must fail at startup if the api key is missing — better than
        booting with a broken embedder that 401s on the first kb_search.

        With the post-refactor default of backend='local' (no creds
        needed), this hard-fail only fires for users who have explicitly
        said they want the api backend.
        """
        import yaml
        project = tmp_path / "runtime"
        project.mkdir()
        config = {
            "project": "test",
            "agent": "claude",
            "embedding": {
                "backend": "api",
                "model": "test-model",
                "base_url": "http://localhost:9999",
                "api_key": "${LLM_API_KEY}",  # unresolved placeholder
            },
            "knowledge": {"chunk_size": 1024, "vector_db": "chroma", "rerank": False},
        }
        (project / "dsagt_config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )

        stripped = {"LLM_API_KEY", "OPENAI_API_KEY"}
        clean_env = {k: v for k, v in os.environ.items() if k not in stripped}
        clean_env["DSAGT_PROJECT_DIR"] = str(project)
        clean_env["MLFLOW_TRACKING_URI"] = f"file://{tmp_path}/mlruns"

        proc = start_server(
            [sys.executable, "-m", "dsagt.commands.knowledge_server"],
            env=clean_env,
        )
        rc = proc.wait(timeout=10)
        assert rc != 0, "backend='api' without api_key must fail at startup"
