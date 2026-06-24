"""
Process-level ``dsagt-server`` entry-point tests.

The single merged server (``dsagt.mcp.server:main``) is spawned as a subprocess
to verify the entry point is wired and fails fast + clearly on a misconfigured
project — without needing a live MLflow backend or network access.

The full boot (init_tracing → shared KB → 23-tool MCP handshake) requires a
running MLflow server, so it is exercised by ``dsagt smoke-test`` (real agent),
not here.  The 23-tool composition + dispatch contract is unit-tested in-process
by ``test_dsagt_server.py``; ``_build_kb_from_config``'s credential validation by
``test_dsagt_server.py::TestBuildKbFromConfig``.
"""

import shutil
import sys

import pytest

from mcp_helpers import start_server

_uv_available = shutil.which("uv") is not None

pytestmark = pytest.mark.skipif(not _uv_available, reason="uv not available")

_SERVER_CMD = [sys.executable, "-m", "dsagt.mcp.server"]


class TestServerEntryPoint:

    def test_fails_fast_without_project_config(self, tmp_path):
        """Run from a dir with no dsagt_config.yaml → clean fail-fast.

        ``dsagt-server`` discovers its project from cwd; launched anywhere else
        it must say so rather than boot a half-configured server.
        """
        empty = tmp_path / "empty"
        empty.mkdir()
        proc = start_server(_SERVER_CMD, cwd=str(empty))
        rc = proc.wait(timeout=15)
        assert rc != 0
        assert "no dsagt_config.yaml in cwd" in proc.stderr.read()

    def test_fails_fast_without_observability_backend(self, tmp_path):
        """A project config lacking ``mlflow.port`` → init_tracing fails fast.

        The merged server requires an observability backend (it autologs every
        LLM call into MLflow); booting without one is a misconfiguration.
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
                "api_key": "test-fake-key",
            },
            "knowledge": {"chunk_size": 1024, "vector_db": "chroma", "rerank": False},
            # no mlflow.port
        }
        (project / "dsagt_config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )
        proc = start_server(_SERVER_CMD, cwd=str(project))
        rc = proc.wait(timeout=15)
        assert rc != 0
        assert "no observability backend configured" in proc.stderr.read()
