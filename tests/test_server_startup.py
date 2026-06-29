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
        """Run from a dir with no .dsagt/config.yaml → clean fail-fast.

        ``dsagt-server`` discovers its project from cwd; launched anywhere else
        it must say so rather than boot a half-configured server.
        """
        empty = tmp_path / "empty"
        empty.mkdir()
        proc = start_server(_SERVER_CMD, cwd=str(empty))
        rc = proc.wait(timeout=15)
        assert rc != 0
        assert "no .dsagt/config.yaml in cwd" in proc.stderr.read()

    def test_mints_session_into_state_on_boot(self, tmp_path):
        """The server owns the session lifecycle: on boot it appends a session
        entry to ``.dsagt/state.yaml`` (serverless — no MLflow backend needed).

        Session minting happens before the (slow) KB build, so the state file
        appears within a second; we poll for it, then terminate.
        """
        import time

        import yaml

        project = tmp_path / "runtime"
        (project / ".dsagt").mkdir(parents=True)
        config = {
            "project": "test",
            "agent": "claude",
            "embedding": {"backend": "local", "model": "BAAI/bge-small-en-v1.5"},
            "knowledge": {"chunk_size": 1024, "rerank": False},
        }
        (project / ".dsagt" / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )
        state_file = project / ".dsagt" / "state.yaml"
        proc = start_server(_SERVER_CMD, cwd=str(project))
        try:
            for _ in range(60):  # up to ~12s
                if state_file.exists():
                    break
                time.sleep(0.2)
            assert state_file.exists(), "server did not mint a session into state.yaml"
            state = yaml.safe_load(state_file.read_text())
            assert state["sessions"][-1]["id"] == 1
            assert state["sessions"][-1]["started_at"].endswith("Z")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
