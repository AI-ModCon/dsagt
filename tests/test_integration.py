"""
Integration tests that hit real endpoints or validate config generation.

Credentials come from the project-root ``.env`` (see conftest.py).  The
full end-to-end agent flow lives in ``tests/smoke_test/run.sh``; what's
here is the narrower layer smoke_test can't cleanly isolate:

- **TestEmbeddingEndpoint** — the raw embedding API returns valid vectors.
  Useful when debugging embedding failures, because the smoke test can't
  tell you whether a failure came from the network or the KB pipeline.
- **TestLLMEndpoint** — the raw LLM API returns a completion.  Same
  reasoning: bisects "is the upstream reachable" from "is the proxy wired
  correctly".
- **TestConfigChain** — dsagt init → static_agent_record → dynamic_agent_record → agent_env
  without touching the network, since config bugs can masquerade as
  runtime errors that are painful to diagnose via smoke.

Run:
    uv run pytest tests/test_integration.py
    uv run pytest -m integration          # all integration tests
    uv run pytest -m "not integration"    # unit tests only
"""

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
            assert result.shape == (1, result.shape[1])
            assert result.shape[1] > 0
        finally:
            client.close()

    def test_embedding_batch(self, embedding_config):
        """Multiple texts produce correct batch shape with consistent dim."""
        client = APIEmbeddingClient(
            model=embedding_config["model"],
            base_url=embedding_config["base_url"],
            api_key=embedding_config["api_key"],
        )
        texts = [
            "data quality assessment",
            "genome assembly metrics",
            "tool registration workflow",
        ]
        try:
            result = client.embed(texts)
            assert result.shape[0] == len(texts)
            assert result.shape[1] > 0
        finally:
            client.close()


# ---------------------------------------------------------------------------
# LLM endpoint
# ---------------------------------------------------------------------------

class TestLLMEndpoint:
    """Validates the LLM API is reachable and returns valid responses.

    Posts to ``/chat/completions`` (OpenAI-shape) because .env's LLM_BASE_URL
    is an OpenAI-compatible gateway — Anthropic-format ``/v1/messages`` would
    404 on it.  The proxy (commands/proxy_server.py) sets
    ``use_chat_completions_url_for_anthropic_messages = True`` for the same
    reason.
    """

    def test_llm_returns_response(self, llm_config):
        """Minimal LLM request returns 200 with non-empty content."""
        # Some gateways expect /v1 already in base_url (RC Chat / Ollama
        # OpenAI-compat), others don't (ai-incubator-api).  Don't double it.
        base = llm_config["base_url"].rstrip("/")
        url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {llm_config['api_key']}",
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

        choices = data.get("choices") or []
        assert choices, f"No choices in response: {data}"
        content = choices[0].get("message", {}).get("content", "")
        assert content, f"Empty content: {data}"


# ---------------------------------------------------------------------------
# Config chain (no network)
# ---------------------------------------------------------------------------

class TestConfigChain:
    """Validates the init → config generation → env var chain (BYOA)."""

    def test_init_generates_valid_config(self, tmp_path, monkeypatch):
        """init_project creates a valid dsagt_config.yaml and returns
        ``(pdir, mlflow_port)``."""
        import yaml
        from dsagt.session import init_project

        monkeypatch.setattr("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path)
        reg = {}
        monkeypatch.setattr("dsagt.session._load_registry", lambda: dict(reg))
        monkeypatch.setattr("dsagt.session._save_registry", lambda r: reg.update(r))
        monkeypatch.setattr("dsagt.session.register_project", lambda n, p: reg.update({n: str(p)}))

        pdir, port = init_project("test-proj", "claude")

        assert isinstance(port, int) and port > 0
        config_path = pdir / "dsagt_config.yaml"
        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        assert config["project"] == "test-proj"
        assert config["agent"] == "claude"
        assert config["mlflow"]["port"] == port
        assert "embedding" in config
        # BYOA: no llm: block in the user-facing YAML.
        assert "llm" not in config


@pytest.mark.skip(reason="proxy mode deferred to Phase 2")
class TestProxyConfigChain:
    """Proxy-mode env-routing matrix.  Re-enable when --proxy_traces is
    restored in Phase 2 — these tests are the regression net for the
    'agent talks to upstream instead of the proxy' bug class."""

    def test_agent_configs_have_correct_ports(self, integration_config, tmp_path):
        from dsagt.agents import agent_env, dynamic_agent_record, static_agent_record

        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        static_agent_record(integration_config, integration_config["agent"], working_dir)
        env = agent_env(integration_config)
        dynamic_agent_record(integration_config, env, working_dir)
        # Phase 2: assert proxy URL lands in the per-agent MCP-config artifact
        # (cline_mcp_settings.json / .roo/mcp.json / codex config.toml).

    def test_env_vars_chain_complete(self, integration_config):
        from dsagt.agents import agent_env

        env = agent_env(integration_config)
        assert env["DSAGT_PROJECT"] == integration_config["project"]
        assert env["DSAGT_AGENT"] == integration_config["agent"]
        if integration_config["agent"] == "claude":
            proxy_port = integration_config["proxy"]["port"]
            assert str(proxy_port) in env["ANTHROPIC_BASE_URL"]

    @pytest.mark.parametrize("agent", ["goose", "claude", "roo", "cline", "codex"])
    def test_agent_env_matrix(self, integration_config, agent):
        from dsagt.agents import agent_env

        config = dict(integration_config)
        config["agent"] = agent
        proxy_port = config["proxy"]["port"]
        model = config["llm"]["model"]
        sentinel = "dsagt-proxy-forwarded-disable-direct-calls"

        env = agent_env(config)
        assert env["DSAGT_AGENT"] == agent

        if agent in ("claude", "roo"):
            assert env["ANTHROPIC_BASE_URL"] == f"http://localhost:{proxy_port}"
            assert env["ANTHROPIC_MODEL"] == model
            assert env["ANTHROPIC_API_KEY"] == sentinel
        elif agent == "goose":
            assert env["OPENAI_HOST"] == f"http://localhost:{proxy_port}"
            assert env["GOOSE_PROVIDER"] == "openai"
            assert env["OPENAI_API_KEY"] == sentinel
        elif agent == "codex":
            assert env["CODEX_HOME"].endswith(".codex-data")
            assert env["OPENAI_API_KEY"] == sentinel
