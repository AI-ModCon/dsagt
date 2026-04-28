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
    """Validates the init -> config generation -> env var chain."""

    def test_init_generates_valid_config(self, tmp_path, monkeypatch):
        """init_project creates a valid dsagt_config.yaml."""
        import yaml
        from dsagt.session import init_project

        monkeypatch.setattr("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path)
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
        """dynamic_agent_record writes the correct proxy URL into .dsagt_env."""
        from dsagt.agents import agent_env, dynamic_agent_record, static_agent_record

        working_dir = tmp_path / "workdir"
        working_dir.mkdir()

        static_agent_record(integration_config, integration_config["agent"], working_dir)
        env = agent_env(integration_config)
        dynamic_agent_record(integration_config, env, working_dir)

        env_content = (working_dir / ".dsagt_env").read_text()
        proxy_port = integration_config["proxy"]["port"]
        assert f"localhost:{proxy_port}" in env_content

    def test_env_vars_chain_complete(self, integration_config):
        """agent_env() produces all critical environment variables."""
        from dsagt.agents import agent_env

        env = agent_env(integration_config)

        assert env["DSAGT_PROJECT"] == integration_config["project"]
        assert env["DSAGT_AGENT"] == integration_config["agent"]
        assert "DSAGT_PROJECT_DIR" in env

        # Claude Code should get ANTHROPIC_BASE_URL pointing at the proxy
        if integration_config["agent"] == "claude-code":
            proxy_port = integration_config["proxy"]["port"]
            assert str(proxy_port) in env["ANTHROPIC_BASE_URL"]

    @pytest.mark.parametrize("agent", ["goose", "claude-code", "roo", "cline", "codex"])
    def test_agent_env_matrix(self, integration_config, agent):
        """Every supported agent gets the right env vars pointing at the proxy.

        Per-agent correctness here is what prevents a silent "agent talks to
        the real upstream instead of the proxy" regression — we've hit that
        bug twice in this codebase, once for claude-code's default model and
        once for goose's ~/.config override.
        """
        from dsagt.agents import agent_env

        config = dict(integration_config)
        config["agent"] = agent
        proxy_port = config["proxy"]["port"]
        model = config["llm"]["model"]
        sentinel = "dsagt-proxy-forwarded-disable-direct-calls"

        env = agent_env(config)

        assert env["DSAGT_AGENT"] == agent

        if agent in ("claude-code", "roo"):
            # Anthropic-native agents that honor env vars: point at proxy,
            # pin model, sentinel key.  Roo rewrites lab-gateway model
            # names to its own default before sending; the proxy aliases
            # that name to the upstream primary (see commands/proxy_server.py
            # _AGENT_PRIMARY_ALIASES).
            assert env["ANTHROPIC_BASE_URL"] == f"http://localhost:{proxy_port}"
            assert env["ANTHROPIC_MODEL"] == model
            assert env["ANTHROPIC_API_KEY"] == sentinel
            assert "OPENAI_HOST" not in env
            assert "GOOSE_MODEL" not in env
        elif agent == "cline":
            # Cline ignores ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY; provider
            # config lives in CLINE_DIR/globalState.json (bootstrapped by
            # `cline auth` in launch_agent).  agent_env's job is just to
            # scope CLINE_DIR per project.
            assert env["CLINE_DIR"].endswith(".cline-data")
            assert "ANTHROPIC_BASE_URL" not in env
            assert "ANTHROPIC_API_KEY" not in env
            assert "OPENAI_HOST" not in env
        elif agent == "goose":
            # OpenAI-native: different env var names, same destination.
            assert env["OPENAI_HOST"] == f"http://localhost:{proxy_port}"
            assert env["GOOSE_PROVIDER"] == "openai"
            assert env["GOOSE_MODEL"] == model
            assert env["OPENAI_API_KEY"] == sentinel
            assert "ANTHROPIC_BASE_URL" not in env
            assert "ANTHROPIC_MODEL" not in env
            assert "CLINE_DIR" not in env
        elif agent == "codex":
            # Codex ignores most env vars except CODEX_HOME (which scopes
            # config.toml + auth state per project) and OPENAI_API_KEY
            # (consumed by [model_providers.dsagt-proxy] env_key).  The
            # proxy URL itself lives in $CODEX_HOME/config.toml, written
            # by _bootstrap_codex at launch.
            assert env["CODEX_HOME"].endswith(".codex-data")
            assert env["OPENAI_API_KEY"] == sentinel
            assert "ANTHROPIC_BASE_URL" not in env
            assert "ANTHROPIC_MODEL" not in env
            assert "CLINE_DIR" not in env
            assert "GOOSE_MODEL" not in env

    def test_mcp_env_block_routes_through_proxy(self, integration_config):
        """The MCP server env block points the embedding endpoint at the
        local proxy with a sentinel key.  The real EMBEDDING_API_KEY only
        lives in the dsagt-proxy subprocess (inherited from os.environ at
        proxy startup), where it's used to forward upstream.  See
        commands/proxy_server.py _generate_config for the model_list entry.
        """
        from dsagt.agents import _mcp_env_block, _PROXY_FORWARDED_SENTINEL

        proxy_port = integration_config["proxy"]["port"]
        env_block = _mcp_env_block(integration_config, proxy_port)

        assert env_block["EMBEDDING_BASE_URL"] == f"http://localhost:{proxy_port}"
        assert env_block["EMBEDDING_API_KEY"] == _PROXY_FORWARDED_SENTINEL
