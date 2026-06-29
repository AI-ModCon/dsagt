"""
Integration tests that hit real endpoints or validate config generation.

Credentials come from the project-root ``.env`` (see conftest.py).  The
full end-to-end agent flow lives in ``tests/smoke_test/run.sh``; what's
here is the narrower layer smoke_test can't cleanly isolate:

- **TestEmbeddingEndpoint** — the raw embedding API returns valid vectors.
  Useful when debugging embedding failures, because the smoke test can't
  tell you whether a failure came from the network or the KB pipeline.
- **TestLLMEndpoint** — the raw LLM API returns a completion.  Same
  reasoning: bisects "is the upstream reachable" from "is the KB/agent
  layer wired correctly".
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

from dsagt.knowledge import APIEmbedder

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Embedding endpoint
# ---------------------------------------------------------------------------


class TestEmbeddingEndpoint:
    """Validates the embedding API is reachable and returns valid vectors."""

    def test_embedding_returns_vector(self, embedding_config):
        """Single text produces a numpy array with positive dimension."""
        client = APIEmbedder(
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
        client = APIEmbedder(
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
    404 on it.
    """

    def test_llm_returns_response(self, llm_config):
        """Minimal LLM request returns 200 with non-empty content."""
        # Some gateways expect /v1 already in base_url (RC Chat / Ollama
        # OpenAI-compat), others don't (ai-incubator-api).  Don't double it.
        base = llm_config["base_url"].rstrip("/")
        url = (
            f"{base}/chat/completions"
            if base.endswith("/v1")
            else f"{base}/v1/chat/completions"
        )
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
        """init_project creates a valid .dsagt/config.yaml and returns the
        project dir (serverless — no port)."""
        import yaml
        from dsagt.session import init_project

        monkeypatch.setattr("dsagt.session.DEFAULT_PROJECTS_BASE", tmp_path)
        reg = {}
        monkeypatch.setattr("dsagt.session._load_registry", lambda: dict(reg))
        monkeypatch.setattr("dsagt.session._save_registry", lambda r: reg.update(r))
        monkeypatch.setattr(
            "dsagt.session.register_project", lambda n, p: reg.update({n: str(p)})
        )

        pdir = init_project("test-proj", "claude")

        config_path = pdir / ".dsagt" / "config.yaml"
        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        assert config["project"] == "test-proj"
        assert config["agent"] == "claude"
        assert "mlflow" not in config
        assert "embedding" in config
        # BYOA: no llm: block in the config.
        assert "llm" not in config
