"""
Shared test fixtures for DSAGT.

Loads institution-specific test configuration from tests/test_site_config.yaml.
Integration tests that need a real embedding API use the `site_config` and
`embedding_config` fixtures. Tests skip automatically if the config file
is missing or the API key is not set.
"""

import os
from pathlib import Path

import pytest
import yaml

_SITE_CONFIG_PATH = Path(__file__).parent / "test_site_config.yaml"


def _load_site_config() -> dict | None:
    """Load test_site_config.yaml if it exists, else return None."""
    if not _SITE_CONFIG_PATH.exists():
        return None
    return yaml.safe_load(_SITE_CONFIG_PATH.read_text()) or {}


@pytest.fixture
def site_config():
    """Institution-specific test config. Skips if test_site_config.yaml is missing."""
    config = _load_site_config()
    if config is None:
        pytest.skip("No test_site_config.yaml — copy from test_site_config.yaml.example")
    return config


@pytest.fixture
def embedding_config(site_config):
    """Embedding-specific config (model, base_url). Skips if API key is unset."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No embedding API key set (LLM_API_KEY or OPENAI_API_KEY)")

    emb = site_config.get("embedding", {})
    return {
        "model": emb.get("model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")),
        "base_url": emb.get("base_url", os.getenv("OPENAI_BASE_URL", "")),
        "api_key": api_key,
    }
