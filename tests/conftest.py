"""
Shared test fixtures for DSAGT.

Loads institution-specific test configuration from tests/test_site_config.yaml.
Integration tests that need real APIs use the `embedding_config`, `llm_config`,
and `integration_config` fixtures. Tests FAIL if the config file is missing or
incomplete — copy from test_site_config.yaml.example and fill in your values.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure tests/ is on sys.path so mcp_helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))

_SITE_CONFIG_PATH = Path(__file__).parent / "test_site_config.yaml"


def _load_site_config() -> dict:
    """Load test_site_config.yaml. Fails if missing."""
    if not _SITE_CONFIG_PATH.exists():
        pytest.fail(
            f"Missing {_SITE_CONFIG_PATH.name} — copy from test_site_config.yaml.example "
            f"and fill in your institution's values"
        )
    return yaml.safe_load(_SITE_CONFIG_PATH.read_text()) or {}


@pytest.fixture
def site_config():
    """Institution-specific test config. Fails if missing."""
    return _load_site_config()


@pytest.fixture
def embedding_config(site_config, monkeypatch):
    """Embedding config from site config. Sets env vars for APIEmbeddingClient.

    Fails if base_url or api_key are missing.
    """
    emb = site_config.get("embedding", {})

    base_url = emb.get("base_url")
    api_key = emb.get("api_key") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = emb.get("model", "text-embedding-3-small")

    if not base_url:
        pytest.fail("embedding.base_url missing in test_site_config.yaml")
    if not api_key:
        pytest.fail(
            "No embedding API key: set embedding.api_key in test_site_config.yaml "
            "or export LLM_API_KEY"
        )

    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("LLM_API_KEY", api_key)
    monkeypatch.setenv("EMBEDDING_MODEL", model)

    return {"model": model, "base_url": base_url, "api_key": api_key}


@pytest.fixture
def llm_config(site_config):
    """LLM endpoint config from site config.

    Fails if llm.base_url or api_key are missing.
    Returns: {"model": ..., "base_url": ..., "api_key": ...}
    """
    llm = site_config.get("llm", {})

    base_url = llm.get("base_url")
    api_key = llm.get("api_key") or os.getenv("LLM_API_KEY")
    model = llm.get("model", "claude-sonnet-4-20250514")

    if not base_url:
        pytest.fail("llm.base_url missing in test_site_config.yaml")
    if not api_key:
        pytest.fail(
            "No LLM API key: set llm.api_key in test_site_config.yaml "
            "or export LLM_API_KEY"
        )

    return {"model": model, "base_url": base_url, "api_key": api_key}


@pytest.fixture
def integration_config(site_config, tmp_path, monkeypatch):
    """Full DSAGT config as load_config() would return.

    Creates a temporary project dir with dsagt_config.yaml derived from
    test_site_config.yaml, patches the registry, and returns the loaded config.
    """
    from dsagt.session import load_config

    project_name = site_config.get("project", "test-integration")

    # Build a dsagt_config.yaml from the site config
    config_data = {
        "project": project_name,
        "agent": site_config.get("agent", "claude-code"),
        "llm": {
            "model": site_config.get("llm", {}).get("model", "claude-sonnet-4-20250514"),
            "api_key": site_config.get("llm", {}).get("api_key", ""),
        },
        "embedding": {
            "model": site_config.get("embedding", {}).get("model", "text-embedding-3-small"),
            "api_key": site_config.get("embedding", {}).get("api_key", ""),
        },
        "proxy": site_config.get("proxy", {"port": 14000}),
        "mlflow": site_config.get("mlflow", {"port": 15001, "backend": "sqlite"}),
    }

    # Write to a temp project dir
    project_dir = tmp_path / project_name
    project_dir.mkdir(parents=True)
    for subdir in ("trace_archive", "mlflow", "tools", "tools/code", "skills", "kb_index"):
        (project_dir / subdir).mkdir()
    (project_dir / "dsagt_config.yaml").write_text(
        yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    )

    # Patch registry so load_config resolves this project to our temp dir
    monkeypatch.setattr(
        "dsagt.session._load_registry",
        lambda: {project_name: str(project_dir)},
    )

    return load_config(project_name)
