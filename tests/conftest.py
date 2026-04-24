"""
Shared test fixtures for DSAGT.

Integration-test credentials come from the project-root ``.env`` file — the
same file smoke_test/run.sh sources.  One source of truth beats two.

Tests that need real APIs use the ``embedding_config``, ``llm_config``, or
``integration_config`` fixtures.  They FAIL loudly when ``.env`` is missing a
required variable, since an integration test with empty creds is worse than
a skipped one (silent false positive).
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

# Ensure tests/ is on sys.path so mcp_helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))

_ENV_PATH = Path(__file__).parent.parent / ".env"


def _load_env() -> dict:
    """Parse .env as KEY=VALUE lines.  Fails if the file is missing.

    Deliberately does not shell-out to dotenv — .env syntax we emit is
    plain KEY=value with no quoting or substitution, so a 10-line parser
    is clearer than pulling in a dependency.
    """
    if not _ENV_PATH.exists():
        pytest.fail(
            f"Missing {_ENV_PATH} — copy .env.example to .env and fill in your "
            f"LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / EMBEDDING_* values"
        )
    result: dict[str, str] = {}
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _require(env: dict, key: str) -> str:
    value = env.get(key, "")
    if not value:
        pytest.fail(f"{key} missing or empty in .env")
    return value


@pytest.fixture
def embedding_config(monkeypatch):
    """Embedding endpoint from .env.  Sets env vars for APIEmbeddingClient."""
    env = _load_env()
    base_url = _require(env, "EMBEDDING_BASE_URL")
    api_key = _require(env, "EMBEDDING_API_KEY")
    model = _require(env, "EMBEDDING_MODEL")

    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("LLM_API_KEY", api_key)
    monkeypatch.setenv("EMBEDDING_MODEL", model)

    return {"model": model, "base_url": base_url, "api_key": api_key}


@pytest.fixture
def llm_config():
    """LLM endpoint from .env."""
    env = _load_env()
    return {
        "model": _require(env, "LLM_MODEL"),
        "base_url": _require(env, "LLM_BASE_URL"),
        "api_key": _require(env, "LLM_API_KEY"),
    }


@pytest.fixture
def integration_config(tmp_path, monkeypatch):
    """Full DSAGT config as ``load_config()`` would return it.

    Builds a dsagt_config.yaml in a temp project dir from .env values, then
    patches the registry so ``load_config(project_name)`` resolves to it.
    """
    from dsagt.session import load_config

    env = _load_env()
    project_name = "test-integration"
    config_data = {
        "project": project_name,
        "agent": "claude-code",
        "llm": {
            "model": _require(env, "LLM_MODEL"),
            "base_url": _require(env, "LLM_BASE_URL"),
            "api_key": _require(env, "LLM_API_KEY"),
        },
        "embedding": {
            "model": _require(env, "EMBEDDING_MODEL"),
            "base_url": _require(env, "EMBEDDING_BASE_URL"),
            "api_key": _require(env, "EMBEDDING_API_KEY"),
        },
        "proxy": {"port": 14000},
        "mlflow": {"port": 15001, "backend": "sqlite"},
    }

    project_dir = tmp_path / project_name
    project_dir.mkdir(parents=True)
    for subdir in ("trace_archive", "mlflow", "tools", "tools/code", "skills", "kb_index"):
        (project_dir / subdir).mkdir()
    (project_dir / "dsagt_config.yaml").write_text(
        yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    )

    monkeypatch.setattr(
        "dsagt.session._load_registry",
        lambda: {project_name: str(project_dir)},
    )

    return load_config(project_name)
