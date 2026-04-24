"""
DSAGT project configuration.

Loads dsagt_config.yaml from a project directory, merges with defaults,
and resolves environment variable references (${VAR_NAME} syntax).

Project directory layout:
    runtime/<project>/
        dsagt_config.yaml   # this config
        trace_archive/      # tool execution records
        mlflow/             # MLflow data
        skills/             # tool registry
        kb_index/           # knowledge base
"""

import os
import re
from pathlib import Path

import yaml

VALID_AGENTS = ("claude-code", "goose", "roo", "cline")
VALID_MLFLOW_BACKENDS = ("sqlite", "flat-file")

DEFAULTS = {
    "llm": {
        "model": "claude-sonnet-4-20250514",
    },
    "embedding": {
        "model": "nomic-embed-text",
    },
    "proxy": {
        "port": 4000,
    },
    "mlflow": {
        "port": 5001,
        "backend": "sqlite",
    },
}

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value):
    """Replace ${VAR_NAME} references with environment variable values.

    Returns the original string with references replaced. Unset variables
    are left as-is so validation can catch them.
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Merge overrides into defaults. Overrides win for leaf values."""
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(project_dir: str | Path) -> dict:
    """Load and validate a project config from its directory.

    Args:
        project_dir: Path to the project directory containing dsagt_config.yaml.

    Returns:
        Fully resolved config dict with defaults applied.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    project_dir = Path(project_dir)
    config_path = project_dir / "dsagt_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}

    config = _deep_merge(DEFAULTS, raw)
    config = _resolve_env_vars(config)

    # Inject the project directory path (not stored in YAML, derived from location)
    config["project_dir"] = str(project_dir.resolve())

    _validate(config)
    return config


def _validate(config: dict) -> None:
    """Validate required fields and values."""
    if not config.get("project"):
        raise ValueError("'project' is required in dsagt_config.yaml")

    agent = config.get("agent")
    if not agent:
        raise ValueError("'agent' is required in dsagt_config.yaml")
    if agent not in VALID_AGENTS:
        raise ValueError(f"'agent' must be one of {VALID_AGENTS}, got '{agent}'")

    backend = config.get("mlflow", {}).get("backend")
    if backend and backend not in VALID_MLFLOW_BACKENDS:
        raise ValueError(f"'mlflow.backend' must be one of {VALID_MLFLOW_BACKENDS}, got '{backend}'")


def project_dir_for(project_name: str, runtime_base: str | Path = "runtime") -> Path:
    """Return the project directory path for a given project name."""
    return Path(runtime_base).resolve() / project_name


def default_config_content(project_name: str, agent: str) -> str:
    """Generate default dsagt_config.yaml content for a new project."""
    return yaml.dump(
        {
            "project": project_name,
            "agent": agent,
            "llm": {
                "model": DEFAULTS["llm"]["model"],
                "api_key": "${LLM_API_KEY}",
            },
            "embedding": {
                "model": DEFAULTS["embedding"]["model"],
                "api_key": "${LLM_API_KEY}",
            },
            "proxy": DEFAULTS["proxy"],
            "mlflow": DEFAULTS["mlflow"],
        },
        default_flow_style=False,
        sort_keys=False,
    )
