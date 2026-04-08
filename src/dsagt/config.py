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
    "categories": {
    "quality_control": "Assessment or filtering of data quality, QC metrics, thresholds, pass/fail rates",
    "data_management": "File organization, data movement, format conversion, naming conventions",
    "transformation": "Data processing steps, parameter choices, pipeline stage configuration",
    "assembly": "Genome assembly, contig generation, scaffolding, assembly QC metrics",
    "configuration": "Tool settings, environment setup, resource allocation decisions",
    "performance": "Runtime, memory usage, throughput, resource consumption observations",
    "tool_usage": "Tool selection rationale, parameter tuning, tool-specific behaviors or quirks",
    "results": "Output summaries, key findings, deliverables produced",
    },
    "extraction": {
        "threshold": 0,
        "outlier_sensitivity": 0.0,
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


def load_config(project_name: str) -> dict:
    """Load and validate a project config by name.

    Resolves the project directory from RUNTIME_DIR / project_name.

    Returns:
        Fully resolved config dict with defaults applied.
        Includes 'project_dir' (absolute path string) derived from the name.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    pdir = project_dir(project_name)
    config_path = pdir / "dsagt_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}

    config = _deep_merge(DEFAULTS, raw)
    config = _resolve_env_vars(config)

    config["project_dir"] = str(pdir)

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


def _repo_root() -> Path:
    """Find the DSAGT repo root by walking up from this file to pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Cannot find DSAGT repo root (no pyproject.toml found)")


RUNTIME_DIR = _repo_root() / "runtime"


def project_dir(project_name: str) -> Path:
    """Return the project directory path for a given project name."""
    return RUNTIME_DIR / project_name


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
            "categories": DEFAULTS["categories"],
            "extraction": DEFAULTS["extraction"],
        },
        default_flow_style=False,
        sort_keys=False,
    )
