"""
DSAgt project lifecycle: configuration, initialization, services, and extraction.

Projects are registered in ~/.dsagt/projects.yaml (name → absolute path).
Default project location is ~/dsagt-projects/<name>/.

Project directory layout::

    <project_dir>/
        dsagt_config.yaml   # project configuration
        trace_archive/      # tool execution records
        mlflow/             # MLflow data
        tools/              # registered CLI tools
        tools/code/         # agent-written tool scripts
        skills/             # instruction-based agent skills
        kb_index/           # knowledge base collections
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from dsagt.memory import delete_session_log, extract_session
from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AGENTS = ("claude-code", "goose", "roo", "cline")
VALID_MLFLOW_BACKENDS = ("sqlite", "flat-file")

DEFAULT_PROJECTS_BASE = Path.home() / "dsagt-projects"
REGISTRY_DIR = Path.home() / ".dsagt"
REGISTRY_FILE = REGISTRY_DIR / "projects.yaml"

DEFAULTS = {
    "llm": {
        "model": "claude-sonnet-4-20250514",
    },
    "embedding": {
        "model": "nomic-embed-text",
        "base_url": "",
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


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_env_vars(value):
    """Replace ${VAR_NAME} references with environment variable values."""
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
                "base_url": "",
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


# ---------------------------------------------------------------------------
# Project registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict[str, str]:
    """Load the project registry. Returns empty dict if no registry exists."""
    if not REGISTRY_FILE.exists():
        return {}
    return yaml.safe_load(REGISTRY_FILE.read_text()) or {}


def _save_registry(registry: dict[str, str]) -> None:
    """Save the project registry."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(yaml.dump(registry, default_flow_style=False))


def register_project(name: str, path: Path) -> None:
    """Add or update a project in the registry."""
    registry = _load_registry()
    registry[name] = str(path.resolve())
    _save_registry(registry)


def list_projects() -> dict[str, str]:
    """Return all registered projects as {name: path}."""
    return _load_registry()


def project_dir(name: str) -> Path:
    """Resolve a project name to its directory via the registry."""
    registry = _load_registry()
    if name not in registry:
        known = ", ".join(registry.keys()) or "(none)"
        raise FileNotFoundError(
            f"Project '{name}' not found. "
            f"Run 'dsagt init {name} --agent <platform>' first.\n"
            f"Known projects: {known}"
        )
    return Path(registry[name])


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(project_name: str) -> dict:
    """Load and validate a project config by name.

    Resolves the project directory from the registry. Returns a fully
    resolved config dict with defaults applied and 'project_dir' injected.
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


# ---------------------------------------------------------------------------
# Project initialization
# ---------------------------------------------------------------------------

def init_project(project_name: str, agent: str, location: Path | None = None) -> Path:
    """Create a new project directory with default config and subdirectories."""
    if agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")

    pdir = (location or DEFAULT_PROJECTS_BASE) / project_name

    if (pdir / "dsagt_config.yaml").exists():
        raise FileExistsError(f"Project already exists: {pdir}")

    pdir.mkdir(parents=True, exist_ok=True)
    for subdir in ("trace_archive", "mlflow", "tools", "tools/code", "skills", "kb_index"):
        (pdir / subdir).mkdir(parents=True, exist_ok=True)

    (pdir / "dsagt_config.yaml").write_text(default_config_content(project_name, agent))

    register_project(project_name, pdir)
    return pdir


def move_project(project_name: str, new_location: Path) -> Path:
    """Move a project directory to a new location and update the registry."""
    old_path = project_dir(project_name)
    new_path = new_location / project_name

    if new_path.exists():
        raise FileExistsError(f"Destination already exists: {new_path}")

    new_location.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    register_project(project_name, new_path)
    return new_path


# ---------------------------------------------------------------------------
# Service start / stop
# ---------------------------------------------------------------------------

def _llm_api_key_env(config: dict) -> dict:
    """Return {"LLM_API_KEY": key} if the config has a resolved LLM API key."""
    key = config.get("llm", {}).get("api_key", "")
    if key and not key.startswith("${"):
        return {"LLM_API_KEY": key}
    return {}


def _pid_file(pdir: Path) -> Path:
    return pdir / ".pids"


def start_services(config: dict) -> dict[str, int]:
    """Start the proxy and MLflow for a project. Returns {name: pid}."""
    pdir = Path(config["project_dir"])
    pids = {}

    # Start MLflow
    mlflow_port = config["mlflow"]["port"]
    mlflow_backend = config["mlflow"]["backend"]
    mlflow_dir = pdir / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)

    if mlflow_backend == "sqlite":
        backend_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
    else:
        backend_uri = str(mlflow_dir)

    mlflow_cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", backend_uri,
        "--default-artifact-root", str(mlflow_dir / "artifacts"),
        "--host", "0.0.0.0",
        "--port", str(mlflow_port),
    ]

    mlflow_log = pdir / "mlflow.log"
    mlflow_proc = subprocess.Popen(
        mlflow_cmd,
        stdout=open(mlflow_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pids["mlflow"] = mlflow_proc.pid
    logger.info("MLflow started (pid %d) → http://localhost:%d", mlflow_proc.pid, mlflow_port)

    # Start proxy
    proxy_port = config["proxy"]["port"]
    otel_endpoint = f"http://localhost:{mlflow_port}"
    trace_dir = str(pdir / "trace_archive")

    proxy_cmd = [
        sys.executable, "-m", "dsagt.commands.proxy_server",
        "--port", str(proxy_port),
        "--records-dir", trace_dir,
        "--session", config["project"],
        "--otel-endpoint", otel_endpoint,
        "--model", config["llm"]["model"],
    ]

    proxy_log = pdir / "proxy.log"
    proxy_proc = subprocess.Popen(
        proxy_cmd,
        stdout=open(proxy_log, "w"),
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
            "DSAGT_EXTRACTION_THRESHOLD": str(
                config.get("extraction", {}).get("threshold", 0)
            ),
            **_llm_api_key_env(config),
        },
        start_new_session=True,
    )
    pids["proxy"] = proxy_proc.pid
    logger.info("Proxy started (pid %d) → http://localhost:%d", proxy_proc.pid, proxy_port)

    pid_path = _pid_file(pdir)
    pid_path.write_text(json.dumps(pids, indent=2) + "\n")

    return pids


def stop_services(project_name: str) -> list[str]:
    """Stop running services for a project."""
    pid_path = _pid_file(project_dir(project_name))
    stopped = []

    if not pid_path.exists():
        return ["No running services found."]

    pids = json.loads(pid_path.read_text())

    for name, pid in pids.items():
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped.append(f"Stopped {name} (pid {pid})")
        except (ProcessLookupError, PermissionError):
            stopped.append(f"{name} (pid {pid}) was not running")

    pid_path.unlink(missing_ok=True)
    return stopped


# ---------------------------------------------------------------------------
# Memory extraction orchestration
# ---------------------------------------------------------------------------

def run_extraction(project_name: str) -> dict:
    """Run memory extraction for a project and clean up the session log."""
    config = load_config(project_name)
    pdir = Path(config["project_dir"])
    trace_dir = pdir / "trace_archive"

    api_key = config.get("llm", {}).get("api_key", "")
    model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    session_id = config.get("project", "")
    categories = config.get("categories", {})

    if not api_key or api_key.startswith("${"):
        logger.warning("No API key available for extraction, skipping")
        delete_session_log(trace_dir)
        return {"status": "skipped", "reason": "no_api_key"}

    kb = KnowledgeBase(index_dir=pdir / "kb_index")
    try:
        return extract_session(
            trace_dir=trace_dir,
            kb=kb,
            api_key=api_key,
            model=model,
            session_id=session_id,
            categories=categories if categories else None,
            runtime_dir=pdir,
            outlier_sensitivity=float(
                config.get("extraction", {}).get("outlier_sensitivity", 0)
            ),
        )
    finally:
        kb.close()
        for suffix in (".jsonl", ".consumed"):
            leftover = trace_dir / f"session_log{suffix}"
            if leftover.exists():
                leftover.unlink()
