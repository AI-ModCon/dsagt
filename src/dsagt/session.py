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
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dsagt import observability as _observability
from dsagt.memory import delete_session_log, extract_session
from dsagt.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AGENTS = ("claude", "goose", "roo", "cline", "codex")
VALID_MLFLOW_BACKENDS = ("sqlite", "flat-file")

DEFAULT_PROJECTS_BASE = Path.home() / "dsagt-projects"
REGISTRY_DIR = Path.home() / ".dsagt"
REGISTRY_FILE = REGISTRY_DIR / "projects.yaml"

DEFAULTS = {
    "llm": {
        "model": "claude-sonnet-4-20250514",
        "base_url": "",
    },
    "embedding": {
        "model": "nomic-embed-text",
        "base_url": "",
    },
    "mlflow": {
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
    "knowledge": {
        "chunk_size": 1024,
        "vector_db": "chroma",
        "rerank": False,
    },
}

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def resolve_env_vars(value):
    """Replace ${VAR_NAME} references with environment variable values."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
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


def default_config_content(project_name: str, agent: str | None = None) -> str:
    """Generate default dsagt_config.yaml content for a new project.

    LLM / embedding fields are written as ``${VAR}`` references rather than
    literal placeholders so a developer who has filled in DSAGT/.env once
    can ``dsagt init`` any number of new projects without re-typing keys.
    ``resolve_env_vars`` substitutes values at config-load time.

    ``agent`` is optional: when ``None``, the YAML omits the ``agent:``
    field and ``dsagt start`` requires ``--agent X`` on first invocation.
    On that first start the agent name is persisted back into the YAML
    and subsequent starts don't need the flag.
    """
    header = (
        "# llm.provider: LiteLLM provider prefix (selects request format + auth).\n"
        "#   Common: openai, anthropic, bedrock, vertex_ai, azure, gemini,\n"
        "#   ollama, mistral, groq, deepseek.\n"
        "#   Full list: https://docs.litellm.ai/docs/providers\n"
    )
    body: dict = {"project": project_name}
    if agent is not None:
        body["agent"] = agent
    body.update({
        "llm": {
            "provider": "${LLM_PROVIDER}",
            "model": "${LLM_MODEL}",
            "base_url": "${LLM_BASE_URL}",
            "api_key": "${LLM_API_KEY}",
        },
        "embedding": {
            "provider": "${EMBEDDING_PROVIDER}",
            "model": "${EMBEDDING_MODEL}",
            "base_url": "${EMBEDDING_BASE_URL}",
            "api_key": "${EMBEDDING_API_KEY}",
        },
        "mlflow": DEFAULTS["mlflow"],
        "knowledge": DEFAULTS["knowledge"],
        "categories": DEFAULTS["categories"],
        "extraction": DEFAULTS["extraction"],
    })
    return header + yaml.dump(body, default_flow_style=False, sort_keys=False)


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
    config = resolve_env_vars(config)
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

    if not config.get("llm", {}).get("provider"):
        raise ValueError(
            "'llm.provider' is required in dsagt_config.yaml — set it to a "
            "LiteLLM provider prefix (openai, anthropic, bedrock, ...). "
            "See https://docs.litellm.ai/docs/providers"
        )


# ---------------------------------------------------------------------------
# Project initialization
# ---------------------------------------------------------------------------

def _collection_exists(path: Path) -> bool:
    """Return True if *path* looks like a persisted KB collection directory.

    Accepts FAISS indexes, ChromaDB-in-a-dir layouts, routed external
    collections, and bare ChromaDB sqlite files (as produced by
    ``dsagt setup-kb`` for description-only collections).
    """
    return (
        path.is_dir()
        and (
            (path / "index.faiss").exists()
            or (path / "chroma_ids.json").exists()
            or (path / "route.json").exists()
            or (path / "chroma.sqlite3").exists()
        )
    )


def setup_runtime_kb(base_index_dir: Path, runtime_dir: Path) -> Path:
    """Symlink base indexes into a project's kb_index directory.

    Creates ``<runtime_dir>/kb_index`` if missing. Each collection under
    *base_index_dir* that looks populated gets a symlink; existing entries
    are left alone so a project can override a base collection locally.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)

    if base_index_dir.exists():
        for collection_dir in base_index_dir.iterdir():
            if _collection_exists(collection_dir):
                dest = runtime_kb_dir / collection_dir.name
                if not dest.exists():
                    dest.symlink_to(collection_dir.resolve())

    return runtime_kb_dir


def init_project(
    project_name: str,
    agent: str | None = None,
    location: Path | None = None,
) -> Path:
    """Create a new project directory with default config and subdirectories.

    The project's data layer (``dsagt_config.yaml``, ``trace_archive/``,
    ``mlflow/``, ``skills/``, runtime KB) is agent-agnostic.  The ``agent``
    parameter is optional:

    - If supplied: the YAML records ``agent: X`` as the project's default,
      and the agent's static files (instructions, state directories) are
      written immediately so the user can edit them before first start.
    - If omitted: the YAML has no ``agent:`` field; ``dsagt start`` will
      require ``--agent X`` on first invocation and persist the choice
      into the YAML at that point.
    """
    if agent is not None and agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")

    pdir = (location or DEFAULT_PROJECTS_BASE) / project_name

    if (pdir / "dsagt_config.yaml").exists():
        raise FileExistsError(f"Project already exists: {pdir}")

    pdir.mkdir(parents=True, exist_ok=True)
    # `tools/` and `tools/code/` are created by ToolRegistry on first server
    # startup so bundled tools get copied in (it short-circuits if tools/
    # already exists).
    for subdir in ("trace_archive", "mlflow", "skills"):
        (pdir / subdir).mkdir(parents=True, exist_ok=True)

    setup_runtime_kb(REGISTRY_DIR / "kb_index", pdir)

    (pdir / "dsagt_config.yaml").write_text(default_config_content(project_name, agent))

    if agent is not None:
        # Eager static-record write so the user can edit instructions
        # files between init and start.  Empty config dict is fine — the
        # static functions don't read it today (signature reserved for
        # future project-specific instruction header injection).
        from dsagt.agents import static_agent_record
        static_agent_record({}, agent, pdir)

    register_project(project_name, pdir)
    return pdir


def persist_agent_choice(project_name: str, agent: str) -> None:
    """Add or update the ``agent:`` field in the project's YAML.

    Called by ``dsagt start`` on the first run that resolves an agent
    from ``--agent`` when the YAML didn't have one — so the next start
    doesn't need the flag.  Subsequent ``--agent`` overrides at start
    are per-run only and don't touch the YAML.
    """
    if agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")
    pdir = project_dir(project_name)
    yaml_path = pdir / "dsagt_config.yaml"
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    raw["agent"] = agent
    # Preserve the comment header from default_config_content so
    # readers still see the provider hint.  Cheap to re-emit.
    header = (
        "# llm.provider: LiteLLM provider prefix (selects request format + auth).\n"
        "#   Common: openai, anthropic, bedrock, vertex_ai, azure, gemini,\n"
        "#   ollama, mistral, groq, deepseek.\n"
        "#   Full list: https://docs.litellm.ai/docs/providers\n"
    )
    yaml_path.write_text(header + yaml.dump(raw, default_flow_style=False, sort_keys=False))


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


def remove_project(project_name: str, keep_files: bool = False) -> Path:
    """Unregister a project. By default also deletes the project directory.

    Raises RuntimeError if the project's .runtime file is present (services
    still running) — stop the session first, or delete .runtime if stale.
    """
    pdir = project_dir(project_name)

    if (pdir / ".runtime").exists():
        raise RuntimeError(
            f"Project '{project_name}' has a .runtime file — services may still be "
            f"running. Stop the session first, or remove {pdir / '.runtime'} if stale."
        )

    if not keep_files and pdir.exists():
        shutil.rmtree(pdir)

    registry = _load_registry()
    registry.pop(project_name, None)
    _save_registry(registry)
    return pdir


# ---------------------------------------------------------------------------
# Service start / stop
# ---------------------------------------------------------------------------

def _llm_api_key_env(config: dict) -> dict:
    """Return {"LLM_API_KEY": key} if the config has a resolved LLM API key."""
    key = config.get("llm", {}).get("api_key", "")
    if key and not key.startswith("${"):
        return {"LLM_API_KEY": key}
    return {}


def _embedding_provider(config: dict) -> str:
    """Resolve embedding provider with a fallback for two cases:

    - YAML key absent (older configs predating ``embedding.provider``)
    - YAML key present but holds an unresolved ``${EMBEDDING_PROVIDER}``
      literal (newer template, but ``.env`` doesn't set the var)
    """
    provider = (config.get("embedding", {}).get("provider") or "").strip()
    if not provider or provider.startswith("${"):
        return "openai_like"
    return provider


# ---------------------------------------------------------------------------
# Service supervision
#
# Each ``dsagt start`` writes ``<project>/.runtime`` containing the random
# ports it picked + the PIDs of the proxy and MLflow.  The next start (or
# ``dsagt stop``) reads that file and reaps anything still alive whose
# command line still names what we started — see ``reap_runtime``.  Random
# ports + a project-local state file means we never have to ask "is this
# listener on port 4000 mine?" — the file IS the answer.
# ---------------------------------------------------------------------------

#: Seconds to wait for SIGTERM-ed processes to exit before SIGKILL.  Long
#: enough for uvicorn + mlflow graceful shutdown (a few seconds), short
#: enough that an unresponsive process doesn't drag teardown out forever.
_STOP_GRACE_SECONDS = 5


def mlflow_command(pdir: Path, mlflow_config: dict, port: int) -> list[str]:
    """Build the argv for launching MLflow against a project's store."""
    mlflow_dir = pdir / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)
    backend_uri = (
        f"sqlite:///{mlflow_dir}/mlflow.db"
        if mlflow_config.get("backend") == "sqlite"
        else str(mlflow_dir)
    )
    return [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", backend_uri,
        "--default-artifact-root", str(mlflow_dir / "artifacts"),
        "--host", "0.0.0.0",
        "--port", str(port),
    ]


def pick_free_port() -> int:
    """Bind ``("", 0)`` so the kernel assigns a free port, then release.

    There's a microsecond race between this returning and the subprocess
    binding the same port — acceptable on a single-user dev machine.  If
    the subprocess fails to bind, the proxy.log tail surfaces the error
    via ``_wait_for_proxy``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _process_command(pid: int) -> str:
    """Return the cmdline for *pid*, or ``""`` if dead/unreadable."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, check=False, timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def reap_runtime(runtime_file: Path) -> list[str]:
    """SIGTERM, grace-wait, SIGKILL any live PIDs in *runtime_file*.

    PID-recycling guard: between writing the PID and reading it back, the
    OS could have reassigned that PID to another process; we only signal
    if its cmdline still names what we started (``mlflow`` / ``proxy``).

    Used by ``start_services`` to clear leftovers from a prior crashed
    run, and by ``stop_services`` (user-invoked teardown).  Idempotent —
    safe when the file is missing or PIDs are already dead.
    """
    if not runtime_file.exists():
        return []
    state = json.loads(runtime_file.read_text())
    pending: dict[str, tuple[int, int]] = {}  # name -> (pid, pgid)
    for name, pid in state.get("pids", {}).items():
        if name not in _process_command(pid):
            continue
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            pending[name] = (pid, pgid)
        except (ProcessLookupError, PermissionError):
            pass

    stopped: list[str] = []
    deadline = time.monotonic() + _STOP_GRACE_SECONDS
    while pending and time.monotonic() < deadline:
        time.sleep(0.2)
        for name in list(pending):
            pid, pgid = pending[name]
            try:
                os.killpg(pgid, 0)  # liveness probe
            except (ProcessLookupError, PermissionError):
                stopped.append(f"Stopped {name} (pid {pid})")
                del pending[name]

    for name, (pid, pgid) in pending.items():
        try:
            os.killpg(pgid, signal.SIGKILL)
            stopped.append(f"Stopped {name} (pid {pid}, SIGKILL after {_STOP_GRACE_SECONDS}s)")
        except ProcessLookupError:
            stopped.append(f"Stopped {name} (pid {pid})")

    runtime_file.unlink(missing_ok=True)
    return stopped


def start_services(config: dict) -> dict[str, int]:
    """Start the proxy and MLflow.  Returns ``{"mlflow": port, "proxy": port}``.

    Picks free ports via the kernel, reaps any leftovers from a prior
    crashed run, writes ``<project>/.runtime`` (pids + ports + start time),
    waits for the proxy to accept connections.  Mutates ``config["mlflow"]``
    and ``config["proxy"]`` in place to record the chosen ports so
    downstream code (``agents.py`` builds env URLs from those keys) sees
    the actually-bound values.
    """
    pdir = Path(config["project_dir"])
    runtime_file = pdir / ".runtime"

    reap_runtime(runtime_file)  # clear leftovers from any prior crashed run

    # Honor a pre-set port (CLI overrides set it on config before calling),
    # otherwise let the kernel pick a free one.
    mlflow_port = config.get("mlflow", {}).get("port") or pick_free_port()
    proxy_port = config.get("proxy", {}).get("port") or pick_free_port()
    config.setdefault("mlflow", {})["port"] = mlflow_port
    config.setdefault("proxy", {})["port"] = proxy_port

    session_id = config.get(
        "session_id",
        f"{config['project']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )

    mlflow_log = pdir / "mlflow.log"
    mlflow_proc = subprocess.Popen(
        mlflow_command(pdir, config.get("mlflow", {}), port=mlflow_port),
        stdout=open(mlflow_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logger.info("MLflow started (pid %d) → http://localhost:%d", mlflow_proc.pid, mlflow_port)

    proxy_cmd = [
        sys.executable, "-m", "dsagt.commands.proxy_server",
        "--port", str(proxy_port),
        "--records-dir", str(pdir / "trace_archive"),
        "--session", session_id,
        "--mlflow-url", f"http://localhost:{mlflow_port}",
        "--model", config["llm"]["model"],
        "--base-url", config["llm"]["base_url"],
        "--provider", config["llm"]["provider"],
        # Embedding routing through the proxy is symmetric with LLM: MCP
        # servers send embedding requests to localhost:<proxy_port>, the
        # proxy translates to whatever upstream the user configured.  See
        # commands/proxy_server.py _generate_config.
        "--embedding-model", config["embedding"]["model"],
        "--embedding-base-url", config["embedding"]["base_url"],
        "--embedding-provider", _embedding_provider(config),
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
            "DSAGT_SESSION_ID": session_id,
            # _DSAGTMlflowLogger reads this in the proxy subprocess to stamp
            # dsagt.agent on every trace.  agent_env() sets it for the agent
            # subprocess; the proxy is a sibling subprocess and needs its own
            # copy — same contract, different process tree.
            "DSAGT_AGENT": config["agent"],
            # DSAGT callback compares this against each request's model to
            # detect sidechannel/wildcard hits.  Env var name owned by
            # dsagt.observability — importing keeps the contract in one place.
            _observability.SIDECHANNEL_PRIMARY_MODEL_ENV: config["llm"]["model"],
            "DSAGT_EXTRACTION_THRESHOLD": str(
                config.get("extraction", {}).get("threshold", 0)
            ),
            **_llm_api_key_env(config),
        },
        start_new_session=True,
    )
    logger.info("Proxy started (pid %d) → http://localhost:%d", proxy_proc.pid, proxy_port)

    runtime_file.write_text(json.dumps({
        "pids": {"mlflow": mlflow_proc.pid, "proxy": proxy_proc.pid},
        "ports": {"mlflow": mlflow_port, "proxy": proxy_port},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    # Wait for the proxy to accept connections.  Without this the agent
    # launches into a half-broken environment and its first LLM call hits
    # ECONNREFUSED.  30s is generous: LiteLLM's transitive imports
    # (transformers, torch deps) take 10-15s on warm cache, longer cold.
    _wait_for_proxy(proxy_port, proxy_proc, proxy_log, timeout=30.0)

    return {"mlflow": mlflow_port, "proxy": proxy_port}


def _wait_for_proxy(
    port: int, proc: subprocess.Popen, log_path: Path, timeout: float = 30.0,
) -> None:
    """Poll ``port`` until the proxy answers, the subprocess dies, or we time out.

    Raises ``RuntimeError`` on failure with the proxy.log tail attached, so
    the failure surfaces at ``dsagt start`` rather than at first agent message.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text().splitlines()[-20:] if log_path.exists() else []
            raise RuntimeError(
                f"Proxy exited with code {proc.returncode} before becoming ready.\n  "
                + "\n  ".join(tail)
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.25)
    raise RuntimeError(
        f"Proxy did not accept connections on port {port} within {timeout:.0f}s. "
        f"See {log_path} for details."
    )


def stop_services(project_name: str) -> list[str]:
    """User-invoked teardown.  Returns ``[]`` when nothing was running."""
    return reap_runtime(project_dir(project_name) / ".runtime")



# ---------------------------------------------------------------------------
# Memory extraction orchestration
# ---------------------------------------------------------------------------

def run_extraction(project_name: str) -> dict:
    """Run memory extraction for a project and clean up the session log."""
    config = load_config(project_name)
    pdir = Path(config["project_dir"])
    trace_dir = pdir / "trace_archive"

    llm_config = config.get("llm", {})
    api_key = llm_config.get("api_key", "")
    model = llm_config.get("model", "claude-sonnet-4-20250514")
    base_url = llm_config.get("base_url", "")
    provider = llm_config.get("provider") or None
    if provider and provider.startswith("${"):
        provider = None
    session_id = config.get("project", "")
    categories = config.get("categories", {})

    if not api_key or api_key.startswith("${"):
        logger.warning("No API key available for extraction, skipping")
        delete_session_log(trace_dir)
        return {"status": "skipped", "reason": "no_api_key"}

    emb_config = config.get("embedding", {})
    kb = KnowledgeBase(
        index_dir=pdir / "kb_index",
        embedder_kwargs={
            "model": emb_config.get("model"),
            "base_url": emb_config.get("base_url"),
            "api_key": emb_config.get("api_key"),
        },
    )
    try:
        return extract_session(
            trace_dir=trace_dir,
            kb=kb,
            api_key=api_key,
            model=model,
            base_url=base_url or None,
            provider=provider,
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
