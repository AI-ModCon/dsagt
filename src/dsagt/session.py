"""
DSAgt project lifecycle: configuration, initialization, services, and extraction.

Projects are registered in ~/dsagt-projects/projects.yaml (name → absolute path).
Default project location is ~/dsagt-projects/<name>/.  Shared bundled-content
KB lives alongside at ~/dsagt-projects/kb_index/ (built by dsagt setup-kb).

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

from dsagt.memory import extract_session
from dsagt.knowledge import KnowledgeBase
from dsagt.provenance import index_trace_archive

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AGENTS = ("claude", "goose", "roo", "cline", "codex", "opencode")
VALID_MLFLOW_BACKENDS = ("sqlite", "flat-file")

DEFAULT_PROJECTS_BASE = Path.home() / "dsagt-projects"
# Registry + shared KB live alongside projects under one visible tree —
# ``~/dsagt-projects/projects.yaml`` (name → path) and
# ``~/dsagt-projects/kb_index/`` (shared bundled-content KB built by
# ``dsagt setup-kb``).  Migrated from ``~/.dsagt/`` on 2026-05-07.
REGISTRY_DIR = DEFAULT_PROJECTS_BASE
REGISTRY_FILE = REGISTRY_DIR / "projects.yaml"
RESERVED_PROJECT_NAMES = ("projects.yaml", "kb_index", ".skill_sources")

DEFAULTS = {
    # ``llm`` block uses ${VAR} placeholders so per-project config
    # references the user's shell env at resolve time.  In BYOA mode
    # (the default), agent_env's proxy_port gate at agents/__init__.py
    # short-circuits the env_overrides call, so this block sits dormant
    # — no agent env-var leaks.  In proxy mode (--enable-proxy),
    # env_overrides reads these values to translate into per-agent env
    # vars and proxy_server.py reads them to render its YAML.
    "llm": {
        "provider": "${LLM_PROVIDER}",
        "model": "${LLM_MODEL}",
        "base_url": "${LLM_BASE_URL}",
        "api_key": "${LLM_API_KEY}",
    },
    "embedding": {
        # Default backend: "local" — sentence-transformers, CPU-side, no
        # API credentials needed.  Switch to "api" to route through
        # litellm to a hosted endpoint (then fill in base_url / api_key
        # below and pick a model name your endpoint serves).
        # Local default is the HuggingFace identifier ``LocalEmbeddingClient``
        # downloads; user can override with any sentence-transformers
        # repo (e.g. ``BAAI/bge-large-en-v1.5`` for higher quality).
        "backend": "local",
        "model": "BAAI/bge-small-en-v1.5",
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
    # External agent-skill catalogs.  ``sources`` are GitHub repos whose
    # SKILL.md skills get indexed into per-source catalog collections for
    # ``search_skills`` (searchable but NOT loaded into agent context).
    # The agent installs a chosen one via the ``install_skill`` MCP tool;
    # the agent setup then mirrors installed + bundled skills into the
    # platform's native skill dir (e.g. ``.claude/skills/``).
    "skills": {
        "sources": [
            {
                "name": "scientific",
                "url": "https://github.com/K-Dense-AI/scientific-agent-skills",
                "branch": "main",
                "subdir": "skills",
            },
        ],
        "populate_catalog": True,  # index sources into the catalog at setup-kb
        "populate_native": True,  # mirror installed+bundled into .claude/skills
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


def default_config_content(
    project_name: str,
    agent: str,
    mlflow_port: int,
) -> str:
    """Generate the internal dsagt_config.yaml for a new project.

    Internal-only: users don't edit this.  Holds the project's
    embedding / mlflow / knowledge / extraction settings plus the
    pinned MLflow port so MCP servers (started fresh per agent run)
    know where to log without relying on shell-env inheritance.

    User credentials are NOT here — the agent reads them from the
    user's shell env directly (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc).
    """
    body: dict = {
        "project": project_name,
        "agent": agent,
        "mlflow": {**DEFAULTS["mlflow"], "port": mlflow_port},
        "embedding": dict(DEFAULTS["embedding"]),
        "knowledge": DEFAULTS["knowledge"],
        "categories": DEFAULTS["categories"],
        "extraction": DEFAULTS["extraction"],
        "skills": DEFAULTS["skills"],
    }
    return yaml.dump(body, default_flow_style=False, sort_keys=False)


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


def kb_from_config(config: dict, index_dir: Path | None = None) -> "KnowledgeBase":
    """Build a KnowledgeBase from a resolved project config.

    Mirrors the embedding-backend resolution used by ``extract_session`` so
    callers (CLI ``skills`` group, catalog sync) get a KB wired to the same
    embedder the project uses.  Defaults to ``<project_dir>/kb_index``.
    """
    pdir = Path(config["project_dir"])
    emb = config.get("embedding", {})
    backend = emb.get("backend", "local")
    if backend == "local":
        model = emb.get("model")
        if model and "/" not in str(model):
            model = None
        embedder_kwargs = {"model": model}
    else:
        embedder_kwargs = {
            "model": emb.get("model"),
            "base_url": emb.get("base_url"),
            "api_key": os.environ.get("EMBEDDING_API_KEY", ""),
        }
    return KnowledgeBase(
        index_dir=index_dir or (pdir / "kb_index"),
        default_embedder=backend,
        embedder_kwargs=embedder_kwargs,
    )


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
        raise ValueError(
            f"'mlflow.backend' must be one of {VALID_MLFLOW_BACKENDS}, got '{backend}'"
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
    return path.is_dir() and (
        (path / "index.faiss").exists()
        or (path / "chroma_ids.json").exists()
        or (path / "route.json").exists()
        or (path / "chroma.sqlite3").exists()
    )


def setup_runtime_kb(base_index_dir: Path, runtime_dir: Path) -> Path:
    """Copy base KB collections into a project's kb_index directory.

    Creates ``<runtime_dir>/kb_index`` if missing.  For each collection
    under *base_index_dir* that looks populated and whose project-local
    twin doesn't already exist, **copies** (not symlinks) the entire
    collection directory into the project's kb_index.

    Why copy instead of symlink: different projects on the same machine
    may run different dsagt versions, and a symlink would let one
    project's ``dsagt setup-kb --rebuild`` mutate every project's view
    of bundled content.  A copy pins each project to whatever bundled
    content was current when the project first ran.

    Existing project-local collections are left alone, so an agent's
    saved tools / skills / ingests are preserved across re-runs.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)

    if not base_index_dir.exists():
        return runtime_kb_dir

    for collection_dir in base_index_dir.iterdir():
        if not _collection_exists(collection_dir):
            continue
        dest = runtime_kb_dir / collection_dir.name
        if dest.exists():
            continue
        # Resolve in case the source collection itself is a symlink
        # (older projects that used the symlink path).
        shutil.copytree(collection_dir.resolve(), dest)

    return runtime_kb_dir


def init_project(
    project_name: str,
    agent: str,
    mlflow_port: int | None = None,
    location: Path | None = None,
) -> tuple[Path, int]:
    """Create a new project directory with default config and subdirectories.

    BYOA model: at init we lay down everything the user needs to point
    their own agent process at our MCP servers.  Picks (or honors) an
    MLflow port and writes it to the internal ``dsagt_config.yaml`` so
    later ``dsagt mlflow <project>`` and the MCP-server children all
    agree on where traces land.

    Returns ``(project_dir, mlflow_port)``.
    """
    if agent not in VALID_AGENTS:
        raise ValueError(f"agent must be one of {VALID_AGENTS}, got '{agent}'")
    if project_name in RESERVED_PROJECT_NAMES:
        raise ValueError(
            f"project name '{project_name}' collides with a reserved entry "
            f"in {DEFAULT_PROJECTS_BASE} (kb_index/ or projects.yaml).  "
            f"Pick another name."
        )

    pdir = (location or DEFAULT_PROJECTS_BASE) / project_name

    if (pdir / "dsagt_config.yaml").exists():
        raise FileExistsError(f"Project already exists: {pdir}")

    pdir.mkdir(parents=True, exist_ok=True)
    # `tools/` and `tools/code/` are created by ToolRegistry on first server
    # startup so bundled tools get copied in (it short-circuits if tools/
    # already exists).
    for subdir in ("trace_archive", "mlflow", "skills", ".dsagt"):
        (pdir / subdir).mkdir(parents=True, exist_ok=True)

    setup_runtime_kb(REGISTRY_DIR / "kb_index", pdir)

    # If the shared KB hasn't been built yet, warn — the project's
    # kb_index/ will be empty until ``dsagt setup-kb`` runs.  Don't
    # rebuild here: that conflicts with the contract that ``dsagt
    # init`` does no embedding work.
    shared_kb = REGISTRY_DIR / "kb_index"
    if not shared_kb.exists() or not any(
        _collection_exists(c) for c in shared_kb.iterdir()
    ):
        print(
            f"  Warning: shared KB at {shared_kb} is empty — "
            "run `dsagt setup-kb` to build bundled tools and skills "
            "before launching your agent.",
            flush=True,
        )

    if mlflow_port is None:
        mlflow_port = pick_free_port()

    (pdir / "dsagt_config.yaml").write_text(
        default_config_content(project_name, agent, mlflow_port)
    )

    register_project(project_name, pdir)
    return pdir, mlflow_port


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
    yaml_path.write_text(
        header + yaml.dump(raw, default_flow_style=False, sort_keys=False)
    )


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
# port MLflow bound + its pid.  The next start (or ``dsagt stop``) reads
# that file and reaps anything still alive whose command line still names
# ``mlflow`` — see ``reap_runtime``.  Random ports + a project-local
# state file means we never have to ask "is this listener on port 5000
# mine?" — the file IS the answer.
# ---------------------------------------------------------------------------

#: Seconds to wait for SIGTERM-ed processes to exit before SIGKILL.  Long
#: enough for uvicorn + mlflow graceful shutdown (a few seconds), short
#: enough that an unresponsive process doesn't drag teardown out forever.
_STOP_GRACE_SECONDS = 5


def mlflow_command(pdir: Path, mlflow_config: dict, port: int) -> list[str]:
    """Build the argv for launching MLflow against a project's store.

    ``--workers 1``: dsagt is a single-user dev tool — the agent makes
    serial LLM calls and MCP-server spans are low-volume.  Default
    workers=4 each spin up a fresh Python process re-importing MLflow's
    full surface (fastapi, sqlalchemy, alembic), so dropping to 1
    shaves ~0.5s off startup with zero observable cost on this load.
    """
    mlflow_dir = pdir / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)
    backend_uri = (
        f"sqlite:///{mlflow_dir}/mlflow.db"
        if mlflow_config.get("backend") == "sqlite"
        else str(mlflow_dir)
    )
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_uri,
        "--default-artifact-root",
        str(mlflow_dir / "artifacts"),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--workers",
        "1",
    ]


def pick_free_port() -> int:
    """Bind ``("", 0)`` so the kernel assigns a free port, then release.

    There's a microsecond race between this returning and the subprocess
    binding the same port — acceptable on a single-user dev machine.  If
    the subprocess fails to bind, the mlflow.log tail surfaces the error
    via ``_wait_for_mlflow``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _process_command(pid: int) -> str:
    """Return the cmdline for *pid*, or ``""`` if dead/unreadable."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
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
            stopped.append(
                f"Stopped {name} (pid {pid}, SIGKILL after {_STOP_GRACE_SECONDS}s)"
            )
        except ProcessLookupError:
            stopped.append(f"Stopped {name} (pid {pid})")

    runtime_file.unlink(missing_ok=True)
    return stopped


def start_services(config: dict) -> dict[str, int]:
    """Start MLflow, optionally start the dsagt-proxy too.

    Picks free ports (or honors pre-set ones), reaps any leftovers from a
    prior crashed run, writes ``<project>/.runtime`` (pids + ports +
    start time), and waits for each service to accept connections.

    Returns ``{"mlflow": port, "proxy": port}`` when the proxy was
    requested (``"proxy"`` key present in config), else just
    ``{"mlflow": port}``.

    The proxy is opt-in via ``dsagt start --enable-proxy``.  Used for
    agents that don't natively emit OTel traces with full LLM-call
    payloads (cline, roo, codex partial) — the proxy interposes on their
    LLM calls and produces traces on their behalf via the same OTLP path
    Claude Code uses natively.  See ``commands/proxy_server.py``.
    """
    pdir = Path(config["project_dir"])
    runtime_file = pdir / ".runtime"

    reap_runtime(runtime_file)  # clear leftovers from any prior crashed run

    # KB bootstrap is intentionally NOT here.  The contract:
    #   * ``dsagt setup-kb`` builds shared ~/dsagt-projects/kb_index/ (one-time
    #     per machine, the only place embedding work happens for bundled
    #     content)
    #   * ``dsagt init`` copies the shared collections into the project
    #   * ``dsagt start`` does no embedding work, no implicit rebuild,
    #     no sentinel checks — just spawns services
    # If the project KB is empty or stale, the MCP servers surface a
    # clear "run dsagt setup-kb" error rather than silently rebuilding.

    mlflow_port = config.get("mlflow", {}).get("port") or pick_free_port()
    config.setdefault("mlflow", {})["port"] = mlflow_port

    proxy_requested = "proxy" in config
    proxy_port = None
    if proxy_requested:
        proxy_port = config["proxy"].get("port") or pick_free_port()
        config["proxy"]["port"] = proxy_port

    session_id = config.get(
        "session_id",
        f"{config['project']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    config["session_id"] = session_id

    mlflow_log = pdir / "mlflow.log"
    mlflow_proc = subprocess.Popen(
        mlflow_command(pdir, config.get("mlflow", {}), port=mlflow_port),
        stdout=open(mlflow_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logger.info(
        "MLflow started (pid %d) → http://localhost:%d",
        mlflow_proc.pid,
        mlflow_port,
    )

    pids = {"mlflow": mlflow_proc.pid}
    ports = {"mlflow": mlflow_port}

    proxy_proc = None
    if proxy_requested:
        # MLflow has to be up before the proxy starts because the proxy's
        # init_tracing() calls mlflow.set_experiment() during startup.
        _wait_for_mlflow(mlflow_port, mlflow_proc, mlflow_log, timeout=30.0)

        proxy_proc = _start_proxy(config, pdir, mlflow_port, proxy_port, session_id)
        pids["proxy"] = proxy_proc.pid
        ports["proxy"] = proxy_port

    runtime_file.write_text(
        json.dumps(
            {
                "pids": pids,
                "ports": ports,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    if not proxy_requested:
        _wait_for_mlflow(mlflow_port, mlflow_proc, mlflow_log, timeout=30.0)
    if proxy_proc is not None:
        _wait_for_proxy(proxy_port, proxy_proc, pdir / "proxy.log", timeout=45.0)

    return ports


def _start_proxy(
    config: dict,
    pdir: Path,
    mlflow_port: int,
    proxy_port: int,
    session_id: str,
) -> subprocess.Popen:
    """Spawn the dsagt-proxy subprocess.

    Forwards LLM + embedding requests using the user's configured
    upstream credentials (LLM_API_KEY / EMBEDDING_API_KEY from
    os.environ) and emits OTLP traces to MLflow via init_tracing.

    Crashes loudly if required config keys are missing — better than
    spawning a half-configured proxy that 500s on the first agent
    request.
    """
    llm = config.get("llm") or {}
    emb = config.get("embedding") or {}
    for required in ("model", "base_url", "provider"):
        if not llm.get(required):
            raise RuntimeError(
                f"--enable-proxy needs config.llm.{required} (got {llm.get(required)!r})"
            )

    cmd = [
        sys.executable,
        "-m",
        "dsagt.commands.proxy_server",
        "--port",
        str(proxy_port),
        "--mlflow-url",
        f"http://localhost:{mlflow_port}",
        "--project",
        config["project"],
        "--session",
        session_id,
        "--records-dir",
        str(pdir / "trace_archive"),
        "--model",
        llm["model"],
        "--base-url",
        llm["base_url"],
        "--provider",
        llm["provider"],
    ]
    # Embedding routing through the proxy is only relevant when the
    # project's embedding backend is ``api`` — in ``local`` mode the
    # knowledge MCP server uses sentence-transformers in-process and
    # never makes HTTP embedding calls, so the proxy doesn't need an
    # embedding route at all.
    if (emb.get("backend") or "local").lower() == "api":
        for required in ("model", "base_url", "provider"):
            if not emb.get(required):
                raise RuntimeError(
                    f"--enable-proxy with embedding.backend=api needs "
                    f"config.embedding.{required} (got {emb.get(required)!r})"
                )
        cmd.extend(
            [
                "--embedding-model",
                emb["model"],
                "--embedding-base-url",
                emb["base_url"],
                "--embedding-provider",
                emb["provider"],
            ]
        )
    proxy_log = pdir / "proxy.log"
    # The proxy needs the *real* upstream credentials in env (not the
    # sentinel agents see).  os.environ already has them from the user's
    # shell or .env file.  We pass DSAGT_PROJECT explicitly so
    # _resolve_experiment_id picks the right experiment.
    proxy_proc = subprocess.Popen(
        cmd,
        stdout=open(proxy_log, "w"),
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "DSAGT_PROJECT": config["project"],
            "DSAGT_PROJECT_DIR": str(pdir),
            "DSAGT_SESSION_ID": session_id,
            "DSAGT_AGENT": config.get("agent", ""),
            "MLFLOW_TRACKING_URI": f"http://localhost:{mlflow_port}",
        },
        start_new_session=True,
    )
    logger.info(
        "Proxy started (pid %d) → http://localhost:%d",
        proxy_proc.pid,
        proxy_port,
    )
    return proxy_proc


def _wait_for_proxy(
    port: int,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float = 45.0,
) -> None:
    """Poll *port* until the proxy answers, the subprocess dies, or we time out.

    Generous default (45s) because LiteLLM's transitive imports
    (transformers/torch dependencies) can take 10-15s on warm cache,
    longer cold.  Raises with proxy.log tail attached so
    ``dsagt start`` surfaces the failure instead of the agent's first
    LLM call hitting ECONNREFUSED.
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


def _wait_for_mlflow(
    port: int,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float = 30.0,
) -> None:
    """Poll *port* until MLflow answers, the subprocess dies, or we time out.

    Raises ``RuntimeError`` on failure with the mlflow.log tail attached,
    so the failure surfaces at ``dsagt start`` rather than at the agent's
    first OTLP export attempt.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text().splitlines()[-20:] if log_path.exists() else []
            raise RuntimeError(
                f"MLflow exited with code {proc.returncode} before becoming ready.\n  "
                + "\n  ".join(tail)
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.25)
    raise RuntimeError(
        f"MLflow did not accept connections on port {port} within {timeout:.0f}s. "
        f"See {log_path} for details."
    )


def stop_services(project_name: str) -> list[str]:
    """User-invoked teardown.  Returns ``[]`` when nothing was running."""
    return reap_runtime(project_dir(project_name) / ".runtime")


# ---------------------------------------------------------------------------
# Memory extraction orchestration
# ---------------------------------------------------------------------------


def run_extraction(project_name: str) -> dict:
    """Two-phase post-session work, both best-effort.

    1. **Tool-execution indexing** (always): embed every JSON record in
       ``<pdir>/trace_archive/`` into the project's ``tool_use``
       collection so the agent can semantic-search prior tool runs in
       future sessions.  Pure embedding work — no LLM call, no
       ``DSAGT_MEMORY_*`` needed.  Local-backend embeddings (BYOA
       default) need no credentials at all.
    2. **LLM-based memory extraction** (gated on ``DSAGT_MEMORY_*``):
       summarises MLflow LLM-call traces into episodic memories.
       Currently reads only ``service.name == "dsagt-proxy"`` traces
       (see ``memory.DSAGT_EXTRACTION_SOURCE_SERVICE_NAME``); in BYOA
       these don't exist yet, so this phase is effectively dormant
       until Phase 2 / native-shape parsers land.
    """
    config = load_config(project_name)
    pdir = Path(config["project_dir"])

    emb_config = config.get("embedding", {})
    backend = emb_config.get("backend", "local")
    # Local backend rejects base_url / api_key (no remote call); api
    # backend reads creds from the shell.  Local model must be a HF
    # identifier (``org/repo``); if it isn't (legacy projects had
    # Ollama-style ``nomic-embed-text`` here), fall through to
    # LocalEmbeddingClient's default by passing model=None.
    if backend == "local":
        model = emb_config.get("model")
        if model and "/" not in str(model):
            model = None
        embedder_kwargs = {"model": model}
    else:
        embedder_kwargs = {
            "model": emb_config.get("model"),
            "base_url": emb_config.get("base_url"),
            "api_key": os.environ.get("EMBEDDING_API_KEY", ""),
        }
    kb = KnowledgeBase(
        index_dir=pdir / "kb_index",
        default_embedder=backend,
        embedder_kwargs=embedder_kwargs,
    )

    # Phase 1: index trace_archive into tool_use collection.
    tool_use_indexed = 0
    try:
        trace_result = index_trace_archive(pdir / "trace_archive", kb)
        tool_use_indexed = trace_result.get("indexed", 0)
    except Exception as e:
        logger.warning("Tool execution indexing failed: %s", e)

    # Phase 2: LLM-based memory extraction.  Skip silently if not configured.
    api_key = os.environ.get("DSAGT_MEMORY_API_KEY", "")
    model = os.environ.get("DSAGT_MEMORY_MODEL", "")
    if not api_key or not model:
        kb.close()
        return {"status": "tool_use_only", "tool_use_indexed": tool_use_indexed}

    base_url = os.environ.get("DSAGT_MEMORY_BASE_URL", "")
    provider = os.environ.get("DSAGT_MEMORY_PROVIDER") or None
    session_id = config.get("session_id") or config.get("project", "")
    categories = config.get("categories", {})

    mlflow_port = config.get("mlflow", {}).get("port")
    mlflow_uri = (
        f"http://localhost:{mlflow_port}"
        if mlflow_port
        else os.environ.get("MLFLOW_TRACKING_URI")
    )
    try:
        result = extract_session(
            project_name=project_name,
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
            mlflow_uri=mlflow_uri,
        )
        result["tool_use_indexed"] = tool_use_indexed
        return result
    finally:
        kb.close()
