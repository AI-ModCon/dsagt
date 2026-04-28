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

VALID_AGENTS = ("claude-code", "goose", "roo", "cline", "codex")
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
        "proxy": DEFAULTS["proxy"],
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

    Raises RuntimeError if the project's .pids file is present (services still
    running) — stop the session first, or delete .pids manually if stale.
    """
    pdir = project_dir(project_name)

    if (pdir / ".pids").exists():
        raise RuntimeError(
            f"Project '{project_name}' has a .pids file — services may still be "
            f"running. Stop the session first, or remove {pdir / '.pids'} if stale."
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


def _pid_file(pdir: Path) -> Path:
    return pdir / ".pids"


def mlflow_command(pdir: Path, mlflow_config: dict, port: int | None = None) -> list[str]:
    """Build the argv for launching MLflow against a project's store.

    Shared by ``start_services`` (background, alongside the proxy) and
    ``dsagt mlflow`` (foreground, for post-session inspection).
    """
    mlflow_dir = pdir / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)

    backend_uri = (
        f"sqlite:///{mlflow_dir}/mlflow.db"
        if mlflow_config["backend"] == "sqlite"
        else str(mlflow_dir)
    )
    return [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", backend_uri,
        "--default-artifact-root", str(mlflow_dir / "artifacts"),
        "--host", "0.0.0.0",
        "--port", str(port if port is not None else mlflow_config["port"]),
    ]


def start_services(config: dict) -> dict[str, int]:
    """Start the proxy and MLflow for a project. Returns {name: pid}."""
    pdir = Path(config["project_dir"])
    pids = {}

    mlflow_port = config["mlflow"]["port"]
    proxy_port = config["proxy"]["port"]

    # Clear stale *DSAGT* processes on our ports before starting.  If the
    # port is held by something else (some other dev server the user runs
    # on 4000), we refuse to start rather than blast it — the user needs
    # to resolve the collision deliberately.
    #
    # Skipping this check lets LiteLLM silently rebind to a random port
    # when 4000 is occupied (proxy_cli line 947-948), which makes the agent
    # route LLM calls past our proxy entirely — no tracing, no provenance.
    for name, port in (("proxy", proxy_port), ("mlflow", mlflow_port)):
        if not port_in_use(port):
            continue
        if port_held_by_foreign_process(port):
            raise RuntimeError(
                f"Port {port} ({name}) is in use by a process that is not "
                f"a DSAGT service.  Something else on your machine is using "
                f"this port — stop it, or change the {name} port in "
                f"dsagt_config.yaml.  Run `lsof -iTCP:{port} -sTCP:LISTEN` "
                f"to see what's holding it."
            )
        killed = kill_processes_on_port(port)
        if killed:
            logger.info(
                "Cleared stale %s process(es) on port %d: pid(s)=%s",
                name, port, killed,
            )
            time.sleep(0.5)  # give the kernel a beat to release the socket

    mlflow_cmd = mlflow_command(pdir, config["mlflow"])

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
    mlflow_url = f"http://localhost:{mlflow_port}"
    trace_dir = str(pdir / "trace_archive")

    # Session id is generated once in _cmd_start and threaded everywhere via
    # config + DSAGT_SESSION_ID.  Fall back to project name for callers that
    # reach start_services without going through the CLI (e.g. tests).
    session_id = config.get(
        "session_id",
        f"{config['project']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )

    proxy_cmd = [
        sys.executable, "-m", "dsagt.commands.proxy_server",
        "--port", str(proxy_port),
        "--records-dir", trace_dir,
        "--session", session_id,
        "--mlflow-url", mlflow_url,
        "--model", config["llm"]["model"],
        "--base-url", config["llm"]["base_url"],
        "--provider", config["llm"]["provider"],
        # Embedding routing through the proxy is symmetric with LLM: MCP
        # servers always send embedding requests to localhost:<proxy_port>,
        # the proxy translates to whatever provider/endpoint the user
        # configured.  See commands/proxy_server.py _generate_config.
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
            # subprocess; the proxy is a sibling subprocess and needs its
            # own copy — same contract, different process tree.
            "DSAGT_AGENT": config["agent"],
            # DSAGT callback compares this against each request's model to
            # detect sidechannel/wildcard hits.  The env var name is owned
            # by dsagt.observability (sidechannel section) — importing it
            # keeps the contract in one place if we ever rename the variable.
            _observability.SIDECHANNEL_PRIMARY_MODEL_ENV: config["llm"]["model"],
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

    # Wait for the proxy to actually accept connections.  Without this
    # we hand a half-broken environment to the agent: dsagt start reports
    # success, the agent launches, then the agent's first LLM call fails
    # with ECONNREFUSED because the proxy died during startup (e.g. bad
    # config, port conflict, missing dependency).  Probing here makes
    # those failures fail loudly at the right place — dsagt start —
    # instead of at first agent message.
    # 30s is generous: LiteLLM's transitive imports (transformers, torch deps)
    # take ~10-15s on warm cache, longer on cold cache or network-backed disks
    # (e.g. OneDrive). _wait_for_proxy fast-fails on process death and orphan
    # listeners, so this timeout only governs the "still loading" case.
    if not _wait_for_proxy(proxy_port, proxy_proc, proxy_log, timeout=30.0):
        raise RuntimeError(
            f"LiteLLM proxy failed to start on port {proxy_port}. "
            f"See {proxy_log} for details. "
            f"Common causes: port already in use, missing LLM_API_KEY, "
            f"or upstream API unreachable."
        )

    return pids


def _wait_for_proxy(
    port: int,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float = 15.0,
) -> bool:
    """Poll the proxy until it accepts connections or the process dies.

    Returns True if the proxy is reachable on ``port`` AND the listener is
    in the proxy's process group, False if it never came up, exited, or the
    listener on ``port`` is a different process (LiteLLM silently rolls to a
    random port when 4000 is already in use — without the pgid check we'd
    accept an orphan as "ready" and the agent would route around our proxy).
    """
    import socket
    deadline = time.monotonic() + timeout
    our_pgid = os.getpgid(proc.pid)
    while time.monotonic() < deadline:
        # Did the subprocess die?  Fail fast — no point waiting longer.
        if proc.poll() is not None:
            logger.error(
                "Proxy process exited with code %d before becoming ready. "
                "Tail of %s:", proc.returncode, log_path,
            )
            try:
                tail = log_path.read_text().splitlines()[-20:]
                for line in tail:
                    logger.error("  %s", line)
            except OSError:
                pass
            return False

        # Is something listening on the port?  Cheap TCP connect first.
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                pass
        except (ConnectionRefusedError, OSError):
            time.sleep(0.25)
            continue

        # Verify the listener belongs to our proxy's process group.  LiteLLM
        # rebinds to a random port when 4000 is taken (proxy_cli line 947-948),
        # so "port is up" alone isn't enough — we must confirm the PID is ours.
        if _listener_pgid_matches(port, our_pgid):
            return True

        logger.error(
            "Port %d is bound but the listener is NOT our proxy subprocess "
            "(likely LiteLLM silently rebound to a random port because %d was "
            "in use). Run `dsagt stop <project>` to clear orphans.",
            port, port,
        )
        return False

    # Deadline exceeded.
    return False


def _listener_pgid_matches(port: int, expected_pgid: int) -> bool:
    """True if every listener on *port* is in process group *expected_pgid*."""
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, check=False, timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof unavailable → we can't verify; trust the TCP connect.  Better
        # to be lenient than to block on a platform without lsof.
        return True
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    if not pids:
        return False
    for pid in pids:
        try:
            if os.getpgid(pid) != expected_pgid:
                return False
        except ProcessLookupError:
            return False
    return True

    logger.error(
        "Proxy did not accept connections on port %d within %.1fs", port, timeout,
    )
    return False


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if ``port`` is unavailable for a fresh listener.

    Two probes — both have to come up clean for the port to count as free:

    1. ``connect_ex`` — a remote-style probe; catches active listeners.
    2. ``bind`` — the authoritative probe; if we can't bind even with
       SO_REUSEADDR, neither can the proxy subprocess we're about to
       launch.  Catches sockets in CLOSE_WAIT / TIME_WAIT and listeners
       whose accept queue is wedged (the connect probe sometimes misses
       these because no SYN-ACK comes back fast enough).

    The bind probe was added after smoke-test runs hit "proxy failed to
    start" with the old single-probe version because LiteLLM's
    bind-or-rebind-to-random-port behavior took over: a stuck orphan on
    4000 wasn't detected by ``connect_ex``, so we picked 4000, then
    LiteLLM silently rebound to a random port, then ``_wait_for_proxy``
    rejected the orphan listener as "not our pgid".
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def pick_free_port(preferred: int, *, max_offset: int = 20) -> tuple[int, str | None]:
    """Try ``preferred``; if taken, pick the next free port within range.

    Returns ``(picked_port, warning_or_None)``.  When the preferred port
    was free, ``warning`` is ``None``.  When we fell back, ``warning``
    is a one-line message naming the original port and the cause
    (likely a stuck dsagt service or an unrelated process), so callers
    can surface it without the user having to dig.

    Bias toward ``preferred`` so behavior stays deterministic when
    nothing's holding the port.  The fallback range is small (default
    20) — wider than that, the right answer is "fix your environment"
    rather than chase ports forever.

    Raises ``RuntimeError`` if every port in the scan range is taken,
    so the failure mode is loud rather than a silent hang.
    """
    if not port_in_use(preferred):
        return preferred, None
    cause = (
        "looks like a stuck dsagt service — try `dsagt stop`"
        if not port_held_by_foreign_process(preferred)
        else "held by another process — `lsof -iTCP:%d -sTCP:LISTEN` to identify" % preferred
    )
    for candidate in range(preferred + 1, preferred + 1 + max_offset):
        if not port_in_use(candidate):
            return candidate, (
                f"Port {preferred} in use ({cause}); falling back to {candidate}."
            )
    raise RuntimeError(
        f"All ports {preferred}–{preferred + max_offset} in use. "
        f"Run `dsagt stop` or pick a different port range with --proxy-port / --mlflow-port."
    )


#: Seconds to wait for a process to exit on SIGTERM before sending SIGKILL.
#: Set high enough for uvicorn/litellm graceful shutdown (a few seconds);
#: low enough that an unresponsive process doesn't drag out shutdown.
#: Used by both ``stop_services`` (PID-file path) and
#: ``kill_processes_on_port`` (port-sweep path).
_STOP_GRACE_SECONDS = 5


# Command-line fingerprints that identify a process as "ours" — only
# processes whose command line contains one of these strings are safe to
# kill on a port sweep.  Port 4000 and 5001 are popular defaults, so we
# must never blast an unrelated dev server / gateway the user happens to
# be running.
_OUR_PROCESS_FINGERPRINTS = (
    "dsagt-proxy",
    "dsagt.commands.proxy_server",
    "mlflow server",
    "-m mlflow",
)


def _process_command(pid: int) -> str:
    """Return the full command line for *pid*, or ``""`` if we can't read it."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, check=False, timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def _parent_pid(pid: int) -> int | None:
    """Return the parent PID of *pid*, or ``None`` if we can't read it."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "ppid="],
            capture_output=True, text=True, check=False, timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip()
    return int(text) if text.isdigit() else None


def _looks_like_our_process(pid: int, max_depth: int = 5) -> bool:
    """True if *pid* or any of its ancestors has a DSAGT / MLflow cmdline.

    Walks up the process tree because MLflow spawns uvicorn workers whose
    own cmdline doesn't mention "mlflow" — only the parent's does.  Without
    the ancestry walk, the port sweep leaves those workers stranded when
    their parent was killed but the workers haven't yet noticed.
    """
    current = pid
    for _ in range(max_depth):
        if current is None or current <= 1:
            return False
        cmd = _process_command(current)
        if any(fp in cmd for fp in _OUR_PROCESS_FINGERPRINTS):
            return True
        current = _parent_pid(current)
    return False


def kill_processes_on_port(port: int, *, only_ours: bool = True) -> list[int]:
    """Kill listeners on *port* whose command line looks like ours.

    SIGTERM, then poll up to ``_STOP_GRACE_SECONDS`` for the process group
    to exit, then SIGKILL stragglers.  Same SIGTERM→wait→SIGKILL pattern
    as ``stop_services``.  Without the wait+SIGKILL the previous behavior
    was fire-and-forget SIGTERM, which left orphans alive long enough
    that the next ``dsagt start`` would race the still-shutting-down
    proxy and silently rebind to a random port.

    When ``only_ours`` is True (default), processes whose command line
    does not contain a DSAGT / MLflow fingerprint are LEFT ALONE — the
    caller gets an empty list and can decide whether to error out.
    Guards against killing an unrelated local service the user happens
    to have running on the same port (4000 / 5001 are common defaults).

    Returns the list of pids that were sent SIGTERM (whether or not
    they exited promptly — caller can compare to ``port_in_use`` if it
    needs to know whether the port actually freed up).
    """
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []

    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    pgids_to_kill: list[tuple[int, int]] = []  # (pid, pgid)
    pgids_seen: set[int] = set()
    for pid in pids:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if pgid in pgids_seen:
            continue
        if only_ours and not _looks_like_our_process(pid):
            logger.warning(
                "Port %d is held by pid %d but its command doesn't look like "
                "a DSAGT / MLflow process; leaving it alone. cmd=%r",
                port, pid, _process_command(pid)[:120],
            )
            continue
        pgids_seen.add(pgid)
        pgids_to_kill.append((pid, pgid))

    # Phase 1: SIGTERM all
    killed: list[int] = []
    alive: list[tuple[int, int]] = []
    for pid, pgid in pgids_to_kill:
        try:
            os.killpg(pgid, signal.SIGTERM)
            killed.append(pid)
            alive.append((pid, pgid))
        except (ProcessLookupError, PermissionError):
            pass

    # Phase 2: poll for exit
    import time
    deadline = time.monotonic() + _STOP_GRACE_SECONDS
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        still_alive: list[tuple[int, int]] = []
        for pid, pgid in alive:
            try:
                os.killpg(pgid, 0)
                still_alive.append((pid, pgid))
            except (ProcessLookupError, PermissionError):
                pass
        alive = still_alive

    # Phase 3: SIGKILL stragglers
    for pid, pgid in alive:
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.info(
                "Sent SIGKILL to pid %d on port %d after %ds SIGTERM grace",
                pid, port, _STOP_GRACE_SECONDS,
            )
        except (ProcessLookupError, PermissionError):
            pass

    return killed


def port_held_by_foreign_process(port: int) -> bool:
    """True if the listener on *port* exists and isn't one of ours."""
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return False
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    return bool(pids) and not all(_looks_like_our_process(p) for p in pids)


def stop_services(project_name: str) -> list[str]:
    """Stop running services for a project.

    SIGTERM, then poll up to ``_STOP_GRACE_SECONDS`` for the process group
    to exit, then SIGKILL.  Without the wait+SIGKILL the previous behavior
    was "fire-and-forget SIGTERM and hope" — uvicorn's graceful shutdown
    takes a couple seconds, and any process that ignores SIGTERM (under
    load, in cleanup callbacks, etc.) would orphan.  That's why users
    kept needing to run ``dsagt stop`` manually after smoke runs to free
    ports 4000/5001 — this function returned "Stopped" before the proxy
    actually finished shutting down.
    """
    pid_path = _pid_file(project_dir(project_name))
    stopped = []

    if not pid_path.exists():
        return ["No running services found."]

    pids = json.loads(pid_path.read_text())

    # Phase 1: SIGTERM all
    pgids = {}
    for name, pid in pids.items():
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            pgids[name] = (pid, pgid)
        except (ProcessLookupError, PermissionError):
            stopped.append(f"{name} (pid {pid}) was not running")

    # Phase 2: poll for exit, then SIGKILL stragglers
    import time
    deadline = time.monotonic() + _STOP_GRACE_SECONDS
    while pgids and time.monotonic() < deadline:
        time.sleep(0.2)
        for name in list(pgids):
            pid, pgid = pgids[name]
            try:
                os.killpg(pgid, 0)  # signal 0 = liveness check, no signal sent
            except ProcessLookupError:
                stopped.append(f"Stopped {name} (pid {pid})")
                del pgids[name]
            except PermissionError:
                # process exists but we can't signal — assume it's gone for our purposes
                stopped.append(f"Stopped {name} (pid {pid})")
                del pgids[name]

    # Phase 3: anything still alive gets SIGKILL
    for name, (pid, pgid) in pgids.items():
        try:
            os.killpg(pgid, signal.SIGKILL)
            stopped.append(f"Stopped {name} (pid {pid}, SIGKILL after {_STOP_GRACE_SECONDS}s)")
        except ProcessLookupError:
            stopped.append(f"Stopped {name} (pid {pid})")

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
