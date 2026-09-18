"""
DSAgt project lifecycle: configuration, initialization, services, and extraction.

Projects are registered in ~/dsagt-projects/projects.yaml (name → absolute path).
Default project location is ~/dsagt-projects/<name>/.  Shared bundled-content
KB lives alongside at ~/dsagt-projects/kb_index/ (provisioned by dsagt init).

Project directory layout::

    <project_dir>/
        .dsagt/
            config.yaml     # project configuration (MCP-server object settings)
            state.yaml      # session log + memory cursor (owned by the MCP server)
            explicit_memories.yaml, ...   # explicit memory
        trace_archive/      # tool execution records
        mlflow.db           # serverless MLflow SQLite trace store (created lazily)
        codes/<name>/       # registered codes (skill-standard dirs: SKILL.md + scripts/)
        skills/             # instruction-based agent skills
        kb_index/           # knowledge base collections
"""

import fcntl
import logging
import os
import re
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dsagt.knowledge import KnowledgeBase
from dsagt.provenance import CodeUseIndexer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AGENTS = ("claude", "goose", "cline", "codex", "opencode")

DEFAULT_PROJECTS_BASE = Path.home() / "dsagt-projects"
# Registry + shared KB live alongside projects under one visible tree —
# ``~/dsagt-projects/projects.yaml`` (name → path) and
# ``~/dsagt-projects/kb_index/`` (shared bundled-content KB provisioned by
# ``dsagt init``).  Migrated from ``~/.dsagt/`` on 2026-05-07.
REGISTRY_DIR = DEFAULT_PROJECTS_BASE
#: Files a cached source clone carries at its root: the commit the clone
#: was taken at and the branch or tag it was asked for.  Written by
#: ``clone_github``; the skill installer repeats the commit in
#: ``PROVENANCE.txt`` and re-clones a cache whose ref differs from the
#: one requested.
SOURCE_COMMIT_FILE = "SOURCE_COMMIT"
SOURCE_REF_FILE = "SOURCE_REF"
REGISTRY_FILE = REGISTRY_DIR / "projects.yaml"
RESERVED_PROJECT_NAMES = ("projects.yaml", "kb_index", ".skill_sources", ".tools")

# Per-project dsagt state lives under a hidden ``.dsagt/`` dir (alongside
# explicit memory): ``config.yaml`` (the MCP-server object settings the user
# chose at ``dsagt init``) and ``state.yaml`` (session log + memory cursor,
# owned by the MCP server).
CONFIG_DIRNAME = ".dsagt"
CONFIG_FILENAME = "config.yaml"
STATE_FILENAME = "state.yaml"


def config_path(pdir: Path) -> Path:
    """Path to a project's ``.dsagt/config.yaml``."""
    return Path(pdir) / CONFIG_DIRNAME / CONFIG_FILENAME


def state_path(pdir: Path) -> Path:
    """Path to a project's ``.dsagt/state.yaml``."""
    return Path(pdir) / CONFIG_DIRNAME / STATE_FILENAME


# Code defaults backfilled into a project's config on read (``_deep_merge``).
# They are code settings, held here as the one definition and filled in for
# the MCP server and the KB; ``.dsagt/config.yaml`` and the ``dsagt init``
# prompts carry user choices only.  Embedding is local (BYOA, no
# credentials).  ``chunk_size`` default
# in :class:`~dsagt.knowledge.KnowledgeBase`; ``skills.populate_native`` in
# :meth:`AgentSetup.setup_skills`.
DEFAULTS = {
    "embedding": {
        "backend": "local",
        "model": "BAAI/bge-small-en-v1.5",
        "base_url": "",
    },
    # External agent-skill catalogs.  ``sources`` are GitHub repos whose
    # SKILL.md skills get indexed into per-source catalog collections for
    # ``search_skills``.  This is the default when a project picks none.
    "skills": {
        "sources": [
            {
                "name": "genesis",
                "url": "https://github.com/AI-ModCon/genesis-skills",
                "branch": "main",
                "subdir": "skills",
            },
        ],
    },
    # Episodic memory: the periodic pass's MemoryExtractor subscriber.  ``enabled``
    # is a compute/storage opt-in that mechanically chunks/tags/embeds each
    # completed turn into session_memory (no credentials).
    "episodic": {
        "enabled": False,
        # Recency weighting for session_memory retrieval: a newer turn edges out
        # a stale one without contradiction detection.  Half-life in days (a
        # *boost*, never a penalty — durable old turns keep full relevance).
        "recency_half_life_days": 14,
    },
}

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


USER_ENV_FILE = Path.home() / ".config" / "dsagt" / "env"


def load_user_env(path: Path = USER_ENV_FILE) -> list[str]:
    """Load credentials from the user-level env file into ``os.environ``.

    ``~/.config/dsagt/env`` holds ``KEY=VALUE`` lines (an ``export`` prefix,
    quotes and ``#`` comments are accepted) for the secrets an agent cannot
    hand to its MCP children: codex and cline start ``dsagt-server`` with only
    the env block baked into their config, never the shell, so
    ``MLFLOW_TRACKING_API_KEY`` / ``EMBEDDING_API_KEY`` exported in a terminal
    never arrive.  The file is the ``~/.netrc`` pattern — in ``$HOME``, mode
    600, never inside a project or an agent config, which is the line the
    credential policy draws.  A key already in the environment wins, so a
    shell export still overrides the file.  Returns the names it set.
    """
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


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


def build_config(
    project_name: str,
    agent: str,
    *,
    knowledge: dict | None = None,
    skills: dict | None = None,
    episodic: dict | None = None,
    readiness: dict | None = None,
) -> dict:
    """Assemble a project's ``.dsagt/config.yaml`` body.

    The schema is a strict 1:1 mirror of the choices ``dsagt init`` offers —
    every key here corresponds to an init prompt and vice versa:

    - ``project`` / ``agent`` — identity (agent + name/location are prompted).
    - ``knowledge.collections`` — the packaged document collections chosen
      (default none; the bundled ``tools`` collection is always provisioned).
    - ``skills.sources`` — the skill-catalog repos chosen.
    - ``episodic`` — written *only when the user opted in* (it's an opt-in, so a
      disabled project stays minimal and backfills ``enabled: false`` on read).
    - ``readiness`` — the AI-readiness check setting (``auto_assess``; see
      :mod:`dsagt.readiness`), written when init asked the question.

    Everything else (embedding backend, chunk_size, populate_native)
    is a code default backfilled on read — NOT a written choice.  Credentials
    are never here (shell env only); no MLflow port (serverless sqlite store).
    """
    body = {
        "project": project_name,
        "agent": agent,
        "knowledge": knowledge or {"collections": []},
        "skills": skills or {"sources": list(DEFAULTS["skills"]["sources"])},
    }
    if episodic:
        body["episodic"] = episodic
    if readiness is not None:
        body["readiness"] = readiness
    return body


def default_config_content(
    project_name: str,
    agent: str,
    *,
    knowledge: dict | None = None,
    skills: dict | None = None,
    episodic: dict | None = None,
    readiness: dict | None = None,
) -> str:
    """Serialize :func:`build_config` to YAML for ``.dsagt/config.yaml``."""
    body = build_config(
        project_name,
        agent,
        knowledge=knowledge,
        skills=skills,
        episodic=episodic,
        readiness=readiness,
    )
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
    """Replace the registry file in one step, so a reader never finds it
    half written."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    scratch = REGISTRY_FILE.with_name(REGISTRY_FILE.name + ".tmp")
    scratch.write_text(yaml.dump(registry, default_flow_style=False))
    os.replace(scratch, REGISTRY_FILE)


@contextmanager
def _registry_update():
    """The registry as a dict to change in place; saved when the block ends.

    The read, the change and the write happen under an exclusive lock, so
    two ``dsagt init`` runs at once both end up registered.
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_DIR / ".projects.yaml.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = _load_registry()
        yield registry
        _save_registry(registry)


def register_project(name: str, path: Path) -> None:
    """Add or update a project in the registry."""
    with _registry_update() as registry:
        registry[name] = str(path.resolve())


def list_projects() -> dict[str, str]:
    """Return all registered projects as {name: path}."""
    return _load_registry()


def kb_from_config(config: dict, index_dir: Path | None = None) -> "KnowledgeBase":
    """Build a KnowledgeBase from a resolved project config.

    Resolves the project's embedding backend so callers (CLI ``skills`` group,
    catalog sync) get a KB wired to the same embedder the project uses.
    Defaults to ``<project_dir>/kb_index``.
    """
    pdir = Path(config["project_dir"])
    emb = config.get("embedding", {})
    backend = emb.get("backend", "local")
    recency = _recency_half_life(config)
    if backend == "local":
        model = emb.get("model")
        if model and "/" not in str(model):
            model = None
        return KnowledgeBase(
            index_dir=index_dir or (pdir / "kb_index"),
            default_embedder=backend,
            model=model,
            recency_half_life_days=recency,
        )
    return KnowledgeBase(
        index_dir=index_dir or (pdir / "kb_index"),
        default_embedder=backend,
        model=emb.get("model"),
        base_url=emb.get("base_url"),
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        recency_half_life_days=recency,
    )


def _recency_half_life(config: dict) -> float | None:
    """Episodic recency half-life (days) when enabled, else ``None`` (off).

    Recency weighting only matters for ``session_memory``, which only has
    content when episodic memory is enabled — so it's gated on that opt-in.
    """
    epi = config.get("episodic", {}) or {}
    return epi.get("recency_half_life_days") if epi.get("enabled") else None


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
    cfg_file = config_path(pdir)

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_file}")

    raw = yaml.safe_load(cfg_file.read_text()) or {}

    config = _deep_merge(DEFAULTS, raw)
    config = resolve_env_vars(config)
    config["project_dir"] = str(pdir)

    _validate(config)
    return config


def read_config_file(pdir: Path) -> dict:
    """Read a project's raw ``.dsagt/config.yaml`` by path (no registry, no
    defaults merge).  Returns ``{}`` if absent — used to prefill the
    re-run ``dsagt init`` dialogue with current values.
    """
    cfg_file = config_path(pdir)
    if not cfg_file.exists():
        return {}
    return yaml.safe_load(cfg_file.read_text()) or {}


def write_config_file(pdir: Path, config: dict) -> None:
    """Write a project's ``.dsagt/config.yaml`` (creates ``.dsagt/`` if
    needed).  ``config`` is the trimmed schema from :func:`build_config`.
    """
    cfg_file = config_path(pdir)
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _validate(config: dict) -> None:
    """Validate required fields and values."""
    if not config.get("project"):
        raise ValueError("'project' is required in .dsagt/config.yaml")

    agent = config.get("agent")
    if not agent:
        raise ValueError("'agent' is required in .dsagt/config.yaml")
    if agent not in VALID_AGENTS:
        raise ValueError(f"'agent' must be one of {VALID_AGENTS}, got '{agent}'")


# ---------------------------------------------------------------------------
# Session state (`.dsagt/state.yaml`) — owned by the MCP server
# ---------------------------------------------------------------------------


def _empty_state() -> dict:
    return {"sessions": [], "memory_cursor": {}}


def read_state(pdir: Path) -> dict:
    """Read ``.dsagt/state.yaml``; return an empty skeleton if absent."""
    sp = state_path(pdir)
    if not sp.exists():
        return _empty_state()
    return yaml.safe_load(sp.read_text()) or _empty_state()


def write_state(pdir: Path, state: dict) -> None:
    """Write ``.dsagt/state.yaml`` (creates ``.dsagt/`` if needed)."""
    sp = state_path(pdir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(yaml.dump(state, default_flow_style=False, sort_keys=False))


def append_session(pdir: Path) -> dict:
    """Append a new session entry and return it.

    Called by the MCP server at startup — the server owns session-id
    minting now (not ``dsagt start``), so a bare-launched agent gets a
    session id too.  Id is a monotonic per-project counter.
    """
    state = read_state(pdir)
    sessions = state.setdefault("sessions", [])
    next_id = max((s.get("id", 0) for s in sessions), default=0) + 1
    entry = {
        "id": next_id,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    sessions.append(entry)
    state.setdefault("memory_cursor", {})
    write_state(pdir, state)
    return entry


def current_session(pdir: Path) -> dict | None:
    """The most recent session entry, or ``None`` if no session has run."""
    sessions = read_state(pdir).get("sessions") or []
    return sessions[-1] if sessions else None


def session_tag(project: str, session_id: int) -> str:
    """The trace-correlation tag for a session: ``<project>-<n>``."""
    return f"{project}-{session_id}"


def current_session_tag(pdir: Path, project: str) -> str | None:
    """The trace tag of the current session, or ``None`` if none exists.

    Used by ``dsagt-run`` to tag tool spans with the same session the MCP
    server minted, read from ``state.yaml``.
    """
    cur = current_session(pdir)
    if cur is None:
        return None
    return session_tag(project, cur["id"])


def update_cursor(pdir: Path, **fields) -> None:
    """Merge fields into ``state.yaml``'s ``memory_cursor`` map."""
    state = read_state(pdir)
    state.setdefault("memory_cursor", {}).update(fields)
    write_state(pdir, state)


def record_trace_source(pdir: Path, source) -> None:
    """Stamp the current session's trace-source token into ``state.yaml``.

    ``source`` is the agent-shaped session token from
    :meth:`dsagt.traces.Reader.active_source` (a transcript path, DB session id,
    or session-dir name).  The MCP server records it once the live collector
    resolves a session, so the *next* session's catch-up can pin this exact
    session — for *every* agent — when re-collecting turns an ungraceful
    shutdown left unlogged.  No-op if no session has been minted yet.
    """
    state = read_state(pdir)
    sessions = state.get("sessions") or []
    if not sessions:
        return
    if sessions[-1].get("trace_source") == source:
        return  # already recorded — avoid churning the file
    sessions[-1]["trace_source"] = source
    write_state(pdir, state)


# ---------------------------------------------------------------------------
# Project initialization
# ---------------------------------------------------------------------------


def _collection_exists(path: Path) -> bool:
    """Return True if *path* looks like a persisted KB collection directory.

    Accepts ChromaDB-in-a-dir layouts, routed external collections, and
    bare ChromaDB sqlite files (as produced by the KB asset build for
    description-only collections).
    """
    return path.is_dir() and (
        (path / "chroma_ids.json").exists()
        or (path / "route.json").exists()
        or (path / "chroma.sqlite3").exists()
    )


def setup_runtime_kb(
    base_index_dir: Path,
    runtime_dir: Path,
    collections: list[str] | None = None,
) -> Path:
    """Copy base KB collections into a project's kb_index directory.

    Creates ``<runtime_dir>/kb_index`` if missing.  For each collection
    under *base_index_dir* that looks populated and has no project-local
    twin yet, **copies** the entire
    collection directory into the project's kb_index.

    *collections*, when given, is an allowlist of collection-directory
    names to copy — so a project gets exactly its requested asset set even
    when the shared KB holds more (e.g. heavy collections another project
    installed).  ``None`` copies every populated collection.

    Why copy instead of symlink: different projects on the same machine
    may run different dsagt versions, and a symlink would let one
    project's KB rebuild mutate every project's view
    of bundled content.  A copy pins each project to whatever bundled
    content was current when the project first ran.

    Existing project-local collections are left alone, so an agent's
    saved tools / skills / ingests are preserved across re-runs.
    """
    runtime_kb_dir = runtime_dir / "kb_index"
    runtime_kb_dir.mkdir(parents=True, exist_ok=True)

    if not base_index_dir.exists():
        return runtime_kb_dir

    allow = set(collections) if collections is not None else None
    for collection_dir in base_index_dir.iterdir():
        if allow is not None and collection_dir.name not in allow:
            continue
        if not _collection_exists(collection_dir):
            continue
        dest = runtime_kb_dir / collection_dir.name
        if dest.exists():
            continue
        # Resolve in case the source collection itself is a symlink
        # (older projects that used the symlink path).
        shutil.copytree(collection_dir.resolve(), dest)

    return runtime_kb_dir


def _provision_kb(
    pdir: Path,
    include: list[str] | None,
    exclude: list[str] | None,
    embedding: dict | None = None,
) -> list[str]:
    """Build the requested KB assets into the shared cache, then copy that
    set into the project.  Returns the resolved asset names.

    The first project on a machine pays the one-time build (the base-skill
    codes + genesis catalog by default); later projects just copy.  The copy is
    scoped to the requested set, so a project gets exactly what was asked
    for regardless of what else the shared cache holds.

    Best-effort: a build failure (offline, no embedding model) degrades to
    an empty-but-valid project KB with a warning rather than aborting
    ``dsagt init`` — the build retries on a later ``dsagt init``.
    """
    from dsagt.commands.setup_core_kb import (
        asset_collection_name,
        ensure_assets,
        resolve_assets,
    )

    assets = resolve_assets(include=include, exclude=exclude)
    if not assets:
        # ``--exclude all``: a valid project with an empty KB.
        (pdir / "kb_index").mkdir(parents=True, exist_ok=True)
        return assets

    shared = REGISTRY_DIR / "kb_index"
    first_ever = not shared.exists() or not any(
        _collection_exists(c) for c in shared.iterdir()
    )
    # Which requested assets aren't in the shared cache yet — i.e. what this
    # init will actually build (and narrate).  Empty → silent fast path.
    pending = [
        a for a in assets if not _collection_exists(shared / asset_collection_name(a))
    ]
    if pending:
        if first_ever:
            print(
                "Performing initial dsagt setup — first project on this "
                "machine (one-time, may take a few minutes):",
                flush=True,
            )
        else:
            print("Provisioning knowledge base assets:", flush=True)

    emb = embedding or DEFAULTS["embedding"]
    backend = emb.get("backend", "local")
    if backend == "local":
        embedder_kwargs = {"model": emb.get("model")}
    else:
        embedder_kwargs = {
            "model": emb.get("model"),
            "base_url": emb.get("base_url"),
            "api_key": os.environ.get("EMBEDDING_API_KEY", ""),
        }

    try:
        ensure_assets(
            assets,
            shared,
            embedding_backend=backend,
            embedder_kwargs=embedder_kwargs,
        )
    except Exception as e:  # noqa: BLE001 — never let a build failure block init
        print(
            f"  Warning: could not build knowledge base ({e}).  The project "
            "works without it; re-run `dsagt init` for a project once a "
            "network / embedding backend is available to install the shared "
            "core KB.",
            flush=True,
        )

    wanted = [asset_collection_name(a) for a in assets]
    setup_runtime_kb(shared, pdir, collections=wanted)

    if pending:
        print("  Knowledge base ready.", flush=True)
    return assets


def _provision_base_skills(
    pdir: Path, embedding: dict | None, *, index_codes: bool
) -> None:
    """Install the base skills (``skills.base_skills``) from their upstream
    repositories into ``<project>/skills/`` and register their codes.

    The base-skill code vectors come with the copied ``codes`` collection,
    embedded once by the shared knowledge-base build, so this step loads no
    embedding model.  *index_codes* is for a project provisioned without the
    ``codes`` asset, which has no copied collection to carry them: the specs
    are then embedded into the project's own knowledge base.  A failed fetch
    is printed, not raised: the project works without the skills, and a
    re-run of ``dsagt init`` installs them once the network is available.
    """
    from dsagt.skills import base_skills, install_base_skills

    names = ", ".join(entry["name"] for entry in base_skills())
    print(f"Installing base skills ({names}) …", flush=True)
    kb = None
    if index_codes:
        print("  Indexing base-skill codes …", flush=True)
        kb = kb_from_config(
            {"project_dir": str(pdir), "embedding": embedding or DEFAULTS["embedding"]}
        )
    try:
        results = install_base_skills(pdir, kb=kb)
    except Exception as e:  # noqa: BLE001 — offline init must still complete
        print(
            f"  Warning: could not install the base skills ({e}).  Re-run "
            "`dsagt init` with network access to install them.",
            flush=True,
        )
        return
    print(
        "  Base skills ready: "
        + ", ".join(f"{r['name']} ({r['action']})" for r in results),
        flush=True,
    )


def init_project(
    project_name: str,
    agent: str,
    location: Path | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    *,
    embedding: dict | None = None,
    knowledge: dict | None = None,
    skills: dict | None = None,
    episodic: dict | None = None,
    readiness: dict | None = None,
) -> Path:
    """Create or reconfigure a project — ``dsagt init`` is re-runnable.

    BYOA model: we lay down everything the user needs to point their own
    agent process at our MCP server.  The trace store is serverless
    (``sqlite:///<pdir>/mlflow.db``), so there's no port to pick — the
    MCP-server children resolve the store from the project dir.

    Idempotent: on a project that already has ``.dsagt/config.yaml`` this
    overwrites the config with the new choices and provisions any
    newly-requested KB assets.  It never deletes agent-saved data; the
    caller (``dsagt init`` in ``cli.py``) handles destructive deltas
    (agent switch, removed collections) with explicit per-change prompts.

    Knowledge base: provisioned with a chosen set of KB assets
    (``include`` / ``exclude``, default = the base-skill codes + genesis catalog),
    built once into the shared ``~/dsagt-projects/kb_index/`` and copied in.

    Returns the project directory.
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

    pdir.mkdir(parents=True, exist_ok=True)
    # ``mlflow.db`` is created lazily by the MLflow client on first span.
    for subdir in ("trace_archive", "skills", "codes", "audit", CONFIG_DIRNAME):
        (pdir / subdir).mkdir(parents=True, exist_ok=True)

    assets = _provision_kb(pdir, include, exclude, embedding=embedding)

    _provision_base_skills(pdir, embedding, index_codes="codes" not in assets)

    write_config_file(
        pdir,
        build_config(
            project_name,
            agent,
            knowledge=knowledge,
            skills=skills,
            episodic=episodic,
            readiness=readiness,
        ),
    )

    register_project(project_name, pdir)
    return pdir


def remove_collection(pdir: Path, collection: str) -> bool:
    """Delete a single KB collection directory from a project's ``kb_index``.

    Used by re-run ``dsagt init`` when the user opts to remove a collection
    that was dropped from the asset set.  Returns True if a directory was
    removed.  Caller must guard agent-populated collections (``code_use``,
    ``session_memory``) — this helper deletes whatever name it's given.
    """
    target = Path(pdir) / "kb_index" / collection
    if target.exists():
        shutil.rmtree(target)
        return True
    return False


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

    ``dsagt start`` runs the agent in the foreground and returns when it
    exits, so removal is a directory delete.
    """
    pdir = project_dir(project_name)

    if not keep_files and pdir.exists():
        shutil.rmtree(pdir)

    with _registry_update() as registry:
        registry.pop(project_name, None)
    return pdir


# ---------------------------------------------------------------------------
# Memory extraction orchestration
# ---------------------------------------------------------------------------


def catch_up_extraction(pdir: Path, config: dict, kb=None) -> dict:
    """Background post-session catch-up — run by the MCP server at startup.

    The MCP server owns the session lifecycle: each launch, it spawns this
    against a snapshot taken at startup, so it processes the *previous*
    session's trailing trace records, never the live one.  This means no
    reliable session-*end* trigger is required, and bare-launched agents get
    full parity.

    Two phases, both best-effort:

    1. **Code-execution indexing** (always): embed the previous session's
       ``<pdir>/trace_archive/`` records into the ``code_use`` collection via
       the shared :class:`~dsagt.provenance.CodeUseIndexer` — idempotent against
       the same ``.dsagt/code_use_acks.json`` the periodic pass uses, so the
       startup catch-up and the periodic pass never double-index.  No LLM, no
       credentials (local-backend default).
    2. **Chat-trace catch-up** (:func:`_catch_up_traces`): re-collect the
       previous session so any turns the periodic pass missed before an ungraceful
       shutdown still reach MLflow (and episodic memory).  Pinned to the
       trace-source token recorded in ``state.yaml`` (uniform across agents);
       transcript-qualified acks dedupe against the live pass, so only dangling
       turns emit.
    """
    pdir = Path(pdir)
    config = {**config, "project_dir": str(pdir)}
    # The server passes its own knowledge base so one embedder serves the
    # session; a second one costs a model load per start.
    if kb is None:
        kb = kb_from_config(config)

    try:
        code_use_indexed = 0
        try:
            # tick_traced: this catch-up runs off any tool-call trace, so tag
            # the indexer's kb.* writes dsagt.source=code_use rather than let
            # them orphan as untagged top-level traces.
            code_use_indexed = CodeUseIndexer(kb, pdir).tick_traced()
        except Exception as e:  # noqa: BLE001 — never let a background task crash
            logger.warning("Code execution indexing failed: %s", e)

        traces_caught_up = 0
        try:
            traces_caught_up = _catch_up_traces(pdir, config, kb)
        except Exception as e:  # noqa: BLE001 — never let a background task crash
            logger.warning("Trace catch-up failed: %s", e)

        return {
            "status": "ok",
            "code_use_indexed": code_use_indexed,
            "traces_caught_up": traces_caught_up,
        }
    finally:
        kb.close()


def _catch_up_traces(pdir: Path, config: dict, kb) -> int:
    """Re-collect the previous session's transcript; return turns newly emitted.

    Builds a trace collector pinned to the previous session's recorded
    trace-source token (and tagged with its session id), then runs one
    ``collect(include_last=True)``.  The collector's transcript-qualified ack files
    are shared with the live pass, so already-logged turns are skipped and only
    those lost to an ungraceful shutdown are emitted to MLflow + episodic memory.
    Uniform across agents — JSONL or SQLite — since the pin is the agent's own
    session token, not a transcript-file assumption.

    Returns 0 (a no-op) when there is no previous session, or it stamped no
    trace-source (a session too short for the periodic pass to record one), where
    guessing would risk reading the *new* session's records.
    """
    from dsagt.memory import episodic_consumers
    from dsagt.observability import experiment_name, resolve_tracking_uri
    from dsagt.traces import make_trace_collector

    sessions = read_state(pdir).get("sessions") or []
    if len(sessions) < 2:
        return 0
    # The most recent earlier session that recorded its source.  A session
    # that ended before the periodic pass stamped one (a headless prompt of a
    # few seconds) would otherwise hide the turns of the session before it.
    prev = next((s for s in reversed(sessions[:-1]) if s.get("trace_source")), None)
    if prev is None:
        return 0
    source = prev["trace_source"]

    project = config.get("project", "")
    prev_tag = session_tag(project, prev["id"])
    collector = make_trace_collector(
        config.get("agent"),
        pdir,
        project,
        prev_tag,
        resolve_tracking_uri(config),
        experiment=experiment_name(config),
        extra_consumers=episodic_consumers(config, kb, pdir, prev_tag),
        source=source,
    )
    if collector is None:
        return 0
    return collector.collect(include_last=True)
