"""
Knowledge-base asset builder — the engine behind ``dsagt init``'s KB setup.
``session._provision_kb`` calls :func:`resolve_assets` + :func:`ensure_assets`
to build the requested assets into the shared ``~/dsagt-projects/kb_index/``
once, then copies them per project.

Asset namespace (the ``--include`` / ``--exclude`` selectors on ``dsagt init``):
- ``tools``                bundled tool specs (cheap, local)
- skill catalogs           ``genesis`` (default), ``scientific``, ``composio``, …
- scientific collections   ``nemo_curator`` (clones the external repo; docs + papers only, not source)

:data:`DEFAULT_ASSETS` (bundled tools + the genesis skill catalog) is the
cheap set installed automatically on a machine's first project.  Embedding
config comes from the project's ``.dsagt/config.yaml`` (local backend by
default — no credentials needed).
"""

import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx

from dsagt.session import (
    REGISTRY_DIR,
    SOURCE_COMMIT_FILE,
    SOURCE_REF_FILE,
    _collection_exists,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = REGISTRY_DIR / "kb_index"

# Default exclusion patterns applied to every core-KB ingest unless a
# collection overrides them.  Goal: skip content that has low retrieval
# value for an agent learning to *use* a library, while keeping docs,
# tutorials, examples (concrete usage patterns), the main library source
# (call signatures + docstrings + type hints), and packaging metadata
# the agent needs to install dependencies correctly.
#
# What's excluded and why:
# - tests/, test/, conftest.py, test_*.py, *_test.py
#       Test internals teach how the library is *tested*, not how it's used.
# - __pycache__/, .git/, *.egg-info/, .pytest_cache/, .mypy_cache/
#       Build/cache artifacts.  Pure noise.
# - _*.py
#       Python convention for private modules — implementation detail, not API.
# - CHANGELOG*, HISTORY*
#       Historical, not how-to-use.
#
# Notably NOT excluded:
# - pyproject.toml, setup.py, setup.cfg
#       Packaging metadata.  The agent uses these to determine which
#       version of a library to install when registering a tool that
#       depends on it ("uv pip install nemo_curator>=X.Y").  Without
#       pyproject.toml in the index, the agent has to guess.
DEFAULT_EXCLUDE_PATTERNS = [
    "tests",
    "test",
    "conftest.py",
    "test_*.py",
    "*_test.py",
    "__pycache__",
    ".git",
    "*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    "_*.py",
    "CHANGELOG*",
    "HISTORY*",
]

# Collection definitions.  Each may set "exclude_patterns" to override
# DEFAULT_EXCLUDE_PATTERNS for that collection (e.g. if a library's tests
# happen to be the canonical usage examples).
COLLECTIONS = {
    "nemo_curator": {
        "description": """# NeMo Curator

NVIDIA's scalable data curation library for LLM training data.

## Key Topics
- Text filtering (heuristic and classifier-based)
- Deduplication (exact, fuzzy, semantic)
- Language identification
- Quality scoring and classification
- PII detection and removal

## Use For
Building data curation pipelines, understanding the available filters,
quality assessment strategies.
""",
        "sources": [
            {
                "type": "github",
                "url": "https://github.com/NVIDIA-NeMo/Curator",
                "branch": "main",
                # Docs only — the library source (and the notebook/script
                # tutorials) are deliberately omitted.  A vector index is a
                # poor way to retrieve code (the agent's native file search
                # beats it), and skipping the source tree keeps ingestion
                # fast for onboarding.  clone_github still pulls the top-level
                # files (README, pyproject.toml, …) for install metadata.
                "include": ["docs"],
            },
        ],
    },
}


_GITHUB_REPO = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def _github_owner_repo(url: str) -> tuple[str, str] | None:
    """``(owner, repo)`` for a GitHub URL in https or ssh form, else ``None``."""
    m = _GITHUB_REPO.match(url.strip())
    return (m["owner"], m["repo"]) if m else None


def _github_api_headers() -> dict[str, str]:
    """The API headers, with the shell's ``GITHUB_TOKEN`` when it has one.

    The token is read from the shell for the request and written nowhere;
    a private repository needs it, a public one does not.
    """
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "dsagt"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_github_tree(url: str, ref: str, into: Path) -> str:
    """Download the tree of a GitHub repository at *ref* into *into*; return the commit.

    Two requests over the httpx client dsagt already has: the commit the
    ref resolves to (which also checks the ref exists), then the tarball at
    that commit, unpacked so *into* holds the repository root.  Raises
    ``httpx.HTTPStatusError`` for a ref or repository the API refuses, which
    for a private repository means the shell has no ``GITHUB_TOKEN``.
    """
    owner_repo = _github_owner_repo(url)
    if owner_repo is None:
        raise ValueError(f"{url!r} is not a GitHub repository URL")
    owner, repo = owner_repo
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        commit = client.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}",
            headers=_github_api_headers(),
        )
        commit.raise_for_status()
        sha = commit.json()["sha"]
        tarball = client.get(
            f"https://api.github.com/repos/{owner}/{repo}/tarball/{sha}",
            headers=_github_api_headers(),
        )
        tarball.raise_for_status()
    archive = into.parent / f"{repo}.tar.gz"
    archive.write_bytes(tarball.content)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        # GitHub's tarball wraps the tree in one <owner>-<repo>-<sha7>/ directory.
        top = members[0].name.split("/", 1)[0]
        tar.extractall(into.parent, filter="data")
    (into.parent / top).rename(into)
    archive.unlink()
    return sha


def _git_clone(url: str, ref: str, into: Path) -> str:
    """Clone *url* at *ref* into *into* with the user's git; return the commit."""
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, url, str(into)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git clone failed: {result.stderr}")
    head = subprocess.run(
        ["git", "-C", str(into), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if head.returncode != 0 or not head.stdout.strip():
        raise RuntimeError(f"Git rev-parse failed after clone: {head.stderr}")
    shutil.rmtree(into / ".git", ignore_errors=True)
    return head.stdout.strip()


def clone_github(
    url: str, dest: Path, branch: str = "main", include: list[str] | None = None
):
    """Fetch a GitHub repository's tree at *branch* into *dest*, optionally
    keeping only specific directories.

    The tree comes as a tarball over HTTPS (:func:`fetch_github_tree`), so
    the end-user install, which is ``pip install``, needs no git; the one
    fallback is the user's git for a repository the API refuses, which is a
    private repository the shell has no ``GITHUB_TOKEN`` for but the user's
    ssh key can reach.  When *include* is set, the named subdirectories are
    copied AND any top-level files at the repo root (README, pyproject.toml,
    setup.py, LICENSE, etc.).  Top-level files are usually small and contain
    critical packaging metadata the agent needs to install dependencies
    correctly when it registers tools against the library.  The commit the
    tree was taken at and the *branch* (or tag) asked for are written to
    ``<dest>/SOURCE_COMMIT`` and ``<dest>/SOURCE_REF`` so a consumer can
    tell what a cached clone holds.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "repo"
        if _github_owner_repo(url) is None:
            commit = _git_clone(url, branch, tmp_path)
        else:
            try:
                commit = fetch_github_tree(url, branch, tmp_path)
            except httpx.HTTPStatusError as err:
                if err.response.status_code not in (401, 403, 404) or not shutil.which(
                    "git"
                ):
                    raise
                logger.info(
                    "GitHub refused %s at %s (%s); cloning with git instead",
                    url,
                    branch,
                    err.response.status_code,
                )
                commit = _git_clone(url, branch, tmp_path)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / SOURCE_COMMIT_FILE).write_text(commit + "\n")
        (dest / SOURCE_REF_FILE).write_text(branch + "\n")

        if include:
            # Copy the requested subdirectories.
            for subdir in include:
                src = tmp_path / subdir
                if src.exists():
                    shutil.copytree(src, dest / subdir, dirs_exist_ok=True)
            # Plus any top-level files at the repo root (pyproject.toml,
            # setup.py, README, LICENSE, ...).  These are tiny and the
            # agent uses them to resolve install commands.
            for f in tmp_path.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
        else:
            shutil.copytree(tmp_path, dest, dirs_exist_ok=True)


def download_arxiv(paper_id: str, dest: Path):
    """Download arXiv paper (source if available, else PDF)."""
    client = httpx.Client(timeout=60.0, follow_redirects=True)

    try:
        # Try source tarball first
        response = client.get(f"https://arxiv.org/e-print/{paper_id}")
        if response.status_code == 200:
            tar_path = dest / f"{paper_id}.tar.gz"
            tar_path.write_bytes(response.content)
            try:
                with tarfile.open(tar_path, "r:*") as tar:
                    # FIX 3: Add filter parameter for Python 3.14 compatibility
                    tar.extractall(dest / paper_id, filter="data")
                tar_path.unlink()
                return
            except tarfile.ReadError:
                tar_path.unlink()

        # Fall back to PDF
        response = client.get(f"https://arxiv.org/pdf/{paper_id}.pdf")
        response.raise_for_status()
        (dest / f"{paper_id}.pdf").write_bytes(response.content)
    finally:
        client.close()


def setup_collection(
    name: str,
    config: dict,
    index_dir: Path,
    embedder_kwargs: dict,
    embedding_backend: str,
    vector_db: str,
    kb=None,
) -> dict:
    """Download sources and ingest a collection.

    When *kb* is provided, the existing KnowledgeBase is reused — its
    embedder cache stays warm so the local model isn't reloaded per
    collection.  When None (default for backwards
    compat with direct callers), a fresh KB is constructed and closed
    around this call.

    ``ensure_assets`` always passes a shared *kb* so a multi-asset build
    pays the model-load cost once, not N times.  It also prints the
    user-facing progress line, so this function stays quiet.
    """
    with tempfile.TemporaryDirectory() as tmp:
        download_dir = Path(tmp) / name
        download_dir.mkdir()

        # Download all sources.  If any source fails, raise immediately —
        # silently skipping a 404 arxiv URL or a broken git clone produces
        # a half-built collection the user can't see is incomplete.  Better
        # to fail loudly and have the user re-run after fixing the issue.
        for source in config["sources"]:
            if source["type"] == "github":
                clone_github(
                    source["url"],
                    download_dir,
                    branch=source.get("branch", "main"),
                    include=source.get("include"),
                )
            elif source["type"] == "arxiv":
                download_arxiv(source["id"], download_dir)
            else:
                raise ValueError(f"Unknown source type: {source['type']!r}")

        # Write DESCRIPTION.md
        (download_dir / "DESCRIPTION.md").write_text(config["description"])

        owned_kb = kb is None
        if owned_kb:
            from dsagt.knowledge import KnowledgeBase

            kb = KnowledgeBase(
                index_dir=index_dir,
                default_embedder=embedding_backend,
                model=embedder_kwargs.get("model"),
                base_url=embedder_kwargs.get("base_url"),
                api_key=embedder_kwargs.get("api_key"),
            )
        try:
            result = kb.ingest(
                download_dir,
                exclude_patterns=config.get("exclude_patterns")
                or DEFAULT_EXCLUDE_PATTERNS,
            )
            skipped = result.get("skipped_files", 0)
            miss_msg = f", {skipped} file misses" if skipped else ""
            print(
                f"  {name}: {result['chunks']} chunks{miss_msg}",
                flush=True,
            )
            return result
        finally:
            if owned_kb:
                kb.close()


def _current_dsagt_version() -> str:
    """Return the installed dsagt package version, or ``"unknown"`` if absent."""
    try:
        from importlib.metadata import version

        return version("dsagt")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Installable KB assets — the namespace for ``dsagt init``'s
# ``--include`` / ``--exclude`` selectors.
#
# Three kinds, all built into the shared ``~/dsagt-projects/kb_index/`` and
# copied per-project:
#   "codes"        bundled tool specs (package data; cheap, fully local)
#   <catalog>      a skill-catalog source from ``skills.KNOWN_SOURCES``
#                  (e.g. "genesis", "k-dense-ai", "composio", "antigravity")
#   <collection>   a heavy scientific doc collection from ``COLLECTIONS``
#                  (e.g. "nemo_curator" — clones external repos)
#
# DEFAULT_ASSETS is the cheap core a first-ever ``dsagt init`` installs
# automatically; everything else is opt-in via ``--include``.
# ---------------------------------------------------------------------------

#: The default per-project / first-init asset set: bundled tools + the
#: genesis skill catalog.  Kept deliberately cheap (one small local embed +
#: one git clone) so onboarding needs no manual step.
DEFAULT_ASSETS: tuple[str, ...] = ("codes", "genesis")


def all_assets() -> list[str]:
    """Every installable asset name, in canonical install order (cheap → heavy)."""
    from dsagt.skills import KNOWN_SOURCES

    return ["codes", *KNOWN_SOURCES, *COLLECTIONS]


def asset_collection_name(asset: str) -> str:
    """The ``kb_index`` collection directory a given asset materializes as."""
    from dsagt.registry import CODES_COLLECTION, CATALOG_COLLECTION_PREFIX
    from dsagt.skills import KNOWN_SOURCES, _repo_slug

    if asset == "codes":
        return CODES_COLLECTION
    if asset in KNOWN_SOURCES:
        return CATALOG_COLLECTION_PREFIX + _repo_slug(KNOWN_SOURCES[asset]["url"])
    if asset in COLLECTIONS:
        return asset
    raise ValueError(f"unknown KB asset: {asset!r}")


def resolve_assets(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Resolve the requested asset set from ``--include`` / ``--exclude``.

    - ``include`` and ``exclude`` are mutually exclusive.
    - The literal ``"all"`` expands to every installable asset.
    - No selector → :data:`DEFAULT_ASSETS`.
    - ``--exclude all`` → ``[]`` (empty stub; the project's KB is created but
      holds no bundled content).

    Returns names in canonical install order so a build pays the cheap
    (local) assets before the heavy (network) ones.
    """
    if include and exclude:
        raise ValueError("--include and --exclude are mutually exclusive")
    known = all_assets()

    def _validate(names: list[str]) -> None:
        bad = [n for n in names if n != "all" and n not in known]
        if bad:
            raise ValueError(
                f"unknown KB asset(s): {', '.join(bad)}.  "
                f"Choose from: {', '.join(known)} (or 'all')."
            )

    if include is not None:
        _validate(include)
        if "all" in include:
            return list(known)
        sel = set(include)
        return [a for a in known if a in sel]
    if exclude is not None:
        _validate(exclude)
        if "all" in exclude:
            return []
        excl = set(exclude)
        return [a for a in DEFAULT_ASSETS if a not in excl]
    return list(DEFAULT_ASSETS)


def _model_is_cached(model_id: str) -> bool:
    """True if *model_id* is already in the local HuggingFace cache.

    Best-effort: on any error (offline, API change) returns True so we never
    print a misleading "downloading" message for an already-present model.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        # The local embedder loads the repository's ONNX export; a str path
        # back means the file is cached (None / sentinel ⇒ not cached).
        return isinstance(try_to_load_from_cache(model_id, "onnx/model.onnx"), str)
    except Exception:
        return True


#: Written into the shared ``codes`` collection by :func:`_build_bundled_tools`;
#: :func:`ensure_assets` rebuilds the collection when the file differs from
#: :func:`_codes_stamp`, so a dsagt upgrade or a changed base-skill code
#: refreshes every later init without a manual cache wipe.
CODES_STAMP_FILE = "CODES_STAMP"


def _codes_entries() -> list[tuple[str, dict]]:
    """The text and metadata of every entry the shared ``codes`` collection holds.

    The package codes are their SKILL.md files.  The base-skill codes are
    rendered from :func:`dsagt.skills.base_skill_code_specs` by the same
    function ``save_tool`` uses, so the embedded text equals the file init
    writes into the project.  Neither depends on a project, which is what
    lets the collection be built once and copied.
    """
    from dsagt.registry import (
        CodeRegistry,
        _parse_frontmatter,
        code_metadata,
        render_code_spec,
    )
    from dsagt.skills import base_skill_code_specs

    entries: list[tuple[str, dict]] = []
    for path in sorted(CodeRegistry._PACKAGE_CODES_DIR.glob("*/SKILL.md")):
        spec = _parse_frontmatter(path)
        if spec.get("name"):
            spec.setdefault("executable", "")
            entries.append((path.read_text(), code_metadata(spec, "bundled")))
    for spec in base_skill_code_specs():
        text = render_code_spec(spec)
        wrapped = _parse_frontmatter_text(text)
        entries.append((text, code_metadata(wrapped, "base-skill")))
    return entries


def _parse_frontmatter_text(text: str) -> dict:
    import yaml

    return yaml.safe_load(text.split("---", 2)[1])


def _codes_stamp(entries: list[tuple[str, dict]]) -> str:
    """A digest of the dsagt version and every entry text."""
    import hashlib

    digest = hashlib.sha256(_current_dsagt_version().encode())
    for text, _ in entries:
        digest.update(text.encode())
    return digest.hexdigest()


def _codes_current(index_dir: Path) -> bool:
    """Whether the shared ``codes`` collection exists and matches the package."""
    coll_dir = index_dir / "codes"
    stamp = coll_dir / CODES_STAMP_FILE
    return (
        _collection_exists(coll_dir)
        and stamp.exists()
        and stamp.read_text().strip() == _codes_stamp(_codes_entries())
    )


def _build_bundled_tools(kb, index_dir: Path) -> int:
    """(Re)build the shared ``codes`` collection: package codes and base-skill codes.

    Wipe-and-rebuild, then write :data:`CODES_STAMP_FILE`.  Returns the
    number indexed.
    """
    from dsagt.registry import CODES_COLLECTION

    coll_dir = index_dir / CODES_COLLECTION
    if coll_dir.exists():
        shutil.rmtree(coll_dir)

    entries = _codes_entries()
    current_version = _current_dsagt_version()
    kb.add_entries(
        texts=[text for text, _ in entries],
        collection=CODES_COLLECTION,
        metadatas=[{**meta, "dsagt_version": current_version} for _, meta in entries],
    )
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / CODES_STAMP_FILE).write_text(_codes_stamp(entries) + "\n")
    return len(entries)


def ensure_assets(
    asset_names: list[str],
    index_dir: Path,
    *,
    embedding_backend: str = "local",
    embedder_kwargs: dict | None = None,
    vector_db: str = "chroma",
    rebuild: bool = False,
    kb=None,
) -> dict:
    """Build the named *asset_names* into *index_dir* (the shared KB cache).

    Idempotent: an asset whose collection already exists is skipped unless
    *rebuild*, so a second ``dsagt init`` pays nothing.  Reuses *kb* when
    given (keeps the embedder model warm); otherwise constructs and closes
    one.  Best-effort per asset for catalogs (a clone failure warns and
    continues); a heavy-collection failure propagates.

    Returns ``{"built": [...], "skipped": [...]}``.
    """
    from dsagt.skills import KNOWN_SOURCES, sync_source

    index_dir.mkdir(parents=True, exist_ok=True)

    # Decide what actually needs building first, so we stay silent and cheap
    # when a later init's requested set is already cached.
    to_build = [
        a
        for a in asset_names
        if rebuild
        or (
            not _codes_current(index_dir)
            if a == "codes"
            else not _collection_exists(index_dir / asset_collection_name(a))
        )
    ]
    skipped = [a for a in asset_names if a not in to_build]
    if not to_build:
        return {"built": [], "skipped": skipped}

    owned = kb is None
    if owned:
        from dsagt.knowledge import KnowledgeBase

        ek = embedder_kwargs or {}
        kb = KnowledgeBase(
            index_dir=index_dir,
            default_embedder=embedding_backend,
            model=ek.get("model"),
            base_url=ek.get("base_url"),
            api_key=ek.get("api_key"),
        )

    built: list[str] = []
    try:
        # The model loads on the first embed below; announce it so the load
        # (or one-time download) isn't a silent pause.  Distinguish the two so
        # we don't claim a download when the model is already cached.  API
        # backend has no local model to load.
        if embedding_backend == "local":
            from dsagt.knowledge import LocalEmbedder

            model_id = (embedder_kwargs or {}).get(
                "model"
            ) or LocalEmbedder.DEFAULT_MODEL
            if _model_is_cached(model_id):
                print("  Loading embedding model …", flush=True)
            else:
                print("  Downloading embedding model (one-time) …", flush=True)

        for asset in to_build:
            if asset == "codes":
                print("  Indexing bundled tools and base-skill codes …", flush=True)
                _build_bundled_tools(kb, index_dir)
                built.append(asset)
            elif asset in KNOWN_SOURCES:
                print(f"  Fetching skill catalog: {asset} …", flush=True)
                try:
                    sync_source(asset, kb=kb, force=rebuild)
                    built.append(asset)
                except Exception as e:  # noqa: BLE001 — best-effort, keep going
                    print(f"    skipped {asset} ({e})", flush=True)
            else:  # heavy scientific collection
                print(
                    f"  Fetching collection: {asset} (may take a few minutes) …",
                    flush=True,
                )
                coll = index_dir / asset_collection_name(asset)
                if coll.exists():
                    shutil.rmtree(coll)
                setup_collection(
                    asset,
                    COLLECTIONS[asset],
                    index_dir,
                    embedder_kwargs=embedder_kwargs or {},
                    embedding_backend=embedding_backend,
                    vector_db=vector_db,
                    kb=kb,
                )
                built.append(asset)
    finally:
        if owned:
            kb.close()

    return {"built": built, "skipped": skipped}
