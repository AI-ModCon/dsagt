"""
dsagt-setup-kb: Build core knowledge base collections.

Downloads and indexes:
- nemo_curator: NVIDIA NeMo Curator (code, docs, tutorials)
- aidrin: AI Data Readiness Inspector (code, papers)

The embedding service is configured via CLI flags or environment variables
(``LLM_API_KEY``, ``OPENAI_BASE_URL``, ``EMBEDDING_MODEL``).  API-backed
embedding of the full core KB typically takes 15-30 minutes.

Usage:
    dsagt-setup-kb
    dsagt-setup-kb --index-dir ./my_kb_index
    dsagt-setup-kb --collection nemo_curator
    dsagt-setup-kb --embedding-base-url https://api.example.com/v1 \\
                   --embedding-api-key sk-... \\
                   --embedding-model text-embedding-3-small
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx

from dsagt.session import REGISTRY_DIR, _collection_exists

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
Building data curation pipelines, understanding filter implementations,
quality assessment strategies.
""",
        "sources": [
            {
                "type": "github",
                "url": "https://github.com/NVIDIA/NeMo-Curator",
                "branch": "main",
                # Top-level subdirs to clone.  examples + tutorials are
                # kept on purpose: agents writing pipelines benefit from
                # concrete usage patterns more than from prose docs alone.
                "include": ["docs", "nemo_curator", "tutorials", "examples"],
            },
        ],
    },
    "aidrin": {
        "description": """# AIDRIN - AI Data Readiness Inspector

Framework for assessing data readiness for AI/ML applications.

## Key Topics
- Data quality metrics (completeness, outliers, duplicates)
- Fairness and bias assessment
- Privacy evaluation
- FAIR principle compliance
- Feature importance analysis

## Use For
Understanding data quality requirements, assessment metrics,
readiness evaluation for ML pipelines.
""",
        "sources": [
            {
                "type": "github",
                # FIX 1: Correct URL (AIDRIN is uppercase)
                "url": "https://github.com/kaveenh/AIDRIN",
                # FIX 2: Correct branch (develop, not main)
                "branch": "develop",
            },
            {"type": "arxiv", "id": "2406.19256"},  # AIDRIN paper
            {"type": "arxiv", "id": "2404.05779"},  # Data Readiness Survey
        ],
    },
}


def clone_github(
    url: str, dest: Path, branch: str = "main", include: list[str] | None = None
):
    """Clone a GitHub repo, optionally keeping only specific directories.

    When *include* is set, the named subdirectories are copied AND any
    top-level files at the repo root (README, pyproject.toml, setup.py,
    LICENSE, etc.).  Top-level files are usually small and contain
    critical packaging metadata the agent needs to install dependencies
    correctly when it registers tools against the library.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "repo"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, url, str(tmp_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

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
            # Copy everything except .git
            shutil.copytree(tmp_path, dest, dirs_exist_ok=True)
            shutil.rmtree(dest / ".git", ignore_errors=True)


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
    embedder cache stays warm so the local sentence-transformers model
    isn't reloaded per collection.  When None (default for backwards
    compat with direct callers), a fresh KB is constructed and closed
    around this call.

    ``run_setup_kb`` always passes a shared *kb* so a multi-collection
    run pays the model-load cost once, not N times.
    """
    print(f"Setting up {name}...", flush=True)

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
                default_index=vector_db,
                embedder_kwargs=embedder_kwargs,
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


def add_setup_kb_args(parser):
    """Add setup-kb arguments to a parser or subparser.

    Called from both the standalone ``dsagt-setup-kb`` entry point and the
    ``dsagt setup-kb`` subcommand so the argument set is defined once.
    """
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Index directory (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--collection",
        choices=list(COLLECTIONS.keys()),
        help="Setup only this collection",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["api", "local"],
        default="local",
        help=(
            "Embedding backend (default: local — sentence-transformers, "
            "no API credentials needed).  Pass --embedding-backend api to "
            "build collections against a hosted embedding endpoint; that "
            "path requires --embedding-base-url and --embedding-api-key (or "
            "the corresponding env vars)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name (falls back to EMBEDDING_MODEL env var)",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="Embedding API base URL (falls back to OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="Embedding API key (falls back to LLM_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--vector-db",
        choices=["chroma", "faiss"],
        default="chroma",
        help="Vector database backend (default: chroma)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-ingest collections that already exist in the index directory "
        "(default: skip existing).",
    )
    parser.add_argument(
        "--no-skill-catalog",
        action="store_true",
        help="Skip cloning + indexing the default external skill catalog "
        "(the K-Dense scientific skills repo).",
    )


def run_setup_kb(args):
    """Run the core knowledge base setup.

    Accepts a parsed argparse.Namespace with the fields added by
    ``add_setup_kb_args``.  Called from both the ``dsagt setup-kb``
    subcommand and the standalone ``dsagt-setup-kb`` entry point.
    """
    # Show only warnings/errors during setup-kb — the per-collection
    # print() lines below are the user-visible progress story.  Surfacing
    # every "Found N files" / "Created M chunks" log line on top creates
    # the noisy play-by-play we deliberately stripped out.
    #
    # ``force=True`` overrides cli.py's earlier basicConfig (which sets
    # INFO + a "[dsagt]" format for the rest of the CLI).  Without
    # ``force``, the second basicConfig is a no-op because the root
    # logger already has handlers, and the INFO-level chatter survives.
    import logging as _logging

    _logging.basicConfig(
        level=_logging.WARNING,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    # Check git is available.
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("git is required but not found on PATH")

    # Resolve embedding config with env-var fallback so the user gets a
    # clear error up front rather than 5 minutes into the first ingest.
    embedder_kwargs: dict = {}
    if args.embedding_backend == "api":
        api_key = (
            args.embedding_api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = args.embedding_base_url or os.getenv("OPENAI_BASE_URL")
        model = args.embedding_model or os.getenv("EMBEDDING_MODEL")
        missing = [
            n
            for n, v in [("api key", api_key), ("base URL", base_url), ("model", model)]
            if not v
        ]
        if missing:
            raise ValueError(
                "API embedding backend requires "
                + ", ".join(missing)
                + ". Pass via --embedding-* flags or set LLM_API_KEY, "
                "OPENAI_BASE_URL, EMBEDDING_MODEL."
            )
        embedder_kwargs = {"api_key": api_key, "base_url": base_url, "model": model}
    elif args.embedding_model:
        embedder_kwargs = {"model": args.embedding_model}

    # Configure LiteLLM retries before any embedding work.  setup-kb is a
    # one-shot bootstrap tool — no project exists yet, no MLflow to trace
    # to, so we skip init_tracing entirely.  @traced decorators inside
    # KnowledgeBase see no backend and short-circuit cleanly.
    from dsagt.observability import configure_litellm_retries

    configure_litellm_retries()

    # One KnowledgeBase per setup-kb invocation.  The embedder cache
    # lives on the KB instance, so creating fresh KBs per collection
    # would reload the local sentence-transformers model every time
    # (~11s × N collections of pure waste).  Threaded through to
    # setup_collection (and used directly for the bundled tools+skills
    # block below) so the model loads once.
    args.index_dir.mkdir(parents=True, exist_ok=True)
    from dsagt.knowledge import KnowledgeBase
    from dsagt.registry import (
        TOOLS_COLLECTION,
        ToolRegistry,
        _parse_frontmatter,
    )

    shared_kb = KnowledgeBase(
        index_dir=args.index_dir,
        default_embedder=args.embedding_backend,
        default_index=args.vector_db,
        embedder_kwargs=embedder_kwargs or {},
    )
    try:
        # Bundled tools: each spec file is a single chunk with rich
        # metadata.  Wipe-and-rebuild every run — there's no version
        # sentinel, so the user controls when this happens.  Bundled
        # *skills* are no longer indexed: every supported agent natively
        # auto-discovers installed/bundled SKILL.md folders, so skill
        # search covers only the external *catalog* tier below.
        current_version = _current_dsagt_version()

        tool_paths = [
            p
            for p in sorted(ToolRegistry._PACKAGE_TOOLS_DIR.glob("*.md"))
            if _parse_frontmatter(p).get("name")
        ]

        coll_dir = args.index_dir / TOOLS_COLLECTION
        if coll_dir.exists():
            shutil.rmtree(coll_dir)

        if tool_paths:
            tool_specs = [_parse_frontmatter(p) for p in tool_paths]
            shared_kb.add_entries(
                texts=[p.read_text() for p in tool_paths],
                collection=TOOLS_COLLECTION,
                metadatas=[
                    {
                        "tool_name": s["name"],
                        "tags": ",".join(s.get("tags", [])),
                        "executable": s.get("executable", ""),
                        "has_dependencies": str(bool(s.get("dependencies"))),
                        "source": "bundled",
                        "dsagt_version": current_version,
                    }
                    for s in tool_specs
                ],
            )

        print("  bundled tools: indexed", flush=True)

        # External skill catalog: clone + index the default source(s) so
        # ``search_skills`` can browse installable skills out of the box.
        # Best-effort — a clone failure (offline, repo moved) warns and
        # continues rather than aborting the whole KB build.
        if not getattr(args, "no_skill_catalog", False):
            from dsagt.commands.skills_catalog import sync_source
            from dsagt.session import DEFAULTS

            for src in DEFAULTS["skills"]["sources"]:
                try:
                    stats = sync_source(src, kb=shared_kb, force=args.rebuild)
                    print(
                        f"  skill catalog {stats['slug']}: {stats['indexed']} indexed",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001 — best-effort, keep going
                    print(
                        f"  skill catalog {src.get('url', src)}: skipped ({e})",
                        flush=True,
                    )

        collections = (
            {args.collection: COLLECTIONS[args.collection]}
            if args.collection
            else COLLECTIONS
        )

        for name, config in collections.items():
            target_dir = args.index_dir / name
            if _collection_exists(target_dir):
                if not args.rebuild:
                    print(
                        f"  {name}: already indexed (use --rebuild to force)",
                        flush=True,
                    )
                    continue
                shutil.rmtree(target_dir)
            setup_collection(
                name,
                config,
                args.index_dir,
                embedder_kwargs=embedder_kwargs,
                embedding_backend=args.embedding_backend,
                vector_db=args.vector_db,
                kb=shared_kb,
            )
    finally:
        shared_kb.close()


def main():
    """Standalone entry point (``dsagt-setup-kb``)."""
    parser = argparse.ArgumentParser(
        description="Setup DSAGT core knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Embedding config is taken from CLI flags, then from these env vars:\n"
            "  LLM_API_KEY / OPENAI_API_KEY   API key for the embedding endpoint\n"
            "  OPENAI_BASE_URL                OpenAI-compatible base URL\n"
            "  EMBEDDING_MODEL                Embedding model name\n\n"
            "Full core KB embedding typically takes 15-30 minutes over an API.\n\n"
            "This command can also be run as: dsagt setup-kb"
        ),
    )
    add_setup_kb_args(parser)
    run_setup_kb(parser.parse_args())


if __name__ == "__main__":
    main()
