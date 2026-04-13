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
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

from dsagt.session import REGISTRY_DIR

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


def clone_github(url: str, dest: Path, branch: str = "main", include: list[str] | None = None):
    """Clone a GitHub repo, optionally keeping only specific directories.

    When *include* is set, the named subdirectories are copied AND any
    top-level files at the repo root (README, pyproject.toml, setup.py,
    LICENSE, etc.).  Top-level files are usually small and contain
    critical packaging metadata the agent needs to install dependencies
    correctly when it registers tools against the library.
    """
    print(f"  Cloning {url} (branch: {branch})...")

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
    print(f"  Downloading arXiv:{paper_id}...")
    
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
) -> dict:
    """Download sources and ingest a collection."""
    print(f"\n{'='*60}")
    print(f"Setting up: {name}")
    print(f"{'='*60}")

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

        # Ingest using KnowledgeBase
        print("  Indexing (this may take several minutes per collection)...")

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
                exclude_patterns=config.get("exclude_patterns") or DEFAULT_EXCLUDE_PATTERNS,
            )
            skipped = result.get("skipped_files", 0)
            skip_msg = f", skipped {skipped} unreadable" if skipped else ""
            print(
                f"  Done: {result['files']} files, {result['chunks']} chunks"
                f"{skip_msg}"
            )
            return result
        finally:
            kb.close()


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
        default="api",
        help="Embedding backend (default: api)",
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="Embedding model name (falls back to EMBEDDING_MODEL env var)",
    )
    parser.add_argument(
        "--embedding-base-url", default=None,
        help="Embedding API base URL (falls back to OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--embedding-api-key", default=None,
        help="Embedding API key (falls back to LLM_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--vector-db",
        choices=["chroma", "faiss"],
        default="chroma",
        help="Vector database backend (default: chroma)",
    )


def run_setup_kb(args):
    """Run the core knowledge base setup.

    Accepts a parsed argparse.Namespace with the fields added by
    ``add_setup_kb_args``.  Called from both the ``dsagt setup-kb``
    subcommand and the standalone ``dsagt-setup-kb`` entry point.
    """
    # Surface batch progress and rate-limit retry warnings.
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
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
        api_key = args.embedding_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = args.embedding_base_url or os.getenv("OPENAI_BASE_URL")
        model = args.embedding_model or os.getenv("EMBEDDING_MODEL")
        missing = [n for n, v in [("api key", api_key), ("base URL", base_url), ("model", model)] if not v]
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

    # Configure LiteLLM retries before any embedding work.
    from dsagt.observability import init_tracing, install_litellm_otel_callback
    init_tracing("dsagt-setup-kb")
    install_litellm_otel_callback()

    print("DSAGT Core Knowledge Base Setup")
    print(f"Index directory: {args.index_dir}")
    print(f"Embedding backend: {args.embedding_backend}")
    print(f"Vector DB: {args.vector_db}")
    print("Note: API-backed embedding of the full core KB typically takes 15-30 minutes.")

    collections = {args.collection: COLLECTIONS[args.collection]} if args.collection else COLLECTIONS

    results = {}
    for name, config in collections.items():
        results[name] = setup_collection(
            name, config, args.index_dir,
            embedder_kwargs=embedder_kwargs,
            embedding_backend=args.embedding_backend,
            vector_db=args.vector_db,
        )

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"  {name}: {result.get('chunks', 0)} chunks")

    print("\nDone!")


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
