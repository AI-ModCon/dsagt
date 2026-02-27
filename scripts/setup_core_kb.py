#!/usr/bin/env python
"""
Setup core knowledge base collections for DSAGT.

Downloads and indexes:
- nemo_curator: NVIDIA NeMo Curator (code, docs, tutorials)
- aidrin: AI Data Readiness Inspector (code, papers)

Usage:
    python scripts/setup_core_kb.py
    python scripts/setup_core_kb.py --index-dir ./my_kb_index
    python scripts/setup_core_kb.py --collection nemo_curator  # Single collection
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

# Default index directory (relative to script location)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_INDEX_DIR = SCRIPT_DIR.parent / "kb_index"

# Collection definitions
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
    """Clone a GitHub repo, optionally keeping only specific directories."""
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
            # Copy only specified directories
            for subdir in include:
                src = tmp_path / subdir
                if src.exists():
                    shutil.copytree(src, dest / subdir, dirs_exist_ok=True)
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


def setup_collection(name: str, config: dict, index_dir: Path) -> dict:
    """Download sources and ingest a collection."""
    print(f"\n{'='*60}")
    print(f"Setting up: {name}")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as tmp:
        download_dir = Path(tmp) / name
        download_dir.mkdir()
        
        # Download all sources
        for source in config["sources"]:
            try:
                if source["type"] == "github":
                    clone_github(
                        source["url"],
                        download_dir,
                        branch=source.get("branch", "main"),
                        include=source.get("include"),
                    )
                elif source["type"] == "arxiv":
                    download_arxiv(source["id"], download_dir)
            except Exception as e:
                print(f"  Warning: {source} failed: {e}")
        
        # Write DESCRIPTION.md
        (download_dir / "DESCRIPTION.md").write_text(config["description"])
        
        # Ingest using KnowledgeBase
        print("  Indexing...")
        
        # Import here to allow script to show help without dependencies
        sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
        from dsagt.knowledge import KnowledgeBase
        
        kb = KnowledgeBase(index_dir=index_dir)
        try:
            result = kb.ingest(download_dir)
            print(f"  Done: {result['files']} files, {result['chunks']} chunks")
            return result
        finally:
            kb.close()


def main():
    parser = argparse.ArgumentParser(description="Setup DSAGT core knowledge base")
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
    args = parser.parse_args()
    
    # Check git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: git is required")
        sys.exit(1)
    
    print("DSAGT Core Knowledge Base Setup")
    print(f"Index directory: {args.index_dir}")
    
    # Setup collections
    collections = {args.collection: COLLECTIONS[args.collection]} if args.collection else COLLECTIONS
    
    results = {}
    for name, config in collections.items():
        try:
            results[name] = setup_collection(name, config, args.index_dir)
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: FAILED - {result['error']}")
        else:
            print(f"  {name}: {result.get('chunks', 0)} chunks")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
