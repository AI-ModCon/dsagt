#!/usr/bin/env python
"""
Scan Directory - Summarize file structure, sizes, and types.

Produces a structured report of a data directory: total size, file counts
by extension, largest files, and directory tree with sizes. Useful for
assessing a dataset before building processing pipelines.

Usage:
    python scan_directory.py <directory>
    python scan_directory.py <directory> --max-depth 3 --top-n 10
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def scan(directory: Path, max_depth: int) -> dict:
    """Walk directory and collect file metadata."""
    files = []
    errors = []

    for root, dirs, filenames in os.walk(directory):
        # Respect max_depth
        depth = Path(root).relative_to(directory).parts
        if len(depth) >= max_depth:
            dirs.clear()
            continue

        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for name in filenames:
            if name.startswith("."):
                continue
            filepath = Path(root) / name
            try:
                stat = filepath.stat()
                files.append({
                    "path": str(filepath.relative_to(directory)),
                    "size": stat.st_size,
                    "extension": filepath.suffix.lower() or "(none)",
                })
            except OSError as e:
                errors.append({"path": str(filepath), "error": str(e)})

    return {"files": files, "errors": errors}


def build_dir_tree(files: list[dict], directory: Path, max_depth: int) -> list[dict]:
    """Aggregate sizes by directory."""
    dir_sizes = defaultdict(lambda: {"size": 0, "file_count": 0})

    for f in files:
        parts = Path(f["path"]).parts
        # Add to each ancestor directory
        for i in range(min(len(parts), max_depth)):
            dir_path = str(Path(*parts[: i + 1])) if i < len(parts) - 1 else str(Path(*parts[:i]))
            if i < len(parts) - 1:
                key = str(Path(*parts[: i + 1]))
                dir_sizes[key]["size"] += f["size"]
                dir_sizes[key]["file_count"] += 1

    # Root level
    root_entry = {"size": sum(f["size"] for f in files), "file_count": len(files)}

    tree = [{"directory": ".", **root_entry}]
    for dir_path in sorted(dir_sizes):
        tree.append({"directory": dir_path, **dir_sizes[dir_path]})

    return tree


def main():
    parser = argparse.ArgumentParser(description="Scan directory structure and file sizes")
    parser.add_argument("directory", help="Path to directory to scan")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum directory depth to traverse (default: 5)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of largest files to list (default: 20)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.exists():
        print(json.dumps({"error": f"Directory not found: {directory}"}), file=sys.stderr)
        sys.exit(1)

    if not directory.is_dir():
        print(json.dumps({"error": f"Not a directory: {directory}"}), file=sys.stderr)
        sys.exit(1)

    try:
        result = scan(directory, args.max_depth)
        files = result["files"]

        # Aggregate by extension
        ext_summary = defaultdict(lambda: {"count": 0, "total_size": 0})
        for f in files:
            ext = ext_summary[f["extension"]]
            ext["count"] += 1
            ext["total_size"] += f["size"]

        ext_report = [
            {"extension": k, "count": v["count"], "total_size": v["total_size"],
             "total_size_human": format_size(v["total_size"])}
            for k, v in sorted(ext_summary.items(), key=lambda x: x[1]["total_size"], reverse=True)
        ]

        # Largest files
        largest = sorted(files, key=lambda f: f["size"], reverse=True)[: args.top_n]
        for f in largest:
            f["size_human"] = format_size(f["size"])

        # Directory tree with sizes
        dir_tree = build_dir_tree(files, directory, args.max_depth)
        for d in dir_tree:
            d["size_human"] = format_size(d["size"])

        total_size = sum(f["size"] for f in files)

        report = {
            "directory": str(directory.resolve()),
            "total_files": len(files),
            "total_size": total_size,
            "total_size_human": format_size(total_size),
            "file_types": ext_report,
            "largest_files": largest,
            "directory_tree": dir_tree,
        }

        if result["errors"]:
            report["scan_errors"] = result["errors"]

        print(json.dumps(report, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
