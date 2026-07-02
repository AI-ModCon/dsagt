#!/usr/bin/env python
"""
Scan Directory (macOS + Linux, core-tools implementation)

Sensible defaults:
- ALWAYS include hidden files/dirs (dotfiles)
- Use core shell tools to enumerate files/sizes (find + stat) and total size (du)
- Return structured JSON for MCP agents

Keeps the existing tool parameters: directory, max_depth, top_n.
"""

import argparse
import json
import subprocess
import sys
import heapq
from collections import defaultdict
from pathlib import Path


def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def is_darwin() -> bool:
    return sys.platform == "darwin"


def run_checked(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def du_total_bytes(directory: Path) -> int:
    """
    Total directory size in bytes using du.
    - Linux: du -sb (GNU)
    - macOS: du -sk (BSD) converted to bytes
    """
    if not is_darwin():
        p = run_checked(["du", "-sb", str(directory)])
        return int(p.stdout.split()[0])

    p = run_checked(["du", "-sk", str(directory)])
    kb = int(p.stdout.split()[0])
    return kb * 1024


def iter_files_sizes_relative(
    directory: Path, max_depth: int
) -> tuple[list[tuple[int, str]], list[dict]]:
    """
    Enumerate files under `directory` up to `max_depth`, INCLUDING hidden.
    Uses: find + stat (platform-specific flags) via -exec ... {} +

    Returns:
      - list of (size_bytes, relpath)
      - list of warnings/errors (non-fatal unless stat/find totally fails)
    """
    errors: list[dict] = []

    # Find files up to max_depth from ".", so relpaths are natural.
    find_cmd = ["find", ".", "-maxdepth", str(max_depth), "-type", "f"]

    # stat formatting:
    #  - Linux: stat -c '%s<TAB>%n'
    #  - macOS: stat -f '%z<TAB>%N'
    stat_cmd = ["stat", "-f", "%z\t%N"] if is_darwin() else ["stat", "-c", "%s\t%n"]

    cmd = find_cmd + ["-exec"] + stat_cmd + ["{}", "+"]

    p = subprocess.run(
        cmd,
        cwd=str(directory),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,  # collect partial results even if some files error
    )

    if p.stderr.strip():
        errors.append({"path": str(directory), "error": p.stderr.strip()[:4000]})

    items: list[tuple[int, str]] = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            size_str, name = line.split("\t", 1)
            size = int(size_str)
            rel = name[2:] if name.startswith("./") else name
            if not rel or rel == ".":
                continue
            items.append((size, rel))
        except Exception:
            errors.append(
                {
                    "path": str(directory),
                    "error": f"Unparseable stat line: {line[:200]}",
                }
            )

    if p.returncode != 0:
        errors.append(
            {"path": str(directory), "error": f"find/stat exited {p.returncode}"}
        )

    return items, errors


def scan(directory: Path, max_depth: int, top_n: int) -> dict:
    ext_summary = defaultdict(lambda: {"count": 0, "total_size": 0})
    dir_summary = defaultdict(lambda: {"size": 0, "file_count": 0})
    scan_errors: list[dict] = []

    total_size_full = du_total_bytes(directory)

    file_items, errs = iter_files_sizes_relative(directory, max_depth=max_depth)
    scan_errors.extend(errs)

    total_files = 0
    total_size = 0

    # Heap entries are (size, tiebreaker, entry). The integer tiebreaker keeps
    # equal-size files orderable so heapq never falls through to comparing dicts.
    largest_heap: list[tuple[int, int, dict]] = []
    tiebreaker = 0

    for size, rel in file_items:
        total_files += 1
        total_size += size

        rel_path = Path(rel)
        ext = rel_path.suffix.lower() or "(none)"
        ext_summary[ext]["count"] += 1
        ext_summary[ext]["total_size"] += size

        # aggregate directory ancestors
        parts = rel_path.parts
        for i in range(len(parts) - 1):
            d = str(Path(*parts[: i + 1]))
            dir_summary[d]["size"] += size
            dir_summary[d]["file_count"] += 1

        entry = {"path": rel, "size": size, "size_human": format_size(size)}
        tiebreaker += 1
        if len(largest_heap) < top_n:
            heapq.heappush(largest_heap, (size, tiebreaker, entry))
        else:
            heapq.heappushpop(largest_heap, (size, tiebreaker, entry))

    file_types = [
        {
            "extension": k,
            "count": v["count"],
            "total_size": v["total_size"],
            "total_size_human": format_size(v["total_size"]),
        }
        for k, v in sorted(
            ext_summary.items(), key=lambda x: x[1]["total_size"], reverse=True
        )
    ]

    largest_files = [
        e for _, _, e in sorted(largest_heap, key=lambda t: (t[0], t[1]), reverse=True)
    ]

    directory_tree = [
        {
            "directory": ".",
            "size": total_size,
            "file_count": total_files,
            "size_human": format_size(total_size),
        }
    ]
    for d in sorted(dir_summary):
        directory_tree.append(
            {
                "directory": d,
                "size": dir_summary[d]["size"],
                "file_count": dir_summary[d]["file_count"],
                "size_human": format_size(dir_summary[d]["size"]),
            }
        )

    report = {
        "directory": str(directory.resolve()),
        "total_files": total_files,
        "total_size": total_size,
        "total_size_human": format_size(total_size),
        # extra context (non-breaking addition)
        "total_size_full": total_size_full,
        "total_size_full_human": format_size(total_size_full),
        "file_types": file_types,
        "largest_files": largest_files,
        "directory_tree": directory_tree,
    }
    if scan_errors:
        report["scan_errors"] = scan_errors
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Scan directory structure and file sizes"
    )
    parser.add_argument("directory", help="Path to directory to scan")
    # accept both hyphen and underscore variants (registry runners vary)
    parser.add_argument("--max-depth", "--max_depth", type=int, default=5)
    parser.add_argument("--top-n", "--top_n", type=int, default=20)
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(
            json.dumps({"error": f"Directory not found: {directory}"}), file=sys.stderr
        )
        sys.exit(1)
    if not directory.is_dir():
        print(json.dumps({"error": f"Not a directory: {directory}"}), file=sys.stderr)
        sys.exit(1)

    try:
        report = scan(directory, args.max_depth, args.top_n)
        print(json.dumps(report, indent=2))
    except subprocess.CalledProcessError as e:
        print(json.dumps({"error": e.stderr.strip() or str(e)}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
