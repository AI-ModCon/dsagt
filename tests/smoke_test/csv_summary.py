"""csv_summary — the quickstart's registerable, executable fixture code.

Stdlib-only (``csv`` + ``statistics``), so registering and running it via
dsagt-run needs no dependency install — the dependency-free replacement for
the csvkit demo.  Summarizes a CSV: column list, row count, per-column null
count, and min/max/mean for numeric columns.  On ``samples.csv`` the null
counts surface the empty ``status`` / ``timestamp`` cells, which is the fact
the quickstart's explicit-memory step stores and recalls.
"""

import argparse
import csv
import json
import statistics
import sys


def _as_number(value):
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        rows = list(reader)

    nulls = {c: 0 for c in columns}
    numeric = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            v = row.get(c)
            if v is None or v.strip() == "":
                nulls[c] += 1
            else:
                n = _as_number(v)
                if n is not None:
                    numeric[c].append(n)

    # A column is "numeric" only if every non-null value parsed as a number.
    numeric_stats = {}
    for c in columns:
        vals = numeric[c]
        non_null = len(rows) - nulls[c]
        if vals and len(vals) == non_null:
            numeric_stats[c] = {
                "min": min(vals),
                "max": max(vals),
                "mean": round(statistics.fmean(vals), 4),
            }

    return {
        "file": path,
        "columns": columns,
        "row_count": len(rows),
        "null_counts": nulls,
        "columns_with_nulls": [c for c in columns if nulls[c] > 0],
        "numeric_stats": numeric_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a CSV: columns, row count, null counts, numeric stats."
    )
    parser.add_argument("file", help="Path to the CSV file")
    args = parser.parse_args()

    try:
        result = summarize(args.file)
    except FileNotFoundError:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "CSV-404",
                    "error": f"no such file: {args.file}",
                }
            )
        )
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
