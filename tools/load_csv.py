#!/usr/bin/env python
"""
Load CSV - Output basic info about a CSV file.

Usage:
    python load_csv.py <location> [--delimiter DELIM]
"""

import argparse
import csv
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Load CSV and output info")
    parser.add_argument("location", help="Path to CSV file")
    parser.add_argument("--delimiter", default=",", help="Field delimiter")
    args = parser.parse_args()
    
    try:
        with open(args.location, "r") as f:
            reader = csv.reader(f, delimiter=args.delimiter)
            header = next(reader)
            rows = list(reader)
        
        result = {
            "file": args.location,
            "columns": header,
            "column_count": len(header),
            "row_count": len(rows),
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
