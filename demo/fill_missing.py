#!/usr/bin/env python
"""
Custom preprocessing script - fills missing occupancy values.

This represents a user's bespoke script that they want to integrate
into the pipeline.

Usage:
    python fill_missing.py <input> <output>
"""

import argparse
import csv
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Fill missing occupancy values")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()
    
    try:
        # Read input
        with open(args.input, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Fill missing occupancy with 0
        filled_count = 0
        for row in rows:
            if row.get("occupancy", "").strip() == "":
                row["occupancy"] = "0"
                filled_count += 1
        
        # Write output
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        result = {
            "status": "ok",
            "input": args.input,
            "output": args.output,
            "rows_processed": len(rows),
            "values_filled": filled_count,
        }
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
