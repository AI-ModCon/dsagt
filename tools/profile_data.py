#!/usr/bin/env python
"""
Profile Data - Generate summary statistics for a CSV file.

Usage:
    python profile_data.py <input> [--output FILE]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict


def infer_type(values):
    """Infer column type from sample values."""
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "empty"
    
    # Try int
    try:
        [int(v) for v in non_empty[:100]]
        return "integer"
    except ValueError:
        pass
    
    # Try float
    try:
        [float(v) for v in non_empty[:100]]
        return "number"
    except ValueError:
        pass
    
    return "string"


def main():
    parser = argparse.ArgumentParser(description="Profile CSV data")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("--output", default="profile.json", help="Output JSON file")
    args = parser.parse_args()
    
    try:
        with open(args.input, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            print(json.dumps({"error": "Empty file"}))
            sys.exit(1)
        
        columns = list(rows[0].keys())
        
        profile = {
            "file": args.input,
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": {},
        }
        
        for col in columns:
            values = [row[col] for row in rows]
            non_null = [v for v in values if v.strip()]
            
            col_profile = {
                "type": infer_type(values),
                "null_count": len(values) - len(non_null),
                "unique_count": len(set(non_null)),
            }
            
            # Numeric stats
            if col_profile["type"] in ("integer", "number"):
                nums = [float(v) for v in non_null]
                if nums:
                    col_profile["min"] = min(nums)
                    col_profile["max"] = max(nums)
                    col_profile["mean"] = sum(nums) / len(nums)
            
            profile["columns"][col] = col_profile
        
        # Write output
        with open(args.output, "w") as f:
            json.dump(profile, f, indent=2)
        
        print(json.dumps({"status": "ok", "output": args.output, "rows": len(rows)}))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
