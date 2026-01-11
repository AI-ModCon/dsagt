#!/usr/bin/env python
"""
Split Data - Split CSV into train/dev/test sets.

Usage:
    python split_data.py <input> <output_dir> [--train_ratio N] [--seed N]
"""

import argparse
import csv
import json
import os
import random
import sys


def main():
    parser = argparse.ArgumentParser(description="Split data into train/dev/test")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    try:
        # Load data
        with open(args.input, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        
        # Shuffle
        random.seed(args.seed)
        random.shuffle(rows)
        
        # Calculate split sizes
        n = len(rows)
        train_end = int(n * args.train_ratio)
        dev_end = train_end + int(n * (1 - args.train_ratio) / 2)
        
        splits = {
            "train": rows[:train_end],
            "dev": rows[train_end:dev_end],
            "test": rows[dev_end:],
        }
        
        # Write splits
        os.makedirs(args.output_dir, exist_ok=True)
        
        for name, split_rows in splits.items():
            path = os.path.join(args.output_dir, f"{name}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(split_rows)
        
        result = {
            "status": "ok",
            "output_dir": args.output_dir,
            "splits": {name: len(rows) for name, rows in splits.items()},
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
