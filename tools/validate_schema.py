#!/usr/bin/env python
"""
Validate Schema - Check CSV against expected column types.

Usage:
    python validate_schema.py <input> <schema>
    
Schema JSON format:
    {
        "columns": {
            "name": {"type": "string", "required": true},
            "age": {"type": "integer", "min": 0, "max": 150}
        }
    }
"""

import argparse
import csv
import json
import sys


def validate_type(value, expected_type):
    """Check if value matches expected type."""
    if not value.strip():
        return True  # Empty handled by required check
    
    if expected_type == "string":
        return True
    elif expected_type == "integer":
        try:
            int(value)
            return True
        except ValueError:
            return False
    elif expected_type == "number":
        try:
            float(value)
            return True
        except ValueError:
            return False
    elif expected_type == "boolean":
        return value.lower() in ("true", "false", "1", "0", "yes", "no")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate CSV against schema")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("schema", help="Path to schema JSON file")
    args = parser.parse_args()
    
    try:
        # Load schema
        with open(args.schema, "r") as f:
            schema = json.load(f)
        
        # Load data
        with open(args.input, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        errors = []
        columns = schema.get("columns", {})
        
        # Check required columns exist
        if rows:
            data_columns = set(rows[0].keys())
            for col, spec in columns.items():
                if spec.get("required") and col not in data_columns:
                    errors.append({"type": "missing_column", "column": col})
        
        # Validate each row
        for i, row in enumerate(rows):
            for col, spec in columns.items():
                if col not in row:
                    continue
                
                value = row[col]
                
                # Required check
                if spec.get("required") and not value.strip():
                    errors.append({
                        "type": "required_missing",
                        "row": i + 1,
                        "column": col,
                    })
                    continue
                
                # Type check
                if not validate_type(value, spec.get("type", "string")):
                    errors.append({
                        "type": "type_error",
                        "row": i + 1,
                        "column": col,
                        "expected": spec.get("type"),
                        "value": value[:50],
                    })
                
                # Range check for numbers
                if value.strip() and spec.get("type") in ("integer", "number"):
                    try:
                        num = float(value)
                        if "min" in spec and num < spec["min"]:
                            errors.append({
                                "type": "range_error",
                                "row": i + 1,
                                "column": col,
                                "message": f"Value {num} below min {spec['min']}",
                            })
                        if "max" in spec and num > spec["max"]:
                            errors.append({
                                "type": "range_error",
                                "row": i + 1,
                                "column": col,
                                "message": f"Value {num} above max {spec['max']}",
                            })
                    except ValueError:
                        pass
        
        result = {
            "valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors[:20],  # Limit output
        }
        
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["valid"] else 1)
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
