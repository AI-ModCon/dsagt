#!/usr/bin/env python
"""
DSAGT Demo - Direct MCP Server Test

Tests the MCP server functionality without Goose.
Simulates what an agent would do.

Usage:
    python demo_test.py
"""

import json
import os
import sys
from pathlib import Path

from dsagt import ToolRegistry


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_result(result):
    if result["success"]:
        print(f"✓ Success")
        if result.get("output"):
            # Pretty print JSON output
            try:
                parsed = json.loads(result["output"])
                print(json.dumps(parsed, indent=2))
            except:
                print(result["output"][:500])
    else:
        print(f"✗ Failed: {result.get('error')}")


def main():
    demo_dir = Path(__file__).parent
    data_file = demo_dir / "building_sensors.csv"
    custom_script = demo_dir / "fill_missing.py"
    output_dir = demo_dir / "output"
    
    print_header("DSAGT Demo - Data Pipeline Test")
    
    print(f"Data file: {data_file}")
    print(f"Custom script: {custom_script}")
    print(f"Output directory: {output_dir}")
    
    # Initialize registry with runtime in demo folder
    runtime_dir = demo_dir / "runtime"
    registry_file = demo_dir.parent / "registry.yaml"
    registry = ToolRegistry(
        source_registry=str(registry_file),
        runtime_dir=str(runtime_dir)
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: List available tools
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Step 1: List Available Tools")
    
    tools = registry.list_tools()
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Load and inspect data
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Step 2: Load Data")
    
    result = registry.call_tool("load_csv", {"location": str(data_file)})
    print_result(result)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Profile data
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Step 3: Profile Data")
    
    profile_output = demo_dir / "output" / "profile.json"
    os.makedirs(demo_dir / "output", exist_ok=True)
    
    result = registry.call_tool("profile_data", {
        "input": str(data_file),
        "output": str(profile_output),
    })
    print_result(result)
    
    # Show profile summary
    if profile_output.exists():
        with open(profile_output) as f:
            profile = json.load(f)
        print(f"\nProfile summary:")
        print(f"  Rows: {profile['row_count']}")
        print(f"  Columns: {profile['column_count']}")
        for col, info in profile["columns"].items():
            null_info = f" ({info['null_count']} nulls)" if info['null_count'] > 0 else ""
            print(f"    • {col}: {info['type']}{null_info}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Register custom preprocessing script
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Step 4: Register Custom Script")
    
    result = registry.register_tool(
        name="fill_missing",
        description="Fill missing occupancy values with 0",
        executable=f"python {custom_script}",
        parameters={
            "input": {"type": "string", "required": True, "description": "Input CSV"},
            "output": {"type": "string", "required": True, "description": "Output CSV"},
        },
    )
    
    if result["success"]:
        print(f"✓ Registered 'fill_missing' tool")
        print(f"  Executable: python {custom_script}")
    else:
        print(f"✗ Failed: {result.get('error')}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Run custom preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Step 5: Run Custom Preprocessing")
    
    cleaned_file = demo_dir / "output" / "cleaned.csv"
    
    result = registry.call_tool("fill_missing", {
        "input": str(data_file),
        "output": str(cleaned_file),
    })
    print_result(result)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Split data
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Step 6: Split Data for ML")
    
    splits_dir = demo_dir / "output" / "splits"
    
    result = registry.call_tool("split_data", {
        "input": str(cleaned_file),
        "output_dir": str(splits_dir),
        "train_ratio": 0.7,
        "seed": 42,
    })
    print_result(result)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Show provenance
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Provenance Log")
    
    provenance_file = runtime_dir / "provenance.log"
    if provenance_file.exists():
        with open(provenance_file) as f:
            for line in f:
                if not line.startswith("#"):
                    print(f"  {line.rstrip()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Demo Complete")
    
    print("Generated files:")
    for f in (demo_dir / "output").rglob("*"):
        if f.is_file():
            print(f"  • {f.relative_to(demo_dir)}")
    
    print(f"\nRuntime registry: {runtime_dir / 'registry.yaml'}")
    print(f"Provenance log: {provenance_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
