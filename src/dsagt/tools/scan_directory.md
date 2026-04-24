---
name: scan_directory
description: Scan a data directory and produce structured report with file counts,
  sizes, and directory tree
executable: dsagt-run --tool scan_directory -- python tools/scan_directory.py
parameters:
  directory:
    type: string
    required: true
    cli: positional
    description: Path to directory to scan
  max_depth:
    type: integer
    required: false
    default: 5
    cli: "--max-depth"
    description: Maximum directory depth to traverse
  top_n:
    type: integer
    required: false
    default: 20
    cli: "--top-n"
    description: Number of largest files to list
---

# scan_directory

Scan a data directory and produce a structured report with file counts, sizes, and directory tree. Use this as your first step when exploring a new dataset to understand its layout before deciding how to process it.

## Shell Command

```bash
python tools/scan_directory.py <directory> [--max_depth N] [--top_n N]
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `directory` | yes | — | Path to directory to scan |
| `max_depth` | no | 5 | Maximum directory depth to traverse |
| `top_n` | no | 20 | Number of largest files to list |

## Example

```bash
python tools/scan_directory.py /data/raw --max_depth 3 --top_n 10
```

## Output

Prints a JSON report containing:
- Total file count and cumulative size
- Directory tree up to `max_depth`
- Top N largest files with sizes
