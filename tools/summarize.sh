#!/bin/bash
# Example shell script tool - summarize CSV
# Usage: ./summarize.sh <file>

FILE="$1"

if [ -z "$FILE" ]; then
    echo '{"error": "No file specified"}' >&2
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "{\"error\": \"File not found: $FILE\"}" >&2
    exit 1
fi

LINES=$(wc -l < "$FILE")
COLS=$(head -1 "$FILE" | tr ',' '\n' | wc -l)
SIZE=$(du -h "$FILE" | cut -f1)

echo "{"
echo "  \"file\": \"$FILE\","
echo "  \"lines\": $LINES,"
echo "  \"columns\": $COLS,"
echo "  \"size\": \"$SIZE\""
echo "}"
