# csvtool CLI Reference

## csvtool filter

Filter rows by column value.

```bash
csvtool filter --input data.csv --column status --value active --output filtered.csv
```

Parameters:
- `--input` (required) — Input CSV file path
- `--column` (required) — Column name to filter on
- `--value` (required) — Value to match
- `--output` (required) — Output CSV file path

## csvtool validate

Check a CSV against a schema.

```bash
csvtool validate --input data.csv --schema schema.json
```

Outputs a JSON report with row count, null counts per column, type violations, and an overall pass/fail status.

## csvtool summary

Print column statistics.

```bash
csvtool summary --input data.csv
```

Outputs: column names, types, null counts, unique value counts, and min/max for numeric columns.
