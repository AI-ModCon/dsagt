# csvtool Troubleshooting

## Encoding errors

If csvtool raises a UnicodeDecodeError, the file may not be UTF-8. Use `--encoding latin-1` or detect encoding with `chardet`.

## Large file performance

For files over 1 GB, csvtool streams rows in chunks. If you still run out of memory, increase the chunk size with `--chunk-size 100000` or filter to specific columns with `--columns col1,col2`.

## Schema validation failures

The validate command checks column types against a JSON schema. Common issues:
- Mixed types in a column (e.g., "123" and "abc" in an integer column)
- Missing required columns
- Null values in non-nullable columns

Fix the data or update the schema to match reality.
