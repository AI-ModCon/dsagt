# API Reference

## DataLoader class

The DataLoader class reads CSV and Parquet files into memory.
It supports lazy loading for files larger than available RAM.

### Methods

- `load(path)` — Load a file and return a DataFrame
- `validate(schema)` — Check data against a JSON schema
- `summary()` — Print column types and null counts
