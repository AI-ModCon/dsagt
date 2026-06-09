# Introspection Commands

## Contents
- File structure and size
- Schema and data sampling (CSV, Parquet, JSON, HDF5/NetCDF, images)
- Existing metadata files
- Directory structure and splits
- Edge case handling

## File Structure and Size

```bash
# Full recursive file listing
find <dataset_dir> -type f | sort

# Count files by extension
find <dataset_dir> -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn

# Total size (human-readable)
du -sh <dataset_dir>

# Size of top-level items
du -sh <dataset_dir>/*
```

## Schema and Data Sampling

```bash
# CSV/TSV — headers, row count, dtypes
head -n 3 <file.csv>
wc -l <file.csv>
python3 -c "import pandas as pd; df = pd.read_csv('<file.csv>', nrows=5); print(df.dtypes); print(df.shape)"

# Parquet
python3 -c "import pandas as pd; df = pd.read_parquet('<file.parquet>'); print(df.dtypes); print(df.shape)"

# JSON — top-level structure
python3 -c "import json; d=json.load(open('<file.json>')); print(type(d), list(d.keys()) if isinstance(d,dict) else f'{len(d)} items')"

# HDF5
python3 -c "import h5py; f=h5py.File('<file.h5>','r'); print(list(f.keys())); f.visititems(lambda n,o: print(n, type(o).__name__))"

# NetCDF
python3 -c "import netCDF4 as nc; d=nc.Dataset('<file.nc>'); print(d.variables.keys()); print(d.dimensions)"

# Images — count by format
find <dataset_dir> -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.tif" \) | wc -l
```

For files larger than 1GB, sample only (e.g., `nrows=1000`) and note this in the review summary.

## Existing Metadata Files

```bash
# Check for common metadata files
for f in README.md README.txt LICENSE CITATION.cff CITATION.bib .zenodo.json datacite.json croissant.json metadata.json; do
  [ -f "<dataset_dir>/$f" ] && echo "FOUND: $f"
done

# Print contents of found files
cat <dataset_dir>/README.md 2>/dev/null
cat <dataset_dir>/CITATION.cff 2>/dev/null
cat <dataset_dir>/LICENSE 2>/dev/null
```

## Directory Structure and Splits

```bash
# Top-level listing and directory tree (2 levels)
ls <dataset_dir>/
find <dataset_dir> -maxdepth 2 -type d | sort
```

Look for `train/`, `test/`, `validation/`, or file naming patterns like `_train.csv`.

## Edge Case Handling

- **Empty directory**: Inform the user; ask whether to proceed with a skeleton card
- **Binary-only files**: Note under "Files & Structure" and mark schema fields N/A
- **Large files (>1GB)**: Sample only; note in the review summary