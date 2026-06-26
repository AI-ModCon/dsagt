# hdf5.py — API Reference

General-purpose helpers for inspecting and repackaging HDF5 files using h5py.
All functions live in `hdf5.py`.  No M3D-C1-specific knowledge is assumed;
these utilities work with any HDF5 file.

```python
from hdf5 import list_h5_files, list_h5_variables, read_h5_dataset, read_h5_attrs, repackage_h5, json_to_h5
```

---

## `list_h5_files(directory, recursive=False)`

Return a sorted list of `.h5` file paths in a directory.

| Parameter   | Type | Description                                        |
|-------------|------|----------------------------------------------------|
| `directory` | path | Directory to search                                |
| `recursive` | bool | If `True`, descend into subdirectories (default `False`) |

Returns `list[str]`.  Returns `[]` if the directory does not exist.

```python
list_h5_files("m3dc1_data")
# ['m3dc1_data/C1.h5', 'm3dc1_data/equilibrium.h5',
#  'm3dc1_data/time_000.h5', 'm3dc1_data/time_001.h5']
```

---

## `list_h5_variables(file_path)`

Return a sorted list of all dataset paths inside an HDF5 file.

| Parameter   | Type | Description          |
|-------------|------|----------------------|
| `file_path` | path | Path to the HDF5 file |

Returns `list[str]` of internal HDF5 paths (e.g. `"group/subgroup/dataset"`).
Returns `[]` if the file cannot be opened or contains no datasets.

```python
list_h5_variables("m3dc1_data/C1.h5")
# ['equilibrium/fields/B', 'equilibrium/fields/psi', ..., 'scalars/E_K3', ...]
```

---

## `read_h5_dataset(file_path, dataset_path)`

Read a single dataset from an HDF5 file and return it as a NumPy array.

| Parameter      | Type | Description                                              |
|----------------|------|----------------------------------------------------------|
| `file_path`    | path | Path to the HDF5 file                                    |
| `dataset_path` | str  | Internal HDF5 path to the dataset, e.g. `"scalars/E_K3"` |

Returns `np.ndarray`.

Raises `KeyError` if `dataset_path` is not found; raises `OSError` if the
file cannot be opened.

```python
ke = read_h5_dataset("m3dc1_data/C1.h5", "scalars/E_K3")
# array of kinetic energy vs timestep
```

---

## `read_h5_attrs(file_path, group_path="")`

Read HDF5 attributes of a group or dataset as a plain Python dict.

| Parameter    | Type | Description                                                         |
|--------------|------|---------------------------------------------------------------------|
| `file_path`  | path | Path to the HDF5 file                                               |
| `group_path` | str  | Internal HDF5 path of the group/dataset. Default `""` reads root attributes. |

Returns `dict[str, Any]`.  NumPy scalar types are cast to Python `int` or
`float` for easy serialisation.

Raises `KeyError` if `group_path` is non-empty and not found; raises `OSError`
if the file cannot be opened.

```python
read_h5_attrs("m3dc1_data/C1.h5", "equilibrium")
# {'version': 45, 'nspace': 2, 'ntimestep': 0, 'time': 0.0}

read_h5_attrs("m3dc1_data/C1.h5")   # root attributes
# {'n0_norm': 2e20, 'b0_norm': 6.5, 'l0_norm': 0.57, ...}
```

---

## `repackage_h5(output_path, sources, variables=None, selection=None, overwrite=False)`

Create a new HDF5 file containing a subset of datasets copied from one or more
source files.

| Parameter     | Type       | Description                                                                 |
|---------------|------------|-----------------------------------------------------------------------------|
| `output_path` | path       | Destination HDF5 file to create                                             |
| `sources`     | list[path] | Source HDF5 files to read from                                              |
| `variables`   | list[str] \| None | Dataset paths to copy from every source. `None` copies all datasets. |
| `selection`   | dict \| None | Per-source override: maps source path → list of dataset paths. Takes precedence over `variables` for that source. |
| `overwrite`   | bool       | If `True`, overwrite an existing `output_path` (default `False`)            |

Returns `list[str]` of dataset paths written into the output file.

Raises `FileExistsError` if `output_path` exists and `overwrite=False`.

**Grouping behaviour** — when multiple sources are provided, datasets are
placed under a group named after the source file stem to avoid collisions:

| Scenario | Destination path |
|---|---|
| Single source | `dataset_path` (top-level, no grouping) |
| Multiple sources, same directory | `{stem}/{dataset_path}` |
| Multiple sources, different directories | `{dir}/{stem}/{dataset_path}` |
| Multiple sources, different grandparent directories | `{granddir}/{dir}/{stem}/{dataset_path}` |

Dataset attributes and compression settings are preserved from the source.

```python
# Copy two datasets from a single file
repackage_h5(
    "subset.h5",
    sources=["m3dc1_data/C1.h5"],
    variables=["scalars/E_K3", "scalars/time"],
)

# Take different variables from two different files
repackage_h5(
    "combined.h5",
    sources=["m3dc1_data/C1.h5", "m3dc1_data/equilibrium.h5"],
    selection={
        "m3dc1_data/C1.h5":          ["scalars/E_K3", "scalars/time"],
        "m3dc1_data/equilibrium.h5": ["fields/psi", "fields/B"],
    },
)
# Result groups: C1/scalars/E_K3, C1/scalars/time,
#                equilibrium/fields/psi, equilibrium/fields/B
```

---

## `json_to_h5(output_path, source, overwrite=False)`

Write arbitrarily nested JSON data to an HDF5 file.

The `source` argument is resolved in order:

1. `dict` or `list` — used directly as parsed data.
2. `str` or `Path` pointing to an existing file — read and parsed as JSON.
3. `str` beginning with `{` or `[` — parsed as an inline JSON string.

**Conversion rules** (applied recursively):

| JSON type | HDF5 result |
|---|---|
| `dict` | group; each key becomes a child name |
| uniform or rectangular list | dataset (converted to `np.ndarray`) |
| jagged, mixed, or list of dicts | numbered subgroups (`"0"`, `"1"`, …) |
| `int` / `float` | scalar dataset |
| `bool` | scalar `uint8` dataset |
| `str` | scalar string dataset |
| `None` | omitted silently |

A top-level `list` is wrapped in a group named `"data"`.

| Parameter     | Type | Description |
|---------------|------|-------------|
| `output_path` | path | Destination HDF5 file to create |
| `source`      | dict, list, str, or Path | JSON data — see above |
| `overwrite`   | bool | If `True`, overwrite an existing `output_path` (default `False`) |

Returns `list[str]` of dataset paths written.

Raises `FileExistsError` if `output_path` exists and `overwrite=False`.
Raises `ValueError` if `source` is a string that is neither a valid file nor valid JSON.

```python
# From a JSON file written by a pipeline tool
json_to_h5("spectrum.h5", "spectrum_output.json")

# From an in-memory dict
m, psi, spec = compute_poloidal_spectrum(...)
json_to_h5("spectrum.h5", {
    "m_modes":  m.tolist(),
    "psi_norm": psi.tolist(),
    "spectrum": spec.tolist(),
})
# Writes datasets: /m_modes, /psi_norm, /spectrum

# From an inline JSON string
json_to_h5("out.h5", '{"time": [0.0, 1.0], "ke": [1e-13, 9e-8]}')

# Nested dict — compute_standard_spectra result
spectra = compute_standard_spectra(...)
json_to_h5("spectra.h5", {
    k: {"m_modes": m.tolist(), "psi_norm": p.tolist(), "spectrum": s.tolist()}
    for k, (m, p, s) in spectra.items()
})
# Writes groups /p, /br, /bz, /bphi, each containing three datasets
```
