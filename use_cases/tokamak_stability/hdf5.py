"""Small helpers for working with HDF5 files using h5py.

Functions:
- ``list_h5_files(directory, recursive=False)``: return list of .h5 files in directory.
- ``list_h5_variables(file_path)``: return list of dataset paths inside an HDF5 file.
- ``read_h5_dataset(file_path, dataset_path)``: read one dataset as a NumPy array.
- ``read_h5_attrs(file_path, group_path="")``: read HDF5 attributes as a plain dict.
- ``repackage_h5(output_path, sources, ...)``: copy a subset of datasets to a new file.
- ``json_to_h5(output_path, source, ...)``: write JSON data to an HDF5 file.

Example:
	>>> list_h5_files('data')
	['data/run1.h5', 'data/run2.h5']

	>>> list_h5_variables('data/run1.h5')
	['group1/dset1', 'group2/sub/dset2']

	>>> read_h5_dataset('data/run1.h5', 'scalars/time')
	array([0., 1., 2., ...])

	>>> read_h5_attrs('data/run1.h5', 'equilibrium')
	{'version': 45, 'nspace': 2, 'ntimestep': 0, 'time': 0.0}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


def list_h5_files(directory: str | Path, recursive: bool = False) -> List[str]:
	"""Return a sorted list of `.h5` files in ``directory``.

	Args:
		directory: path to the directory to search.
		recursive: if True, search subdirectories recursively.

	Returns:
		Sorted list of file paths as strings.
	"""
	p = Path(directory)
	if not p.exists():
		return []
	if recursive:
		files = [str(x) for x in p.rglob('*.h5') if x.is_file()]
	else:
		files = [str(x) for x in p.glob('*.h5') if x.is_file()]
	return sorted(files)


def list_h5_variables(file_path: str | Path) -> List[str]:
	"""Return a sorted list of dataset paths (variables) in an HDF5 file.

	The returned names are the internal HDF5 paths (e.g. "group/dset").

	Args:
		file_path: path to the HDF5 file.

	Returns:
		Sorted list of dataset path strings. Returns empty list if file
		cannot be opened or contains no datasets.
	"""
	p = Path(file_path)
	if not p.exists():
		return []

	datasets: List[str] = []
	try:
		with h5py.File(p, 'r') as f:
			def visitor(name, obj):
				if isinstance(obj, h5py.Dataset):
					datasets.append(name)

			f.visititems(visitor)
	except (OSError, IOError):
		return []

	return sorted(datasets)


__all__ = [
	'list_h5_files',
	'list_h5_variables',
	'read_h5_dataset',
	'read_h5_attrs',
	'repackage_h5',
	'json_to_h5',
]


def read_h5_dataset(file_path: str | Path, dataset_path: str) -> np.ndarray:
	"""Read one dataset from an HDF5 file and return it as a NumPy array.

	Args:
		file_path:    Path to the HDF5 file.
		dataset_path: Internal HDF5 path to the dataset (e.g. ``"scalars/E_K3"``).

	Returns:
		NumPy array containing the dataset values.

	Raises:
		KeyError: If ``dataset_path`` is not found in the file.
		OSError:  If the file cannot be opened.
	"""
	with h5py.File(Path(file_path), 'r') as f:
		if dataset_path not in f:
			raise KeyError(f"Dataset '{dataset_path}' not found in {file_path}")
		return f[dataset_path][()]


def read_h5_attrs(file_path: str | Path, group_path: str = "") -> Dict[str, Any]:
	"""Read HDF5 attributes of a group or dataset as a plain Python dict.

	Args:
		file_path:  Path to the HDF5 file.
		group_path: Internal HDF5 path of the group or dataset whose attributes
		            are to be read. Empty string (default) reads root-level attributes.

	Returns:
		Dict mapping attribute name to value. NumPy scalar types are cast to
		Python ``int`` or ``float`` for easy serialisation.

	Raises:
		KeyError: If ``group_path`` is not empty and is not found in the file.
		OSError:  If the file cannot be opened.
	"""
	result: Dict[str, Any] = {}
	with h5py.File(Path(file_path), 'r') as f:
		obj = f[group_path] if group_path else f
		for key, val in obj.attrs.items():
			if hasattr(val, 'item'):
				result[key] = val.item()
			elif isinstance(val, np.ndarray):
				result[key] = val.tolist()
			else:
				result[key] = val
	return result


def repackage_h5(
	output_path: str | Path,
	sources: List[str | Path],
	variables: List[str] | None = None,
	selection: dict | None = None,
	overwrite: bool = False,
) -> List[str]:
	"""Create a new HDF5 file containing a subset of datasets from sources.

	Args:
		output_path: destination HDF5 file to create.
		sources: list of source HDF5 file paths to read from.
		variables: optional list of dataset paths to copy from any source.
			If omitted, all datasets from each source are copied.
		selection: optional mapping of source-path -> list of dataset paths
			to copy from that specific source. Keys should be strings or
			Path-like objects. If provided, it takes precedence for that
			source. Use this when you need different variables from different
			files — something ``variables`` cannot express. For example, to
			take ``temperature`` from ``run1.h5`` and ``pressure`` from
			``run2.h5``:

				repackage_h5(
					"out.h5",
					sources=["run1.h5", "run2.h5"],
					selection={
						"run1.h5": ["temperature"],
						"run2.h5": ["pressure"],
					},
				)
		overwrite: if True, overwrite existing `output_path`.

	Returns:
		List of dataset paths written into the output file.

	Notes:
		- If multiple source files are being repackaged,
		  all variables are stored under a group named after the source file stem
		  (e.g. "file1/var1", "file2/var1").
		- Large datasets are read into memory when copying.
	"""
	outp = Path(output_path)
	if outp.exists() and not overwrite:
		raise FileExistsError(f"Output file exists: {outp}")
	outp.parent.mkdir(parents=True, exist_ok=True)

	written: List[str] = []

	# Normalize selection keys to strings for easy lookup
	norm_sel = {}
	if selection:
		for k, v in selection.items():
			norm_sel[str(Path(k))] = list(v)

	# First pass: resolve which variables will be copied from each source
	source_vars: list[tuple[Path, list[str]]] = []
	for src in sources:
		srcp = Path(src)
		if not srcp.exists():
			continue
		if str(srcp) in norm_sel:
			vars_to_copy = norm_sel[str(srcp)]
		elif variables is not None:
			vars_to_copy = list(variables)
		else:
			vars_to_copy = list_h5_variables(srcp)
		source_vars.append((srcp, vars_to_copy))

	# Determine grouping depth based on where the source files live.
	# - Single source: no grouping, write at top level.
	# - Multiple sources, same directory: group by filename stem only.
	# - Multiple sources, different directories, same grandparent: group by directory + stem.
	# - Multiple sources, different grandparent directories: group by grandparent + directory + stem.
	use_groups = len(source_vars) > 1
	use_dir_prefix = use_groups and len({srcp.parent.resolve() for srcp, _ in source_vars}) > 1
	use_granddir_prefix = use_dir_prefix and len({srcp.parent.parent.resolve() for srcp, _ in source_vars}) > 1

	# Second pass: copy datasets
	for srcp, vars_to_copy in source_vars:
		try:
			with h5py.File(srcp, 'r') as fin:
				with h5py.File(outp, 'a') as fout:
					for v in vars_to_copy:
						if v not in fin:
							continue
						src_ds = fin[v]

						if use_granddir_prefix:
							dest_path = f"{srcp.parent.parent.name}/{srcp.parent.name}/{srcp.stem}/{v.lstrip('/')}"
						elif use_dir_prefix:
							dest_path = f"{srcp.parent.name}/{srcp.stem}/{v.lstrip('/')}"
						elif use_groups:
							dest_path = f"{srcp.stem}/{v.lstrip('/')}"
						else:
							dest_path = v

						parts = dest_path.strip('/').split('/')
						grp = fout
						for part in parts[:-1]:
							grp = grp.require_group(part)

						# Prepare kwargs to preserve storage hints where possible
						ds_kwargs = {}
						if getattr(src_ds, 'compression', None) is not None:
							ds_kwargs['compression'] = src_ds.compression
						if getattr(src_ds, 'compression_opts', None) is not None:
							ds_kwargs['compression_opts'] = src_ds.compression_opts
						if getattr(src_ds, 'chunks', None) is not None:
							ds_kwargs['chunks'] = src_ds.chunks

						# Create dataset (reads into memory)
						data = src_ds[()]
						name = parts[-1]
						if name in grp:
							del grp[name]
						new_ds = grp.create_dataset(name, data=data, **ds_kwargs)

						# Copy attributes
						for k, val in src_ds.attrs.items():
							new_ds.attrs[k] = val

						written.append('/' + dest_path.strip('/'))
		except (OSError, IOError):
			continue

	return written


# ---------------------------------------------------------------------------
# json_to_h5 helpers
# ---------------------------------------------------------------------------

def _try_numeric_array(lst: list) -> "np.ndarray | None":
    """Return a numeric ndarray if lst is a uniform or rectangular numeric list, else None."""
    try:
        arr = np.array(lst)
    except (ValueError, TypeError):
        return None
    return arr if arr.dtype.kind in ('i', 'u', 'f', 'b') else None


def _write_json_node(parent: "h5py.Group", name: str, value: Any) -> List[str]:
    """Recursively write one JSON value into an HDF5 group.

    Conversion rules:
        None        → omitted (returns [])
        dict        → group; keys become child names
        numeric list / rectangular list-of-lists → dataset
        other list  → numbered subgroups ("0", "1", …)
        bool        → scalar uint8 dataset (checked before int)
        int / float → scalar dataset
        str         → scalar string dataset
    """
    if value is None:
        return []

    if isinstance(value, dict):
        grp = parent.require_group(name)
        written: List[str] = []
        for k, v in value.items():
            written.extend(_write_json_node(grp, str(k), v))
        return written

    if isinstance(value, list):
        arr = _try_numeric_array(value)
        if arr is not None:
            ds = parent.create_dataset(name, data=arr)
            return [ds.name]
        grp = parent.require_group(name)
        written = []
        for i, item in enumerate(value):
            written.extend(_write_json_node(grp, str(i), item))
        return written

    # bool must be tested before int (bool is a subclass of int in Python)
    if isinstance(value, bool):
        ds = parent.create_dataset(name, data=np.uint8(value))
        return [ds.name]

    if isinstance(value, (int, float)):
        ds = parent.create_dataset(name, data=value)
        return [ds.name]

    if isinstance(value, str):
        ds = parent.create_dataset(name, data=value)
        return [ds.name]

    return []  # unrecognised type: skip


def json_to_h5(
    output_path: str | Path,
    source: "str | Path | dict | list",
    overwrite: bool = False,
) -> List[str]:
    """Write JSON data to an HDF5 file, mapping structure to groups and datasets.

    The ``source`` argument is resolved in this order:

    1. ``dict`` or ``list`` — used directly as parsed JSON data.
    2. ``str`` or ``Path`` pointing to an existing file — read and parsed as JSON.
    3. ``str`` beginning with ``{`` or ``[`` — parsed as an inline JSON string.

    Conversion rules applied recursively:

    ==================  =====================================================
    JSON type           HDF5 result
    ==================  =====================================================
    ``dict``            group; each key becomes a child name
    uniform / rect list dataset (converted to ``np.ndarray``)
    jagged / mixed list numbered subgroups (``"0"``, ``"1"``, …)
    list of dicts       numbered subgroups
    ``int`` / ``float`` scalar dataset
    ``bool``            scalar ``uint8`` dataset
    ``str``             scalar string dataset
    ``None``            omitted silently
    ==================  =====================================================

    A top-level ``list`` is wrapped in a group named ``"data"``.

    Args:
        output_path: Destination HDF5 file to create.
        source:      JSON data as a parsed dict/list, a path to a ``.json``
                     file, or an inline JSON string.
        overwrite:   If ``True``, overwrite an existing ``output_path``.

    Returns:
        List of HDF5 dataset paths written into the output file.

    Raises:
        FileExistsError: If ``output_path`` exists and ``overwrite=False``.
        ValueError:      If ``source`` is a string that is neither a valid
                         file path nor valid JSON.

    Examples::

        # From a JSON file written by a tool
        json_to_h5("spectrum.h5", "spectrum_output.json")

        # From an in-memory dict (e.g. output of compute_poloidal_spectrum)
        m, psi, spec = compute_poloidal_spectrum(...)
        json_to_h5("spectrum.h5", {"m_modes": m.tolist(),
                                    "psi_norm": psi.tolist(),
                                    "spectrum": spec.tolist()})

        # From an inline JSON string
        json_to_h5("out.h5", '{"time": [0.0, 1.0], "ke": [1e-13, 9e-8]}')
    """
    import json as _json

    outp = Path(output_path)
    if outp.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {outp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(source, (dict, list)):
        data = source
    elif isinstance(source, (str, Path)):
        src_path = Path(source)
        if src_path.exists():
            with src_path.open() as fh:
                data = _json.load(fh)
        else:
            try:
                data = _json.loads(str(source))
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    f"source is neither an existing file nor valid JSON: {exc}"
                ) from exc
    else:
        raise TypeError(
            f"source must be dict, list, str, or Path; got {type(source).__name__}"
        )

    written: List[str] = []
    with h5py.File(outp, 'w') as f:
        if isinstance(data, dict):
            for k, v in data.items():
                written.extend(_write_json_node(f, str(k), v))
        else:
            written.extend(_write_json_node(f, "data", data))

    return written


