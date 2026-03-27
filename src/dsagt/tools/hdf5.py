"""Small helpers for working with HDF5 files using h5py.

Functions:
- ``list_h5_files(directory, recursive=False)``: return list of .h5 files in directory.
- ``list_h5_variables(file_path)``: return list of dataset paths inside an HDF5 file.

Example:
	>>> list_h5_files('data')
	['data/run1.h5', 'data/run2.h5']

	>>> list_h5_variables('data/run1.h5')
	['group1/dset1', 'group2/sub/dset2']
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import h5py


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
	'repackage_h5',
]


def repackage_h5(
	output_path: str | Path,
	sources: List[str | Path],
	selection: dict | None = None,
	variables: List[str] | None = None,
	overwrite: bool = False,
) -> List[str]:
	"""Create a new HDF5 file containing a subset of datasets from sources.

	Args:
		output_path: destination HDF5 file to create.
		sources: list of source HDF5 file paths to read from.
		selection: optional mapping of source-path -> list of dataset paths
			to copy from that specific source. Keys should be strings or
			Path-like objects. If provided, it takes precedence for that
			source.
		variables: optional list of dataset paths to copy from any source
			(first match wins). If omitted, all datasets from each source
			are copied.
		overwrite: if True, overwrite existing `output_path`.

	Returns:
		List of dataset paths written into the output file.

	Notes:
		- If a dataset path already exists in the output file, the dataset
		  will be written under a prefix named after the source file stem
		  (e.g. "source1/<dataset_path>").
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

	for src in sources:
		srcp = Path(src)
		if not srcp.exists():
			continue

		try:
			with h5py.File(srcp, 'r') as fin:
				if str(srcp) in norm_sel:
					vars_to_copy = norm_sel[str(srcp)]
				elif variables is not None:
					vars_to_copy = list(variables)
				else:
					vars_to_copy = list_h5_variables(srcp)

				with h5py.File(outp, 'a') as fout:
					for v in vars_to_copy:
						if v not in fin:
							continue
						src_ds = fin[v]

						dest_path = v
						# If already exists, move under a source-based prefix
						try:
							fout[v]
							exists = True
						except Exception:
							exists = False
						if exists:
							prefix = srcp.stem
							dest_path = f"{prefix}/{v.lstrip('/')}"

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
						# If a dataset with same name exists in this group, replace
						name = parts[-1]
						if name in grp:
							del grp[name]
						new_ds = grp.create_dataset(name, data=data, **ds_kwargs)

						# copy attributes
						for k, val in src_ds.attrs.items():
							new_ds.attrs[k] = val

						written.append('/' + dest_path.strip('/'))
		except (OSError, IOError):
			continue

	return written


