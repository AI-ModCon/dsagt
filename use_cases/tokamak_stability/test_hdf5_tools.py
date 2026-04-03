import h5py
from pathlib import Path
from dsagt.tools.hdf5 import list_h5_files, list_h5_variables, repackage_h5


def _make_file(path: Path, datasets: dict):
    """Helper: create an HDF5 file with the given {dataset_path: data} mapping."""
    with h5py.File(path, 'w') as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)


# ---------------------------------------------------------------------------
# list_h5_files / list_h5_variables
# ---------------------------------------------------------------------------

def test_list_h5_files(tmp_path):
    _make_file(tmp_path / "a.h5", {"x": [1]})
    _make_file(tmp_path / "b.h5", {"x": [2]})
    files = list_h5_files(tmp_path, recursive=False)
    assert len(files) == 2

def test_list_h5_variables(tmp_path):
    src = tmp_path / "src.h5"
    _make_file(src, {"group1/dset1": [1, 2, 3], "dset_root": 42})
    vars_ = list_h5_variables(src)
    assert "group1/dset1" in vars_
    assert "dset_root" in vars_


# ---------------------------------------------------------------------------
# repackage_h5 — grouping behaviour
# ---------------------------------------------------------------------------

def test_repackage_single_source_no_grouping(tmp_path):
    """Single source: variables written at the top level, no stem prefix."""
    src = tmp_path / "src.h5"
    _make_file(src, {"var1": [1, 2, 3], "var2": [4, 5, 6]})

    out = tmp_path / "out.h5"
    written = repackage_h5(out, [src])

    assert "/var1" in written
    assert "/var2" in written
    with h5py.File(out, 'r') as f:
        assert "var1" in f
        assert "var2" in f


def test_repackage_multiple_sources_same_dir_stem_grouping(tmp_path):
    """Multiple sources in the same directory: group by file stem only."""
    src1 = tmp_path / "src1.h5"
    src2 = tmp_path / "src2.h5"
    _make_file(src1, {"group1/dset1": [1, 2, 3]})
    _make_file(src2, {"group2/dset2": [[1.0, 2.0], [3.0, 4.0]]})

    out = tmp_path / "combined.h5"
    written = repackage_h5(out, [src1, src2], variables=["group1/dset1", "group2/dset2"])

    assert "/src1/group1/dset1" in written
    assert "/src2/group2/dset2" in written
    with h5py.File(out, 'r') as f:
        assert f["src1/group1/dset1"][()].tolist() == [1, 2, 3]
        assert f["src2/group2/dset2"][()].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_repackage_different_dirs_same_grandparent_dir_prefix(tmp_path):
    """Sources in different subdirs of the same parent: group by dir + stem."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    src1 = dir1 / "file.h5"
    src2 = dir2 / "file.h5"
    _make_file(src1, {"var1": [1, 2]})
    _make_file(src2, {"var1": [3, 4]})

    out = tmp_path / "out.h5"
    written = repackage_h5(out, [src1, src2], variables=["var1"])

    assert "/dir1/file/var1" in written
    assert "/dir2/file/var1" in written
    with h5py.File(out, 'r') as f:
        assert f["dir1/file/var1"][()].tolist() == [1, 2]
        assert f["dir2/file/var1"][()].tolist() == [3, 4]


def test_repackage_different_grandparent_dirs_granddir_prefix(tmp_path):
    """Sources under different grandparent dirs: group by grandparent + dir + stem."""
    grand1 = tmp_path / "grand1" / "dir"
    grand2 = tmp_path / "grand2" / "dir"
    grand1.mkdir(parents=True)
    grand2.mkdir(parents=True)
    src1 = grand1 / "file.h5"
    src2 = grand2 / "file.h5"
    _make_file(src1, {"var1": [10, 20]})
    _make_file(src2, {"var1": [30, 40]})

    out = tmp_path / "out.h5"
    written = repackage_h5(out, [src1, src2], variables=["var1"])

    assert "/grand1/dir/file/var1" in written
    assert "/grand2/dir/file/var1" in written
    with h5py.File(out, 'r') as f:
        assert f["grand1/dir/file/var1"][()].tolist() == [10, 20]
        assert f["grand2/dir/file/var1"][()].tolist() == [30, 40]


# ---------------------------------------------------------------------------
# repackage_h5 — selection argument
# ---------------------------------------------------------------------------

def test_repackage_selection_different_vars_per_source(tmp_path):
    """selection allows taking different variables from different source files.

    This scenario cannot be expressed with `variables` alone, since that applies
    the same list to every source. Here run1.h5 has both temperature and pressure
    but we only want temperature from it; run2.h5 also has both but we only want
    pressure from it.
    """
    run1 = tmp_path / "run1.h5"
    run2 = tmp_path / "run2.h5"
    _make_file(run1, {"temperature": [300.0, 310.0], "pressure": [1.0, 1.1]})
    _make_file(run2, {"temperature": [320.0, 330.0], "pressure": [1.2, 1.3]})

    out = tmp_path / "out.h5"
    written = repackage_h5(
        out,
        sources=[run1, run2],
        selection={
            str(run1): ["temperature"],
            str(run2): ["pressure"],
        },
    )

    # Both sources are in the same dir, so stem-level grouping applies
    assert "/run1/temperature" in written
    assert "/run2/pressure" in written
    # Variables not selected must be absent
    with h5py.File(out, 'r') as f:
        assert "run1/temperature" in f
        assert "run2/pressure" in f
        assert "run1/pressure" not in f
        assert "run2/temperature" not in f


def test_repackage_selection_overrides_variables(tmp_path):
    """selection takes precedence over variables for sources that appear in it."""
    src1 = tmp_path / "src1.h5"
    src2 = tmp_path / "src2.h5"
    _make_file(src1, {"a": [1], "b": [2]})
    _make_file(src2, {"a": [3], "b": [4]})

    out = tmp_path / "out.h5"
    # variables says take "a" from everything, but selection overrides src1 to "b"
    written = repackage_h5(
        out,
        sources=[src1, src2],
        variables=["a"],
        selection={str(src1): ["b"]},
    )

    assert "/src1/b" in written
    assert "/src2/a" in written
    assert "/src1/a" not in written


# ---------------------------------------------------------------------------
# repackage_h5 — attributes are preserved
# ---------------------------------------------------------------------------

def test_repackage_attributes_preserved(tmp_path):
    src = tmp_path / "src.h5"
    with h5py.File(src, 'w') as f:
        ds = f.create_dataset("dset", data=[1, 2, 3])
        ds.attrs["units"] = "m"

    out = tmp_path / "out.h5"
    repackage_h5(out, [src])

    with h5py.File(out, 'r') as f:
        assert f["dset"].attrs.get("units") == "m"
