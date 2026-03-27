import h5py
from pathlib import Path
from dsagt.tools.hdf5 import list_h5_files, list_h5_variables, repackage_h5


def test_repackage_hdf5_basic(tmp_path):
    """Create two small HDF5 source files, then repackage selected datasets."""
    d = tmp_path
    file1 = d / "src1.h5"
    file2 = d / "src2.h5"

    # create source files
    with h5py.File(file1, 'w') as f:
        f.create_dataset('group1/dset1', data=[1, 2, 3])
        f.create_dataset('dset_root', data=42)
        f['group1/dset1'].attrs['units'] = 'm'

    with h5py.File(file2, 'w') as f:
        f.create_dataset('group2/dset2', data=[[1.0, 2.0], [3.0, 4.0]])
        f['group2/dset2'].attrs['desc'] = 'matrix'

    # list .h5 files
    files = list_h5_files(d, recursive=False)
    assert len(files) == 2

    # list variables in the first file
    vars1 = list_h5_variables(file1)
    assert 'group1/dset1' in vars1
    assert 'dset_root' in vars1

    # repackage a subset into a new file
    out = d / 'combined.h5'
    written = repackage_h5(out, [file1, file2], variables=['group1/dset1', 'group2/dset2'], overwrite=True)
    assert any('group1/dset1' in w for w in written)
    assert any('group2/dset2' in w for w in written)

    # verify contents and attributes
    with h5py.File(out, 'r') as f:
        # group1/dset1 should be present (or under a source-stem prefix)
        if 'group1/dset1' in f:
            ds_path = 'group1/dset1'
        else:
            ds_path = f"{file1.stem}/group1/dset1"
        assert f[ds_path][()].tolist() == [1, 2, 3]
        assert f[ds_path].attrs.get('units') == 'm'

        # group2/dset2
        if 'group2/dset2' in f:
            ds2 = 'group2/dset2'
        else:
            ds2 = f"{file2.stem}/group2/dset2"
        assert (f[ds2][()].tolist() == [[1.0, 2.0], [3.0, 4.0]])
        assert f[ds2].attrs.get('desc') == 'matrix'
