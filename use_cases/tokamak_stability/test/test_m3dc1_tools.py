"""Tests for m3dc1_tools.py.

Unit tests use temporary fixtures and run without any external data or
libraries. Integration tests require the example dataset at DATA_DIR and are
automatically skipped when it is absent. Tests for m3dc1/fpy-dependent
functions are additionally skipped when those libraries are not installed.
"""
import pytest
import numpy as np
import h5py
from pathlib import Path

from m3dc1_tools import (
    read_c1input,
    list_time_snapshots,
    read_snapshot_time,
    read_scalar_traces,
    read_case_metadata,
    read_mesh_vertices,
    make_evaluation_grid,
    compute_ke_growth_trace,
    compute_growth_rate,
    compute_q95,
)

DATA_DIR = Path("/Users/kparfrey/data/asv/run1/sparc_1425")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sparc_case():
    if not DATA_DIR.exists():
        pytest.skip(f"Example data not available at {DATA_DIR}")
    return DATA_DIR


@pytest.fixture
def require_m3dc1():
    try:
        import m3dc1  # noqa: F401
        import fpy    # noqa: F401
    except ImportError:
        pytest.skip("m3dc1/fpy not installed")


@pytest.fixture
def minimal_c1input(tmp_path):
    """Write a minimal C1input with all tracked parameters."""
    content = """&inputnl
\tntimemax = 500\t! steps
\tntimepr = 250\t! output interval
\tdt = 0.5\t! time step
\tlinear = 1\t! linear run
\tnumvar = 3\t! 6-field
\tion_mass = 2.\t! deuterium
\tzeff = 1.\t! Z_eff
pscale = 0.815
batemanscale = 1.063
ntor = 9
 /
"""
    (tmp_path / "C1input").write_text(content)
    return tmp_path


@pytest.fixture
def case_with_snapshots(tmp_path):
    """Case directory containing dummy time_NNN.h5 snapshot files."""
    for idx in (0, 2, 5):
        fname = tmp_path / f"time_{idx:03d}.h5"
        with h5py.File(fname, "w") as f:
            f.attrs["ntimestep"] = idx * 100
            f.attrs["time"] = float(idx * 100)
    return tmp_path


# ---------------------------------------------------------------------------
# read_c1input — unit tests
# ---------------------------------------------------------------------------

def test_read_c1input_all_params(minimal_c1input):
    params = read_c1input(minimal_c1input)
    assert params["ntor"] == 9
    assert params["ntimemax"] == 500
    assert params["ntimepr"] == 250
    assert abs(params["dt"] - 0.5) < 1e-9
    assert params["linear"] == 1
    assert params["numvar"] == 3
    assert abs(params["ion_mass"] - 2.0) < 1e-9
    assert abs(params["zeff"] - 1.0) < 1e-9
    assert abs(params["pscale"] - 0.815) < 1e-6
    assert abs(params["batemanscale"] - 1.063) < 1e-6


def test_read_c1input_missing_file(tmp_path):
    params = read_c1input(tmp_path)
    assert all(v is None for v in params.values())


def test_read_c1input_partial_params(tmp_path):
    (tmp_path / "C1input").write_text("&inputnl\n\tntor = 3\n /\n")
    params = read_c1input(tmp_path)
    assert params["ntor"] == 3
    assert params["pscale"] is None
    assert params["linear"] is None


def test_read_c1input_returns_int_for_int_keys(minimal_c1input):
    params = read_c1input(minimal_c1input)
    assert isinstance(params["ntor"], int)
    assert isinstance(params["linear"], int)
    assert isinstance(params["numvar"], int)


def test_read_c1input_returns_float_for_float_keys(minimal_c1input):
    params = read_c1input(minimal_c1input)
    assert isinstance(params["pscale"], float)
    assert isinstance(params["batemanscale"], float)


def test_read_c1input_comment_lines_ignored(tmp_path):
    content = "! ntor = 999\n&inputnl\nntor = 5\n /\n"
    (tmp_path / "C1input").write_text(content)
    params = read_c1input(tmp_path)
    assert params["ntor"] == 5


# ---------------------------------------------------------------------------
# list_time_snapshots — unit tests
# ---------------------------------------------------------------------------

def test_list_time_snapshots_finds_files(case_with_snapshots):
    result = list_time_snapshots(case_with_snapshots)
    assert result == [0, 2, 5]


def test_list_time_snapshots_empty_dir(tmp_path):
    assert list_time_snapshots(tmp_path) == []


def test_list_time_snapshots_ignores_non_numeric(tmp_path):
    with h5py.File(tmp_path / "time_abc.h5", "w"):
        pass
    with h5py.File(tmp_path / "time_001.h5", "w") as f:
        f.attrs["time"] = 1.0
    result = list_time_snapshots(tmp_path)
    assert result == [1]


def test_list_time_snapshots_deduplicated(tmp_path):
    """Only the file indices matter; no duplicates."""
    for name in ("time_003.h5", "time_003.h5"):
        with h5py.File(tmp_path / name, "w"):
            pass
    assert list_time_snapshots(tmp_path) == [3]


# ---------------------------------------------------------------------------
# read_snapshot_time — unit tests
# ---------------------------------------------------------------------------

def test_read_snapshot_time_from_file(case_with_snapshots):
    t = read_snapshot_time(case_with_snapshots, 2)
    assert abs(t - 200.0) < 1e-6


def test_read_snapshot_time_missing_returns_nan(tmp_path):
    t = read_snapshot_time(tmp_path, 99)
    assert np.isnan(t)


def test_read_snapshot_time_from_c1h5(tmp_path):
    """Falls back to reading from C1.h5/time_NNN group."""
    with h5py.File(tmp_path / "C1.h5", "w") as f:
        grp = f.create_group("time_001")
        grp.attrs["time"] = 42.0
    t = read_snapshot_time(tmp_path, 1)
    assert abs(t - 42.0) < 1e-6


# ---------------------------------------------------------------------------
# make_evaluation_grid — unit tests
# ---------------------------------------------------------------------------

def _sample_mesh():
    rng = np.random.default_rng(0)
    R = rng.uniform(1.0, 2.0, 50).astype(np.float32)
    Z = rng.uniform(-1.0, 1.0, 50).astype(np.float32)
    return R, Z


def test_make_evaluation_grid_mesh_mode():
    R_m, Z_m = _sample_mesh()
    R, Z, phi_arr = make_evaluation_grid(R_m, Z_m, mode="mesh")
    assert R.shape == R_m.shape
    assert Z.shape == Z_m.shape
    assert phi_arr.shape == R_m.shape
    np.testing.assert_array_equal(R, R_m)


def test_make_evaluation_grid_grid_mode():
    R_m, Z_m = _sample_mesh()
    R, Z, phi_arr = make_evaluation_grid(R_m, Z_m, mode="grid", grid_res=10)
    assert R.shape == (10, 10)
    assert Z.shape == (10, 10)
    assert phi_arr.shape == (10, 10)


def test_make_evaluation_grid_phi_constant():
    R_m, Z_m = _sample_mesh()
    _, _, phi_arr = make_evaluation_grid(R_m, Z_m, phi=1.23)
    assert np.all(phi_arr == 1.23)


def test_make_evaluation_grid_bounding_box():
    R_m, Z_m = _sample_mesh()
    R, Z, _ = make_evaluation_grid(R_m, Z_m, mode="grid", grid_res=20)
    assert R.min() >= R_m.min() - 1e-6
    assert R.max() <= R_m.max() + 1e-6
    assert Z.min() >= Z_m.min() - 1e-6
    assert Z.max() <= Z_m.max() + 1e-6


def test_make_evaluation_grid_invalid_mode():
    R_m, Z_m = _sample_mesh()
    with pytest.raises(ValueError):
        make_evaluation_grid(R_m, Z_m, mode="invalid")


# ---------------------------------------------------------------------------
# compute_q95 — unit tests
# ---------------------------------------------------------------------------

def test_compute_q95_analytical():
    psin = np.linspace(0, 1, 11)
    q = 1 + 2 * psin          # q=1 at axis, q=3 at edge → q95 = 1 + 2*0.95 = 2.9
    assert abs(compute_q95(psin, q) - 2.9) < 1e-10


def test_compute_q95_below_range_returns_nan():
    psin = np.linspace(0, 0.9, 10)   # does not reach 0.95
    q = np.ones(10)
    assert np.isnan(compute_q95(psin, q))


def test_compute_q95_empty_returns_nan():
    assert np.isnan(compute_q95(np.array([]), np.array([])))


def test_compute_q95_at_boundary():
    psin = np.array([0.0, 0.95, 1.0])
    q = np.array([1.0, 2.5, 3.0])
    assert abs(compute_q95(psin, q) - 2.5) < 1e-10


# ---------------------------------------------------------------------------
# compute_growth_rate — unit tests
# ---------------------------------------------------------------------------

def _make_c1h5_with_scalars(tmp_path, ke_values, time_values=None):
    if time_values is None:
        time_values = np.arange(len(ke_values), dtype=float)
    with h5py.File(tmp_path / "C1.h5", "w") as f:
        sc = f.create_group("scalars")
        sc.create_dataset("E_K3", data=np.array(ke_values, dtype=np.float32))
        sc.create_dataset("time", data=np.array(time_values, dtype=np.float32))
    return tmp_path


def test_compute_growth_rate_nan_on_zeros(tmp_path):
    _make_c1h5_with_scalars(tmp_path, [0.0, 0.0, 0.0])
    assert np.isnan(compute_growth_rate(tmp_path))


def test_compute_growth_rate_nan_on_single_nonzero(tmp_path):
    _make_c1h5_with_scalars(tmp_path, [0.0, 1.0])
    assert np.isnan(compute_growth_rate(tmp_path))


def test_compute_growth_rate_exponential_growth(tmp_path):
    """For E_K3 = exp(2*gamma*t), the recovered rate should equal gamma."""
    gamma_true = 0.05
    t = np.arange(0, 101, dtype=float)
    ke = np.exp(2 * gamma_true * t)
    ke[0] = 0.0   # initial timestep often zero
    _make_c1h5_with_scalars(tmp_path, ke, t)
    gamma_est = compute_growth_rate(tmp_path)
    assert abs(gamma_est - gamma_true) < 1e-4


def test_compute_ke_growth_trace_shape(tmp_path):
    ke = np.linspace(1e-14, 1e-7, 1001)
    t = np.arange(1001, dtype=float)
    _make_c1h5_with_scalars(tmp_path, ke, t)
    time_out, ke_out = compute_ke_growth_trace(tmp_path)
    assert len(time_out) == 1001
    assert len(ke_out) == 1001


# ---------------------------------------------------------------------------
# Integration tests — require real data at DATA_DIR
# ---------------------------------------------------------------------------

def test_read_c1input_sparc1425(sparc_case):
    params = read_c1input(sparc_case)
    assert params["ntor"] == 9
    assert params["linear"] == 1
    assert params["numvar"] == 3
    assert abs(params["pscale"] - 0.815) < 0.01
    assert abs(params["ion_mass"] - 2.0) < 0.01


def test_list_time_snapshots_sparc1425(sparc_case):
    snaps = list_time_snapshots(sparc_case)
    assert snaps == [0, 1]


def test_read_snapshot_time_alfven(sparc_case):
    t = read_snapshot_time(sparc_case, 0, units="alfven")
    assert abs(t - 0.0) < 1e-6
    t1 = read_snapshot_time(sparc_case, 1, units="alfven")
    assert abs(t1 - 1000.0) < 1.0


def test_read_scalar_traces_subset(sparc_case):
    traces = read_scalar_traces(sparc_case, names=["E_K3", "time"])
    assert "E_K3" in traces
    assert "time" in traces
    assert len(traces["time"]) == 1001
    assert traces["E_K3"][-1] > traces["E_K3"][1]   # mode grows


def test_read_scalar_traces_all(sparc_case):
    traces = read_scalar_traces(sparc_case)
    assert len(traces) > 50
    assert "xmag" in traces


def test_read_case_metadata_sparc1425(sparc_case):
    meta = read_case_metadata(sparc_case)
    assert meta["params"]["ntor"] == 9
    assert meta["snapshots"] == [0, 1]
    assert abs(meta["final_time"] - 1000.0) < 1.0
    assert abs(meta["R_mag"] - 1.855) < 0.01
    assert abs(meta["Z_mag"]) < 0.01
    assert abs(meta["R_xpoint"] - 1.492) < 0.01
    assert meta["psi_min"] < 0          # negative at axis
    assert meta["psi_lcfs"] > meta["psi_min"]


def test_read_mesh_vertices_sparc1425(sparc_case):
    R, Z = read_mesh_vertices(sparc_case / "C1.h5")
    assert R.shape == Z.shape
    assert len(R) > 1000
    assert len(R) < 18555   # fewer unique vertices than elements
    assert R.min() >= 1.24
    assert R.max() <= 2.56
    assert Z.min() >= -1.26
    assert Z.max() <= 1.26


def test_read_mesh_vertices_equilibrium_h5(sparc_case):
    R1, Z1 = read_mesh_vertices(sparc_case / "C1.h5")
    R2, Z2 = read_mesh_vertices(sparc_case / "equilibrium.h5")
    np.testing.assert_array_equal(R1, R2)
    np.testing.assert_array_equal(Z1, Z2)


def test_compute_ke_growth_trace_sparc1425(sparc_case):
    time, ke = compute_ke_growth_trace(sparc_case)
    assert len(time) == 1001
    assert len(ke) == 1001
    assert ke[-1] > ke[1]


def test_compute_growth_rate_sparc1425(sparc_case):
    gamma = compute_growth_rate(sparc_case)
    assert np.isfinite(gamma)
    assert 0.001 < gamma < 0.1    # reasonable range for a linear MHD run


def test_compute_growth_rate_time_idx_sparc1425(sparc_case):
    gamma_full = compute_growth_rate(sparc_case)
    gamma_t1 = compute_growth_rate(sparc_case, time_idx=1)
    # With all data, both should be close (time_idx=1 → ntimestep=1000 = full trace)
    assert abs(gamma_full - gamma_t1) < 1e-6


# ---------------------------------------------------------------------------
# Integration tests — additionally require m3dc1 + fpy
# ---------------------------------------------------------------------------

def test_compute_flux_average_profiles_sparc1425(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_flux_average_profiles
    profiles = compute_flux_average_profiles(sparc_case)
    assert "p" in profiles
    assert "q" in profiles
    psin, q = profiles["q"]
    assert abs(psin[0]) < 0.05
    assert abs(psin[-1] - 1.0) < 0.05
    assert q.min() > 0.5


def test_compute_q95_sparc1425(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_flux_average_profiles
    profiles = compute_flux_average_profiles(sparc_case)
    psin, q = profiles["q"]
    q95 = compute_q95(psin, q)
    assert np.isfinite(q95)
    assert q95 > 1.0


def test_compute_miller_geometry_sparc1425(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_miller_geometry
    shape = compute_miller_geometry(sparc_case)
    assert set(shape.keys()) == {"R0", "a", "kappa", "delta"}
    assert 1.5 < shape["R0"] < 2.5
    assert shape["a"] > 0
    assert shape["kappa"] > 1.0


def test_compute_perturbed_fields_psi(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_perturbed_fields
    R, Z = read_mesh_vertices(sparc_case / "C1.h5")
    R, Z, phi = make_evaluation_grid(R, Z, mode="mesh")
    fields = compute_perturbed_fields(sparc_case, 1, R, Z, phi, fields=["psi"])
    assert "psi" in fields
    assert fields["psi"].shape == R.shape
    assert np.abs(fields["psi"]).max() > 0


def test_compute_perturbed_fields_zero_at_t0(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_perturbed_fields
    R, Z = read_mesh_vertices(sparc_case / "C1.h5")
    R, Z, phi = make_evaluation_grid(R, Z, mode="mesh")
    fields = compute_perturbed_fields(sparc_case, 0, R, Z, phi, fields=["psi"])
    assert "psi" in fields
    assert np.abs(fields["psi"]).max() < 1e-3


def test_compute_perturbed_fields_vector_B(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_perturbed_fields
    R, Z = read_mesh_vertices(sparc_case / "C1.h5")
    R, Z, phi = make_evaluation_grid(R, Z, mode="mesh")
    fields = compute_perturbed_fields(sparc_case, 1, R, Z, phi, fields=["B"])
    assert "BR" in fields
    assert "BPHI" in fields
    assert "BZ" in fields


def test_compute_perturbed_fields_skip_respected(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_perturbed_fields
    R, Z = read_mesh_vertices(sparc_case / "C1.h5")
    R, Z, phi = make_evaluation_grid(R, Z, mode="mesh")
    # psi should be absent when skipped
    fields = compute_perturbed_fields(
        sparc_case, 1, R, Z, phi, fields=["psi", "te"], skip_fields=["psi"]
    )
    assert "psi" not in fields
    assert "te" in fields


def test_compute_poloidal_spectrum_shape(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_poloidal_spectrum
    m_modes, psi_norm, spec = compute_poloidal_spectrum(sparc_case, 1, "p", points=50)
    assert m_modes.dtype == int or np.issubdtype(m_modes.dtype, np.integer)
    assert len(psi_norm) == 50
    assert spec.shape == (len(m_modes), 50)
    assert psi_norm[0] >= 0.0
    assert psi_norm[-1] <= 1.0 + 1e-6


def test_compute_poloidal_spectrum_full_fft(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_poloidal_spectrum
    m_half, _, spec_half = compute_poloidal_spectrum(sparc_case, 1, "p", points=50)
    m_full, _, spec_full = compute_poloidal_spectrum(
        sparc_case, 1, "p", points=50, full_fft=True
    )
    # full FFT should have exactly `points` m-modes (not mirrored)
    assert len(m_full) == 50
    # half-spectrum is mirrored so has 2*m_max+1 modes
    assert len(m_half) > 50


def test_compute_standard_spectra_keys(sparc_case, require_m3dc1):
    from m3dc1_tools import compute_standard_spectra
    result = compute_standard_spectra(sparc_case, 1, points=50)
    assert set(result.keys()) == {"p", "br", "bz", "bphi"}
    for key, (m, psi, spec) in result.items():
        assert spec.ndim == 2
        assert spec.shape[1] == 50
