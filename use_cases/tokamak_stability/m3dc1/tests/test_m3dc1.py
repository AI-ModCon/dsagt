"""Unit tests for the m3dc1 package.

Tests are split into:
  - Unit tests (no data, no write_neo_input): mock or analytic only.
  - Integration tests (require sparc_1425 data and write_neo_input on PATH):
    marked with @pytest.mark.integration.

Run unit tests only:
    pytest fusion_io/m3dc1/tests/test_m3dc1.py -m "not integration"

Run all tests (requires data):
    pytest fusion_io/m3dc1/tests/test_m3dc1.py
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Fixtures
SPARC_DIR = Path("/Users/kparfrey/data/asv/run1/sparc_1425")
HAS_DATA = SPARC_DIR.is_dir()

try:
    import fio_py  # noqa: F401
    HAS_FIO_PY = True
except ImportError:
    HAS_FIO_PY = False

integration = pytest.mark.skipif(
    not (HAS_DATA and HAS_FIO_PY),
    reason="Requires sparc_1425 data and fio_py extension (fusion-io must be built)",
)


# ---------------------------------------------------------------------------
# Unit tests — eval_field
# ---------------------------------------------------------------------------

def _make_mock_sim(ftype="scalar", return_val=42.0):
    """Create a minimal fpy.sim_data mock."""
    sim = MagicMock()
    sim.typedict = {
        "psi": ("psi", "scalar", None, "composite"),
        "B":   ("magnetic field", "vector", None, "simple"),
        "p":   ("total pressure", "scalar", None, "simple"),
    }
    fld = MagicMock()
    fld.ftype = ftype
    if ftype == "scalar":
        fld.evaluate.return_value = (return_val,)
    else:
        fld.evaluate.return_value = (1.0, 2.0, 3.0)
    sim.get_field.return_value = fld
    sim.timeslice = 0
    return sim


def test_eval_field_scalar_shape():
    from m3dc1 import eval_field
    sim = _make_mock_sim(ftype="scalar", return_val=5.0)
    R = np.ones((3, 4))
    phi = np.zeros((3, 4))
    Z = np.zeros((3, 4))
    out = eval_field("psi", R, phi, Z, coord="scalar", sim=sim, quiet=True)
    assert out.shape == (3, 4)
    assert np.all(out == 5.0)


def test_eval_field_vector_shape():
    from m3dc1 import eval_field
    sim = _make_mock_sim(ftype="vector")
    R = np.ones(10)
    phi = np.zeros(10)
    Z = np.zeros(10)
    out = eval_field("B", R, phi, Z, coord="vector", sim=sim, quiet=True)
    assert out.shape == (3, 10)


def test_eval_field_vector_component_shape():
    from m3dc1 import eval_field
    sim = _make_mock_sim(ftype="vector")
    R = np.ones(5)
    phi = np.zeros(5)
    Z = np.zeros(5)
    out = eval_field("B", R, phi, Z, coord="R", sim=sim, quiet=True)
    assert out.shape == (5,)
    assert np.all(out == 1.0)


def test_eval_field_nan_on_none():
    from m3dc1 import eval_field
    sim = _make_mock_sim(ftype="scalar")
    fld = MagicMock()
    fld.ftype = "scalar"
    # Simulate out-of-domain: fpy returns (None,)
    fld.evaluate.return_value = (None,)
    sim.get_field.return_value = fld
    R = np.array([1.0])
    out = eval_field("psi", R, R * 0, R * 0, coord="scalar", sim=sim, quiet=True)
    assert np.isnan(out[0])


# ---------------------------------------------------------------------------
# Unit tests — _neo_input.flux_surface_average
# ---------------------------------------------------------------------------

def test_fsa_uniform():
    """FSA of f=1 with Jac=1 over [0,2pi]x[0,2pi] should be 1."""
    from m3dc1._neo_input import flux_surface_average
    ntheta, nphi = 64, 4
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    q = np.ones((ntheta, nphi))
    j = np.ones((ntheta, nphi))
    result = flux_surface_average(q, j, theta, phi)
    assert abs(result - 1.0) < 1e-10


def test_fsa_linear():
    """FSA of cos(theta) ≈ 0 (trapezoidal error bounded by 1/ntheta for open interval)."""
    from m3dc1._neo_input import flux_surface_average
    ntheta, nphi = 512, 4
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    th2d = theta[:, np.newaxis] * np.ones((1, nphi))
    q = np.cos(th2d)
    j = np.ones((ntheta, nphi))
    result = flux_surface_average(q, j, theta, phi)
    # Trapezoidal rule on open interval [0, 2π) has O(1/N) error for periodic f
    assert abs(result) < 2.0 / ntheta


# ---------------------------------------------------------------------------
# Unit tests — get_time_of_slice (mocked h5py)
# ---------------------------------------------------------------------------

def test_get_time_of_slice_alfven():
    """Returns float in Alfvén units from h5 mock."""
    from m3dc1 import get_time_of_slice
    mock_h5 = MagicMock()
    mock_h5.__enter__ = lambda s: s
    mock_h5.__exit__ = MagicMock(return_value=False)
    mock_h5.__contains__ = lambda s, k: k == "time_001"
    mock_h5.__getitem__ = lambda s, k: MagicMock(attrs={"time": 500.0})

    with patch("h5py.File", return_value=mock_h5):
        t = get_time_of_slice(1, filename="C1.h5", units="alfven", quiet=True)
    assert t == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Unit tests — get_timetrace (mocked sim)
# ---------------------------------------------------------------------------

def test_get_timetrace_ke_alias():
    """'ke' is aliased to E_K3."""
    from m3dc1 import get_timetrace
    sim = MagicMock()
    time = np.linspace(0, 100, 101)
    ke = np.exp(0.01 * time)
    sim._all_traces = {"time": time, "E_K3": ke}
    sim.filename = "C1.h5"
    t, vals, label, units_str = get_timetrace("ke", sim=sim, quiet=True)
    assert label == "E_K3"
    assert len(vals) == len(t)


def test_get_timetrace_growth():
    """Growth mode: gamma of exp(2*gamma0*t) = gamma0."""
    from m3dc1 import get_timetrace
    gamma0 = 0.005
    sim = MagicMock()
    t_arr = np.linspace(0, 1000, 1001)
    ke_arr = np.exp(2 * gamma0 * t_arr)
    sim._all_traces = {"time": t_arr, "E_K3": ke_arr}
    sim.filename = "C1.h5"
    t, vals, label, units_str = get_timetrace(
        "ke", sim=sim, growth=True, quiet=True
    )
    # Should recover gamma0 (to within trapezoidal discretisation error)
    assert np.allclose(vals, gamma0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Unit tests — get_shape
# ---------------------------------------------------------------------------

def test_get_shape_returns_dict():
    """get_shape returns dict with correct keys even on simple mock data."""
    # We can't call get_shape without a real sim and matplotlib, so just
    # verify the key set when provided a working mock.
    from m3dc1 import get_shape
    import matplotlib
    matplotlib.use("Agg")

    sim = MagicMock()
    # Use plain dicts so dict["key"] access works without MagicMock magic-method subtleties
    sim._all_attrs = {
        "scalars": {
            "psi_lcfs": np.array([0.25]),
            "xmag": np.array([1.8]),
            "zmag": np.array([0.0]),
        }
    }

    # Mesh: circle of radius 0.5 centred at R=1.8
    n = 100
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R_c = 1.8 + 0.5 * np.cos(angles)
    Z_c = 0.0 + 0.5 * np.sin(angles)
    elements = np.zeros((n, 6))
    elements[:, 4] = R_c
    elements[:, 5] = Z_c
    mesh_obj = MagicMock()
    mesh_obj.elements = elements
    sim.get_mesh.return_value = mesh_obj
    sim.timeslice = 0

    # Stub eval_field to return circular psi (centre at R=1.8, Z=0)
    with patch("m3dc1.eval_field") as mock_ef:
        def _psi_stub(field, R, phi, Z, **kw):
            return (R - 1.8) ** 2 + Z ** 2
        mock_ef.side_effect = _psi_stub
        result = get_shape(sim, res=50)

    assert set(result.keys()) == {"R0", "a", "kappa", "delta"}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@integration
def test_eval_field_psi_sparc1425():
    """psi increases monotonically from axis to LCFS."""
    from fpy import sim_data
    from m3dc1 import eval_field
    sim = sim_data(filename=str(SPARC_DIR / "C1.h5"), time=0)
    R_mag = float(np.asarray(sim._all_attrs["scalars"]["xmag"])[0])
    Z_mag = float(np.asarray(sim._all_attrs["scalars"]["zmag"])[0])
    R_line = np.linspace(R_mag, R_mag + 0.5, 20)
    psi = eval_field("psi", R_line, np.zeros(20), np.full(20, Z_mag),
                     sim=sim, time=0, quiet=True)
    assert np.all(np.diff(psi[~np.isnan(psi)]) > 0)


@integration
def test_get_time_of_slice_mks():
    """Returns a positive finite time in seconds for sparc_1425 snapshot 1."""
    from m3dc1 import get_time_of_slice
    t = get_time_of_slice(1, filename=str(SPARC_DIR / "C1.h5"),
                          units="mks", quiet=True)
    assert np.isfinite(t) and t > 0


@integration
def test_get_shape_sparc1425():
    """R0 ≈ 1.85 m, kappa > 1, 0 < delta < 1."""
    from fpy import sim_data
    from m3dc1 import get_shape
    sim = sim_data(filename=str(SPARC_DIR / "C1.h5"), time=0)
    shape = get_shape(sim, res=100)
    assert shape, "get_shape returned empty dict"
    assert abs(shape["R0"] - 1.855) < 0.1, f"R0={shape['R0']}"
    assert shape["kappa"] > 1, f"kappa={shape['kappa']}"
    assert 0 < shape["delta"] < 1, f"delta={shape['delta']}"


@integration
def test_flux_average_q_sparc1425():
    """q-profile is monotonically increasing for sparc_1425 equilibrium."""
    from fpy import sim_data
    from m3dc1 import flux_average
    sim = sim_data(filename=str(SPARC_DIR / "C1.h5"), time=0)
    psin, q = flux_average("q", sim=sim, points=30)
    assert np.all(np.diff(q) > 0), f"q not monotone: {q}"


@integration
def test_flux_coordinates_psi_norm_sparc1425():
    """fc.psi_norm spans [0.01, 0.99] for sparc_1425."""
    from fpy import sim_data
    from m3dc1 import flux_coordinates
    sim = sim_data(filename=str(SPARC_DIR / "C1.h5"), time=0)
    fc_obj = flux_coordinates(sim=sim, points=30)
    psin = fc_obj.fc.psi_norm
    assert psin[0] < 0.05
    assert psin[-1] > 0.95
