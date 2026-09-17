"""Tests for m3dc1_plots.py.

Unit tests run without real data or m3dc1.  Integration tests (marked
``integration``) require real data at the path in CASE_DIR and a working
m3dc1 + fpy installation.
"""
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

import m3dc1_plots
from m3dc1_plots import _parse_geqdsk, _save_first_new_fig

CASE_DIR = Path("/Users/kparfrey/data/asv/run1/sparc_1425")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_geqdsk(tmp_path) -> Path:
    """Write a tiny but structurally valid GEQDSK file."""
    nw, nh = 4, 5
    header = f"  EQ  0  0  0 {nw} {nh}\n"
    # rdim zdim rcentr rleft zmid
    scalars_1 = "  1.0  2.0  1.5  0.5  0.0\n"
    # rmaxis zmaxis simag sibry bcentr
    scalars_2 = "  1.5  0.0  0.1  0.9  2.0\n"
    # current + 4 padding zeros (first 5 of the next block)
    scalars_3 = "  1.0e6  0.0  0.0  0.0  0.0\n"

    def _row(vals):
        return "  ".join(f"{v: .6e}" for v in vals) + "\n"

    fpol   = [1.0] * nw
    pres   = [0.0] * nw
    ffprim = [0.0] * nw
    pprim  = [0.0] * nw
    psirz  = list(np.linspace(0.1, 0.9, nw * nh))
    qpsi   = [2.0] * nw

    nbbbs, limitr = 3, 2
    rbbbs = [1.2, 1.5, 1.8]
    zbbbs = [-0.5, 0.5, 0.0]
    rlim  = [1.0, 2.0]
    zlim  = [-1.0, 1.0]

    lines = [header, scalars_1, scalars_2, scalars_3]

    def _write_1d(lst):
        chunk = 5
        for i in range(0, len(lst), chunk):
            lines.append(_row(lst[i: i + chunk]))

    _write_1d(fpol)
    _write_1d(pres)
    _write_1d(ffprim)
    _write_1d(pprim)
    _write_1d(psirz)
    _write_1d(qpsi)
    lines.append(f"  {nbbbs}  {limitr}\n")
    bdry = []
    for r, z in zip(rbbbs, zbbbs):
        bdry += [r, z]
    _write_1d(bdry)
    lim = []
    for r, z in zip(rlim, zlim):
        lim += [r, z]
    _write_1d(lim)

    gfile = tmp_path / "geqdsk"
    gfile.write_text("".join(lines))
    return gfile


@pytest.fixture
def fake_case_dir(tmp_path) -> Path:
    """Create a minimal case directory with a synthetic C1.h5 for h5py-only tests."""
    import h5py
    case_dir = tmp_path / "sparc_test"
    case_dir.mkdir()

    n_steps = 20
    time = np.linspace(0, 10, n_steps + 1)
    ke = 1e-6 * np.exp(0.5 * time)

    with h5py.File(case_dir / "C1.h5", "w") as f:
        sc = f.create_group("scalars")
        sc.create_dataset("time",   data=time)
        sc.create_dataset("E_K3",   data=ke)
        sc.create_dataset("xmag",   data=[1.5])
        sc.create_dataset("zmag",   data=[0.0])
        sc.create_dataset("xnull",  data=[1.3])
        sc.create_dataset("znull",  data=[-1.2])
        sc.create_dataset("psimin", data=[0.1])
        sc.create_dataset("psi_lcfs", data=[0.9])

    return case_dir


# ---------------------------------------------------------------------------
# Unit tests — _parse_geqdsk helper
# ---------------------------------------------------------------------------

class TestParseGeqdsk:
    def test_basic_structure(self, minimal_geqdsk):
        geo = _parse_geqdsk(minimal_geqdsk)
        assert "rg" in geo and "zg" in geo
        assert "psirzn" in geo
        assert "rmaxis" in geo

    def test_grid_shape(self, minimal_geqdsk):
        geo = _parse_geqdsk(minimal_geqdsk)
        nw, nh = 4, 5
        assert len(geo["rg"]) == nw
        assert len(geo["zg"]) == nh
        assert geo["psirz"].shape == (nh, nw)

    def test_psirzn_normalisation(self, minimal_geqdsk):
        geo = _parse_geqdsk(minimal_geqdsk)
        # psirzn should span 0..1 given simag=0.1, sibry=0.9, psirz=linspace(0.1,0.9)
        assert float(np.nanmin(geo["psirzn"])) == pytest.approx(0.0, abs=0.01)
        assert float(np.nanmax(geo["psirzn"])) == pytest.approx(1.0, abs=0.01)

    def test_boundary_parsed(self, minimal_geqdsk):
        geo = _parse_geqdsk(minimal_geqdsk)
        assert len(geo["rbbbs"]) == 3
        assert len(geo["zbbbs"]) == 3

    def test_limiter_parsed(self, minimal_geqdsk):
        geo = _parse_geqdsk(minimal_geqdsk)
        assert len(geo["rlim"]) == 2

    def test_z_grid_centred(self, minimal_geqdsk):
        geo = _parse_geqdsk(minimal_geqdsk)
        # zmid=0, zdim=2 → zg should span -1..1
        assert geo["zg"][0] == pytest.approx(-1.0, abs=1e-9)
        assert geo["zg"][-1] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests — _save_first_new_fig helper
# ---------------------------------------------------------------------------

class TestSaveFirstNewFig:
    def test_saves_file(self, tmp_path):
        import matplotlib.pyplot as plt
        before = set(plt.get_fignums())
        fig = plt.figure()
        out = tmp_path / "fig.png"
        _save_first_new_fig(before, out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_closes_all_new_figs(self, tmp_path):
        import matplotlib.pyplot as plt
        before = set(plt.get_fignums())
        plt.figure()
        plt.figure()
        out = tmp_path / "fig.png"
        _save_first_new_fig(before, out, dpi=72)
        after = set(plt.get_fignums())
        assert after == before

    def test_raises_if_no_new_fig(self, tmp_path):
        import matplotlib.pyplot as plt
        before = set(plt.get_fignums())
        out = tmp_path / "fig.png"
        with pytest.raises(RuntimeError, match="no new figure"):
            _save_first_new_fig(before, out, dpi=72)


# ---------------------------------------------------------------------------
# Unit tests — Category A functions with synthetic data
# ---------------------------------------------------------------------------

class TestPlotKineticEnergy:
    def test_creates_file(self, fake_case_dir, tmp_path):
        out = tmp_path / "ke.png"
        result = m3dc1_plots.plot_kinetic_energy(fake_case_dir, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 100

    def test_no_annotation_option(self, fake_case_dir, tmp_path):
        out = tmp_path / "ke_no_ann.png"
        m3dc1_plots.plot_kinetic_energy(fake_case_dir, out, annotate_growth_rate=False)
        assert out.exists()

    def test_raises_on_all_zero_ke(self, tmp_path):
        import h5py
        case_dir = tmp_path / "zero_ke"
        case_dir.mkdir()
        with h5py.File(case_dir / "C1.h5", "w") as f:
            sc = f.create_group("scalars")
            sc.create_dataset("time", data=np.zeros(5))
            sc.create_dataset("E_K3", data=np.zeros(5))
        with pytest.raises(ValueError, match="non-zero"):
            m3dc1_plots.plot_kinetic_energy(case_dir, tmp_path / "ke.png")


class TestPlotGrowthRateVsTime:
    def test_creates_file(self, fake_case_dir, tmp_path):
        out = tmp_path / "gr.png"
        result = m3dc1_plots.plot_growth_rate_vs_time(fake_case_dir, out)
        assert result == out
        assert out.exists()


class TestPlotGeqdsk:
    def test_creates_file(self, minimal_geqdsk, tmp_path):
        out = tmp_path / "geqdsk.png"
        result = m3dc1_plots.plot_geqdsk(minimal_geqdsk, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 100

    def test_returns_path_object(self, minimal_geqdsk, tmp_path):
        out = tmp_path / "geqdsk.png"
        result = m3dc1_plots.plot_geqdsk(minimal_geqdsk, out)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# Integration tests — require real data + m3dc1/fpy
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
class TestIntegrationCategoryA:
    def test_plot_kinetic_energy(self, tmp_path):
        out = tmp_path / "ke.png"
        m3dc1_plots.plot_kinetic_energy(CASE_DIR, out)
        assert out.exists()

    def test_plot_growth_rate_vs_time(self, tmp_path):
        out = tmp_path / "gr.png"
        m3dc1_plots.plot_growth_rate_vs_time(CASE_DIR, out)
        assert out.exists()

    def test_plot_flux_average_profiles(self, tmp_path):
        out = tmp_path / "profiles.png"
        m3dc1_plots.plot_flux_average_profiles(CASE_DIR, out)
        assert out.exists()

    def test_plot_safety_factor(self, tmp_path):
        out = tmp_path / "q.png"
        m3dc1_plots.plot_safety_factor(CASE_DIR, out)
        assert out.exists()

    def test_plot_poloidal_spectrum(self, tmp_path):
        out = tmp_path / "spec.png"
        m3dc1_plots.plot_poloidal_spectrum(CASE_DIR, 1, "p", out)
        assert out.exists()

    def test_plot_standard_spectra(self, tmp_path):
        out = tmp_path / "spectra.png"
        m3dc1_plots.plot_standard_spectra(CASE_DIR, 1, out)
        assert out.exists()

    def test_plot_perturbed_field_map(self, tmp_path):
        out = tmp_path / "field_psi.png"
        m3dc1_plots.plot_perturbed_field_map(CASE_DIR, 1, "psi", out)
        assert out.exists()

    def test_plot_geqdsk_uses_case_dir_gfile(self, tmp_path):
        out = tmp_path / "geqdsk.png"
        gfile = CASE_DIR / "geqdsk"
        if not gfile.exists():
            pytest.skip("geqdsk file not present in case dir")
        m3dc1_plots.plot_geqdsk(gfile, out)
        assert out.exists()


@pytest.mark.integration
class TestIntegrationWrappers:
    def test_plot_stability_summary(self, tmp_path):
        paths = m3dc1_plots.plot_stability_summary(CASE_DIR, 1, tmp_path)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()

    def test_plot_equilibrium_overview(self, tmp_path):
        paths = m3dc1_plots.plot_equilibrium_overview(CASE_DIR, tmp_path)
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()

    def test_plot_case_summary_returns_combined(self, tmp_path):
        paths = m3dc1_plots.plot_case_summary(CASE_DIR, 1, tmp_path)
        assert len(paths) >= 2

    def test_prefix_applied(self, tmp_path):
        paths = m3dc1_plots.plot_stability_summary(
            CASE_DIR, 1, tmp_path, prefix="run1_"
        )
        for p in paths:
            assert p.name.startswith("run1_")
