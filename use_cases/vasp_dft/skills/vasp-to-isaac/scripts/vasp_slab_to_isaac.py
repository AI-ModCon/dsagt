#!/usr/bin/env python3
"""Convert a VASP DFT calculation directory to an ISAAC AI-ready record (v1.05).

Usage:
    python vasp_to_isaac.py <vasp_dir> [output.json]
"""

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Vasprun
from pymatgen.io.vasp.outputs import Outcar


# ---------------------------------------------------------------------------
# ULID generator (26-char Crockford base32, timestamp-prefixed)
# ---------------------------------------------------------------------------
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    ts = int(time.time() * 1000)
    ts_part = ""
    for _ in range(10):
        ts_part = _B32[ts & 0x1F] + ts_part
        ts >>= 5
    return ts_part + "".join(random.choices(_B32, k=16))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# VASP version from vasp.out
# ---------------------------------------------------------------------------
def _vasp_version(vasp_dir: Path) -> str:
    for fname in ("vasp.out", "OUTCAR"):
        p = vasp_dir / fname
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                if "vasp." in line.lower():
                    tok = [t for t in line.split() if t.lower().startswith("vasp.")]
                    if tok:
                        return tok[0]
    return "unknown"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
_ISMEAR_MAP = {
    -5: "tetrahedron_blochl",
    -4: "tetrahedron",
    -1: "Fermi-Dirac",
     0: "Gaussian",
     1: "Methfessel-Paxton_1",
     2: "Methfessel-Paxton_2",
}


def convert(
    vasp_dir: str | Path,
    output: str | Path | None = None,
    electrode_type: str | None = None,
    surface_facet: str | None = None,
    surface_coverage: str | None = None,
) -> dict:
    vasp_dir = Path(vasp_dir)

    print(f"Parsing vasprun.xml ...", flush=True)
    vrun = Vasprun(
        str(vasp_dir / "vasprun.xml"),
        parse_dos=False,
        parse_eigen=False,
    )
    incar = Incar.from_file(str(vasp_dir / "INCAR"))
    kpts = Kpoints.from_file(str(vasp_dir / "KPOINTS"))

    struct = vrun.final_structure
    lat = struct.lattice
    comp = struct.composition

    # --- energies ---
    final_energy = float(vrun.final_energy)
    last = vrun.ionic_steps[-1]
    free_energy = float(last["e_fr_energy"])

    # --- forces ---
    forces_arr = np.array(last["forces"])
    max_force = float(np.max(np.linalg.norm(forces_arr, axis=1)))

    # --- stress (kBar → eV/Å³, factor 0.006242) ---
    stress_kbar = last.get("stress")  # 3×3 or None
    stress_eV_A3 = None
    if stress_kbar is not None:
        stress_eV_A3 = (np.array(stress_kbar) * 0.006242).tolist()

    # --- magnetic moment (from OUTCAR via pymatgen Outcar) ---
    mag_total = None
    if incar.get("ISPIN") == 2:
        try:
            outcar = Outcar(str(vasp_dir / "OUTCAR"))
            if outcar.total_mag is not None:
                mag_total = round(float(outcar.total_mag), 6)
        except Exception:
            pass

    # --- DFT params ---
    encut     = float(incar.get("ENCUT", 400))
    ispin     = int(incar.get("ISPIN", 1))
    ismear    = int(incar.get("ISMEAR", 0))
    smearing_method = _ISMEAR_MAP.get(ismear, f"ISMEAR={ismear}")
    sigma     = float(incar.get("SIGMA", 0.05))
    ibrion    = int(incar.get("IBRION", -1))
    isif      = int(incar.get("ISIF", 2))
    ediff     = float(incar.get("EDIFF", 1e-4))
    ediffg    = float(incar.get("EDIFFG", -0.03))
    nsw       = int(incar.get("NSW", 0))
    ldau      = bool(incar.get("LDAU", False))
    ldaul     = incar.get("LDAUL", [])
    ldauu     = incar.get("LDAUU", [])
    ldauj     = incar.get("LDAUJ", [])
    effective_u = ldau and any(float(u) != 0 for u in ldauu)
    functional = "PBE+U" if effective_u else "PBE"

    # k-point mesh
    kgrid = list(kpts.kpts[0]) if kpts.kpts else None
    kgamma = kpts.style.name.lower() == "gamma"

    # Hubbard-U per element
    hub_u = None
    if ldau:
        hub_u = {}
        for i, el in enumerate(comp.elements):
            hub_u[el.symbol] = {
                "L":   int(ldaul[i])   if i < len(ldaul) else -1,
                "U_eV": float(ldauu[i]) if i < len(ldauu) else 0.0,
                "J_eV": float(ldauj[i]) if i < len(ldauj) else 0.0,
            }

    # Vacuum thickness estimate (c minus slab z-span)
    z_coords = [s.coords[2] for s in struct.sites]
    slab_thickness = max(z_coords) - min(z_coords)
    vacuum = round(float(lat.c) - slab_thickness, 3)

    # Ionic relaxation trajectory (energy per step)
    trajectory = []
    for i, step in enumerate(vrun.ionic_steps):
        trajectory.append({
            "step":            i + 1,
            "e_fr_energy_eV":  round(float(step["e_fr_energy"]), 8),
            "e_0_energy_eV":   round(float(step["e_0_energy"]),  8),
        })

    # Per-atom forces (final step)
    per_atom_forces = [
        {
            "atom_index": i,
            "element":    str(struct[i].specie.symbol),
            "fx_eV_A":    round(float(forces_arr[i, 0]), 8),
            "fy_eV_A":    round(float(forces_arr[i, 1]), 8),
            "fz_eV_A":    round(float(forces_arr[i, 2]), 8),
            "magnitude_eV_A": round(float(np.linalg.norm(forces_arr[i])), 8),
        }
        for i in range(len(struct))
    ]

    # Assets
    asset_map = [
        ("vasprun.xml",  "primary_output"),
        ("CONTCAR",      "final_structure"),
        ("OUTCAR",       "detailed_output"),
        ("INCAR",        "input_parameters"),
        ("POSCAR",       "initial_structure"),
        ("KPOINTS",      "kpoint_mesh"),
        ("POTCAR",       "pseudopotential"),
    ]
    assets = []
    for fname, role in asset_map:
        fpath = vasp_dir / fname
        if fpath.exists():
            assets.append({
                "name":   fname,
                "role":   role,
                "path":   str(fpath.resolve()),
                "sha256": _sha256(fpath),
                "size_bytes": fpath.stat().st_size,
            })

    vasp_ver = _vasp_version(vasp_dir)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -----------------------------------------------------------------------
    # ISAAC record
    # -----------------------------------------------------------------------
    record = {
        "isaac_record_version": "1.05",
        "record_id":   _ulid(),
        "record_type": "evidence",
        "record_domain": "simulation",
        "source_type": "computation",

        "timestamps": {
            "created_utc": now_utc,
        },

        # ------ SAMPLE ------
        "sample": {
            "material":         comp.reduced_formula,
            "reduced_formula":  comp.reduced_formula,
            "composition":      {el.symbol: int(amt) for el, amt in comp.items()},
            "num_atoms":        struct.num_sites,
            "electrode_type":   electrode_type,
            "geometry":         "slab",
            "surface_facet":    surface_facet if surface_facet is not None else vasp_dir.parent.name,
            "surface_coverage": surface_coverage if surface_coverage is not None else vasp_dir.name,
            "periodic_boundary_conditions": [True, True, True],
            "slab_thickness_angstrom": round(slab_thickness, 3),
            "vacuum_thickness_angstrom": vacuum,
        },

        # ------ SYSTEM ------
        "system": {
            "domain":       "simulation",
            "technique":    "DFT",
            "code":         "VASP",
            "code_version": vasp_ver,
        },

        # ------ COMPUTATION ------
        "computation": {
            "code":              "VASP",
            "code_version":      vasp_ver,
            "functional":        functional,
            "pseudopotential":   "PAW-PBE",
            "basis_set":         "plane_wave",
            "cutoff_energy_eV":  encut,
            "spin_polarized":    ispin == 2,
            "smearing": {
                "method":   smearing_method,
                "sigma_eV": sigma,
            },
            "kpoint_sampling": {
                "scheme":         "Gamma",
                "mesh":           kgrid,
                "gamma_centered": kgamma,
            },
            "hubbard_u":            ldau,
            "hubbard_u_parameters": hub_u,
            "slab_model": {
                "lattice_a_angstrom": round(float(lat.a), 6),
                "lattice_b_angstrom": round(float(lat.b), 6),
                "lattice_c_angstrom": round(float(lat.c), 6),
                "alpha_deg":          round(float(lat.alpha), 4),
                "beta_deg":           round(float(lat.beta),  4),
                "gamma_deg":          round(float(lat.gamma), 4),
                "volume_angstrom3":   round(float(lat.volume), 4),
            },
            "ionic_relaxation": {
                "algorithm":                        "CG" if ibrion == 2 else f"IBRION={ibrion}",
                "isif":                             isif,
                "cell_shape_relaxed":               isif >= 4,
                "cell_volume_relaxed":              isif >= 6,
                "max_ionic_steps":                  nsw,
                "ionic_steps_completed":            len(vrun.ionic_steps),
                "force_convergence_eV_per_angstrom": abs(ediffg),
                "electronic_convergence_eV":         ediff,
                "converged":                         vrun.converged,
            },
            "output_quantities": [
                "total_energy", "free_energy", "forces", "stress", "magnetic_moments"
            ],
        },

        # ------ MEASUREMENT ------
        "measurement": {
            "qc_status": "pass" if vrun.converged else "fail",
            "channels": [
                {
                    "name":   "ionic_relaxation_trajectory",
                    "data":   trajectory,
                    "x_axis": "ionic_step",
                    "y_axis": "e_fr_energy_eV",
                },
                {
                    "name": "final_atomic_forces",
                    "data": per_atom_forces,
                },
            ],
        },

        # ------ DESCRIPTORS ------
        "descriptors": [
            {
                "name":        "total_energy",
                "kind":        "DFT_total_energy",
                "source":      "VASP/vasprun.xml",
                "value":       round(final_energy, 8),
                "units":       "eV",
                "uncertainty": None,
            },
            {
                "name":        "free_energy",
                "kind":        "DFT_free_energy",
                "source":      "VASP/vasprun.xml",
                "value":       round(free_energy, 8),
                "units":       "eV",
                "uncertainty": None,
            },
            {
                "name":        "max_force",
                "kind":        "maximum_atomic_force",
                "source":      "VASP/vasprun.xml",
                "value":       round(max_force, 6),
                "units":       "eV/angstrom",
                "uncertainty": None,
            },
            {
                "name":        "total_magnetic_moment",
                "kind":        "spin_magnetic_moment",
                "source":      "VASP/vasprun.xml",
                "value":       mag_total,
                "units":       "mu_B",
                "uncertainty": None,
            },
            {
                "name":        "stress_tensor_eV_A3",
                "kind":        "stress_tensor",
                "source":      "VASP/vasprun.xml",
                "value":       stress_eV_A3,
                "units":       "eV/angstrom^3",
                "uncertainty": None,
            },
            {
                "name":        "num_ionic_steps",
                "kind":        "relaxation_step_count",
                "source":      "VASP/vasprun.xml",
                "value":       len(vrun.ionic_steps),
                "units":       None,
                "uncertainty": None,
            },
            {
                "name":        "converged",
                "kind":        "ionic_convergence_flag",
                "source":      "VASP/vasprun.xml",
                "value":       vrun.converged,
                "units":       None,
                "uncertainty": None,
            },
        ],

        # ------ ASSETS ------
        "assets": assets,

        # ------ LINKS ------
        "links": [],
    }

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Written → {out}")

    return record


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert VASP slab calculation to ISAAC v1.05 JSON")
    p.add_argument("vasp_dir",         help="Path to VASP calculation directory")
    p.add_argument("output",           nargs="?", default=None, help="Output JSON path (default: stdout)")
    p.add_argument("--electrode-type", default=None, help="e.g. anode, cathode (default: null)")
    p.add_argument("--surface-facet",  default=None, help="Miller index string, e.g. '101' (default: parent dir name)")
    p.add_argument("--surface-coverage", default=None, help="Coverage label, e.g. '4O' (default: dir name)")
    args = p.parse_args()

    rec = convert(
        args.vasp_dir,
        args.output,
        electrode_type=args.electrode_type,
        surface_facet=args.surface_facet,
        surface_coverage=args.surface_coverage,
    )

    if not args.output:
        print(json.dumps(rec, indent=2))
