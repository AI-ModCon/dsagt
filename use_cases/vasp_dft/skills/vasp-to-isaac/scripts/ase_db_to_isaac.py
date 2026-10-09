#!/usr/bin/env python3
"""Convert an ASE SQLite database to ISAAC AI-ready records (v1.05).

Reads bulk DFT entries from an ASE .db file and outputs one ISAAC JSON per entry.
Complement to make_ase_db.py — expects the key-value pairs written by that script.

Usage:
    python ase_db_to_isaac.py <input.db> <output_dir>
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from ase.db import connect


# ---------------------------------------------------------------------------
# ULID generator
# ---------------------------------------------------------------------------
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    ts = int(time.time() * 1000)
    ts_part = ""
    for _ in range(10):
        ts_part = _B32[ts & 0x1F] + ts_part
        ts >>= 5
    return ts_part + "".join(random.choices(_B32, k=16))


def _voigt_to_3x3(voigt: np.ndarray) -> list:
    """Convert ASE Voigt stress [xx,yy,zz,yz,xz,xy] → symmetric 3×3 list."""
    xx, yy, zz, yz, xz, xy = voigt
    return [
        [float(xx), float(xy), float(xz)],
        [float(xy), float(yy), float(yz)],
        [float(xz), float(yz), float(zz)],
    ]


def row_to_isaac(row) -> dict:
    from datetime import datetime, timezone

    atoms = row.toatoms()
    kvp = row.key_value_pairs
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- composition ---
    symbols = atoms.get_chemical_symbols()
    composition = {}
    for s in symbols:
        composition[s] = composition.get(s, 0) + 1

    # --- energy ---
    energy = float(row.energy)
    natoms = row.natoms
    energy_per_atom = round(energy / natoms, 8)

    # --- forces ---
    forces_arr = row.forces  # (N, 3) eV/Å
    max_force = float(np.max(np.linalg.norm(forces_arr, axis=1)))
    per_atom_forces = [
        {
            "atom_index":     i,
            "element":        symbols[i],
            "fx_eV_A":        round(float(forces_arr[i, 0]), 8),
            "fy_eV_A":        round(float(forces_arr[i, 1]), 8),
            "fz_eV_A":        round(float(forces_arr[i, 2]), 8),
            "magnitude_eV_A": round(float(np.linalg.norm(forces_arr[i])), 8),
        }
        for i in range(natoms)
    ]

    # --- stress ---
    stress_3x3 = None
    if row.stress is not None:
        stress_3x3 = _voigt_to_3x3(np.array(row.stress))

    # --- lattice ---
    cell = atoms.get_cell()
    from ase.geometry import cell_to_cellpar
    a, b, c, alpha, beta, gamma = cell_to_cellpar(cell)
    volume = float(atoms.get_volume())

    # --- metadata from key-value pairs ---
    mp_id           = kvp.get("mp_id", "")
    chemical_system = kvp.get("chemical_system", "")
    reduced_formula = kvp.get("reduced_formula", row.formula)
    functional      = kvp.get("functional", "PBE")
    encut           = float(kvp.get("encut", 0))
    kpoints_str     = kvp.get("kpoints", "")
    spin_polarized  = bool(kvp.get("spin_polarized", False))
    hubbard_u       = bool(kvp.get("hubbard_u", False))
    vasp_version    = kvp.get("vasp_version", "unknown")
    sg_symbol       = kvp.get("space_group_symbol", None)
    sg_number       = kvp.get("space_group_number", None)
    converged       = bool(kvp.get("converged", True))
    mag_total       = kvp.get("total_magnetic_moment", None)
    source_database = kvp.get("source_database", "Materials Project")

    # Parse kpoints string back to list if possible
    try:
        kgrid = [int(x) for x in kpoints_str.strip("[]").split(",")]
    except Exception:
        kgrid = None

    record = {
        "isaac_record_version": "1.05",
        "record_id":     _ulid(),
        "record_type":   "evidence",
        "record_domain": "simulation",
        "source_type":   "computation",

        "timestamps": {
            "created_utc": now_utc,
        },

        # ------ SAMPLE ------
        "sample": {
            "material":          reduced_formula,
            "reduced_formula":   reduced_formula,
            "chemical_system":   chemical_system,
            "composition":       composition,
            "num_atoms":         natoms,
            "geometry":          kvp.get("geometry", "bulk"),
            "source_database":   source_database,
            "source_id":         mp_id,
            "space_group_symbol": sg_symbol,
            "space_group_number": sg_number,
            "periodic_boundary_conditions": [True, True, True],
        },

        # ------ SYSTEM ------
        "system": {
            "domain":       "simulation",
            "technique":    "DFT",
            "code":         "VASP",
            "code_version": vasp_version,
        },

        # ------ COMPUTATION ------
        "computation": {
            "code":             "VASP",
            "code_version":     vasp_version,
            "functional":       functional,
            "pseudopotential":  "PAW-PBE",
            "basis_set":        "plane_wave",
            "cutoff_energy_eV": encut,
            "spin_polarized":   spin_polarized,
            "kpoint_sampling": {
                "scheme":         "Gamma",
                "mesh":           kgrid,
                "gamma_centered": True,
            },
            "hubbard_u": hubbard_u,
            "calculation_type": "single_point",
            "bulk_model": {
                "lattice_a_angstrom": round(float(a), 6),
                "lattice_b_angstrom": round(float(b), 6),
                "lattice_c_angstrom": round(float(c), 6),
                "alpha_deg":          round(float(alpha), 4),
                "beta_deg":           round(float(beta),  4),
                "gamma_deg":          round(float(gamma), 4),
                "volume_angstrom3":   round(volume, 4),
            },
            "electronic_convergence": {
                "converged": converged,
            },
            "output_quantities": [
                "total_energy", "free_energy", "forces", "stress",
                "magnetic_moments", "dos", "charge_density",
            ],
        },

        # ------ MEASUREMENT ------
        "measurement": {
            "qc_status": "pass" if converged else "fail",
            "channels": [
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
                "source":      "ASE_db",
                "value":       round(energy, 8),
                "units":       "eV",
                "uncertainty": None,
            },
            {
                "name":        "total_energy_per_atom",
                "kind":        "DFT_total_energy_per_atom",
                "source":      "ASE_db",
                "value":       energy_per_atom,
                "units":       "eV/atom",
                "uncertainty": None,
            },
            {
                "name":        "max_force",
                "kind":        "maximum_atomic_force",
                "source":      "ASE_db",
                "value":       round(max_force, 6),
                "units":       "eV/angstrom",
                "uncertainty": None,
            },
            {
                "name":        "total_magnetic_moment",
                "kind":        "spin_magnetic_moment",
                "source":      "ASE_db",
                "value":       round(float(mag_total), 6) if mag_total is not None else None,
                "units":       "mu_B",
                "uncertainty": None,
            },
            {
                "name":        "stress_tensor_eV_A3",
                "kind":        "stress_tensor",
                "source":      "ASE_db",
                "value":       stress_3x3,
                "units":       "eV/angstrom^3",
                "uncertainty": None,
            },
            {
                "name":        "converged",
                "kind":        "electronic_convergence_flag",
                "source":      "ASE_db",
                "value":       converged,
                "units":       None,
                "uncertainty": None,
            },
        ],

        # ------ ASSETS ------
        "assets": [],

        # ------ LINKS ------
        "links": [],
    }

    return record


def convert_db(db_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with connect(db_path) as db:
        total = db.count()
        print(f"Found {total} entries in {db_path}")

        ok, failed = [], []
        for row in db.select():
            label = row.key_value_pairs.get("mp_id") or row.key_value_pairs.get("reduced_formula") or str(row.id)
            try:
                record = row_to_isaac(row)
                fname = f"{row.key_value_pairs.get('chemical_system', 'unknown')}_{label}_isaac.json"
                fpath = out / fname
                with open(fpath, "w") as f:
                    json.dump(record, f, indent=2)
                print(f"  {label} → {fname}")
                ok.append(label)
            except Exception as e:
                print(f"  FAILED {label}: {e}")
                failed.append(label)

    print(f"\nDone: {len(ok)} written, {len(failed)} failed → {output_dir}")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input.db> <output_dir>")
        sys.exit(1)

    convert_db(sys.argv[1], sys.argv[2])
