#!/usr/bin/env python3
"""Convert a VASP bulk single-point (DOS) calculation to an ISAAC AI-ready record (v1.05).

Designed for the ternary_oxides_bulk_DFT dataset:
  /global/cfs/cdirs/m5268/ruchika/ternary_oxides_bulk_DFT/

Expected layout:
  <chem_system>_<formula>_<mp_id>/
      dos/
          vasprun.xml  INCAR  KPOINTS  POSCAR  CONTCAR  OUTCAR  ...

Usage:
    python vasp_bulk_to_isaac.py <material_dir> [output.json]
"""

import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Incar, Kpoints, Vasprun
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


def _parse_dir_name(name: str) -> dict:
    """Extract chemical_system, formula, mp_id from e.g. 'Ag-Al-O_AlAgO2_mp-9631'."""
    parts = name.split("_")
    mp_id = None
    if parts and re.match(r"mp-\d+$", parts[-1]):
        mp_id = parts[-1]
        parts = parts[:-1]
    formula = parts[-1] if len(parts) >= 2 else name
    chem_system = parts[0] if parts else ""
    return {"chemical_system": chem_system, "formula": formula, "mp_id": mp_id}


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
def convert(material_dir: str | Path, output: str | Path | None = None) -> dict:
    material_dir = Path(material_dir).resolve()

    # Auto-detect dos subdir
    if (material_dir / "dos" / "vasprun.xml").exists():
        vasp_dir = material_dir / "dos"
    elif (material_dir / "vasprun.xml").exists():
        vasp_dir = material_dir
        material_dir = material_dir.parent
    else:
        raise FileNotFoundError(
            f"No vasprun.xml found in {material_dir} or {material_dir}/dos"
        )

    meta = _parse_dir_name(material_dir.name)

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

    # --- stress (kBar → eV/Å³) ---
    stress_kbar = last.get("stress")
    stress_eV_A3 = None
    if stress_kbar is not None:
        stress_eV_A3 = (np.array(stress_kbar) * 0.006242).tolist()

    # --- magnetic moment ---
    mag_total = None
    if incar.get("ISPIN") == 2:
        try:
            outcar = Outcar(str(vasp_dir / "OUTCAR"))
            if outcar.total_mag is not None:
                mag_total = round(float(outcar.total_mag), 6)
        except Exception:
            pass

    # --- DFT params ---
    encut  = float(incar.get("ENCUT", 400))
    ispin  = int(incar.get("ISPIN", 1))
    ismear = int(incar.get("ISMEAR", 0))
    sigma  = float(incar.get("SIGMA", 0.05))
    ediff  = float(incar.get("EDIFF", 1e-4))
    nelm   = int(incar.get("NELM", 60))
    nsw    = int(incar.get("NSW", 0))
    nedos  = int(incar.get("NEDOS", 301))
    lorbit = int(incar.get("LORBIT", 0))
    ldau   = bool(incar.get("LDAU", False))
    ldaul  = incar.get("LDAUL", [])
    ldauu  = incar.get("LDAUU", [])
    ldauj  = incar.get("LDAUJ", [])

    # PBE+U only if LDAU flag is set AND at least one U > 0
    effective_u = ldau and any(float(u) != 0 for u in ldauu)
    functional = "PBE+U" if effective_u else "PBE"

    kgrid  = list(kpts.kpts[0]) if kpts.kpts else None
    kgamma = kpts.style.name.lower() == "gamma"

    hub_u = None
    if ldau:
        hub_u = {}
        for i, el in enumerate(comp.elements):
            hub_u[el.symbol] = {
                "L":    int(ldaul[i])    if i < len(ldaul) else -1,
                "U_eV": float(ldauu[i]) if i < len(ldauu) else 0.0,
                "J_eV": float(ldauj[i]) if i < len(ldauj) else 0.0,
            }

    # Space group (best-effort)
    try:
        sg_symbol, sg_number = struct.get_space_group_info()
    except Exception:
        sg_symbol, sg_number = None, None

    # Per-atom forces (single step)
    per_atom_forces = [
        {
            "atom_index":     i,
            "element":        str(struct[i].specie.symbol),
            "fx_eV_A":        round(float(forces_arr[i, 0]), 8),
            "fy_eV_A":        round(float(forces_arr[i, 1]), 8),
            "fz_eV_A":        round(float(forces_arr[i, 2]), 8),
            "magnitude_eV_A": round(float(np.linalg.norm(forces_arr[i])), 8),
        }
        for i in range(len(struct))
    ]

    # Assets
    asset_map = [
        ("vasprun.xml", "primary_output"),
        ("CONTCAR",     "final_structure"),
        ("OUTCAR",      "detailed_output"),
        ("INCAR",       "input_parameters"),
        ("POSCAR",      "initial_structure"),
        ("KPOINTS",     "kpoint_mesh"),
        ("POTCAR",      "pseudopotential"),
        ("DOSCAR",      "dos_output"),
        ("CHGCAR",      "charge_density"),
    ]
    assets = []
    for fname, role in asset_map:
        fpath = vasp_dir / fname
        if fpath.exists():
            assets.append({
                "name":       fname,
                "role":       role,
                "path":       str(fpath),
                "sha256":     _sha256(fpath),
                "size_bytes": fpath.stat().st_size,
            })

    vasp_ver = _vasp_version(vasp_dir)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -----------------------------------------------------------------------
    # ISAAC record
    # -----------------------------------------------------------------------
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
            "material":          comp.reduced_formula,
            "reduced_formula":   comp.reduced_formula,
            "chemical_system":   meta["chemical_system"],
            "composition":       {el.symbol: int(amt) for el, amt in comp.items()},
            "num_atoms":         struct.num_sites,
            "geometry":          "bulk",
            "source_database":   "Materials Project",
            "source_id":         meta["mp_id"],
            "space_group_symbol": sg_symbol,
            "space_group_number": sg_number,
            "periodic_boundary_conditions": [True, True, True],
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
            "code":             "VASP",
            "code_version":     vasp_ver,
            "functional":       functional,
            "pseudopotential":  "PAW-PBE",
            "basis_set":        "plane_wave",
            "cutoff_energy_eV": encut,
            "spin_polarized":   ispin == 2,
            "smearing": {
                "method":   "Gaussian" if ismear == 0 else f"ISMEAR={ismear}",
                "sigma_eV": sigma,
            },
            "kpoint_sampling": {
                "scheme":         "Gamma",
                "mesh":           kgrid,
                "gamma_centered": kgamma,
            },
            "hubbard_u":            ldau,
            "hubbard_u_parameters": hub_u,
            "calculation_type": "single_point",
            "dos_settings": {
                "nedos":          nedos,
                "lorbit":         lorbit,
                "projected_dos":  lorbit >= 10,
            },
            "bulk_model": {
                "lattice_a_angstrom": round(float(lat.a), 6),
                "lattice_b_angstrom": round(float(lat.b), 6),
                "lattice_c_angstrom": round(float(lat.c), 6),
                "alpha_deg":          round(float(lat.alpha), 4),
                "beta_deg":           round(float(lat.beta),  4),
                "gamma_deg":          round(float(lat.gamma), 4),
                "volume_angstrom3":   round(float(lat.volume), 4),
            },
            "electronic_convergence": {
                "ediff_eV":  ediff,
                "nelm":      nelm,
                "nsw":       nsw,
                "converged": vrun.converged,
            },
            "output_quantities": [
                "total_energy", "free_energy", "forces", "stress",
                "magnetic_moments", "dos", "charge_density",
            ],
        },

        # ------ MEASUREMENT ------
        "measurement": {
            "qc_status": "pass" if vrun.converged else "fail",
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
                "source":      "VASP/vasprun.xml",
                "value":       round(final_energy, 8),
                "units":       "eV",
                "uncertainty": None,
            },
            {
                "name":        "total_energy_per_atom",
                "kind":        "DFT_total_energy_per_atom",
                "source":      "VASP/vasprun.xml",
                "value":       round(final_energy / struct.num_sites, 8),
                "units":       "eV/atom",
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
                "source":      "VASP/OUTCAR",
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
                "name":        "converged",
                "kind":        "electronic_convergence_flag",
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
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <material_dir> [output.json]")
        sys.exit(1)

    material_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    rec = convert(material_dir, out_path)

    if not out_path:
        print(json.dumps(rec, indent=2))
