#!/usr/bin/env python3
"""Build an ASE SQLite database from the ternary_oxides_bulk_DFT dataset.

Usage:
    python make_ase_db.py <output.db> <material_dir> [material_dir ...]
    python make_ase_db.py ternary_oxides.db /path/to/ternary_oxides_bulk_DFT/*/
"""

import re
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Incar, Kpoints, Vasprun
from pymatgen.io.vasp.outputs import Outcar


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
    parts = name.split("_")
    mp_id = None
    if parts and re.match(r"mp-\d+$", parts[-1]):
        mp_id = parts[-1]
        parts = parts[:-1]
    formula = parts[-1] if len(parts) >= 2 else name
    chem_system = parts[0] if parts else ""
    return {"chemical_system": chem_system, "formula": formula, "mp_id": mp_id}


def add_to_db(db, material_dir: str | Path) -> str:
    material_dir = Path(material_dir).resolve()

    if (material_dir / "dos" / "vasprun.xml").exists():
        vasp_dir = material_dir / "dos"
    elif (material_dir / "vasprun.xml").exists():
        vasp_dir = material_dir
        material_dir = material_dir.parent
    else:
        raise FileNotFoundError(f"No vasprun.xml in {material_dir} or {material_dir}/dos")

    meta = _parse_dir_name(material_dir.name)

    print(f"  {material_dir.name} ...", flush=True)
    vrun = Vasprun(str(vasp_dir / "vasprun.xml"), parse_dos=False, parse_eigen=False)
    incar = Incar.from_file(str(vasp_dir / "INCAR"))
    kpts = Kpoints.from_file(str(vasp_dir / "KPOINTS"))

    struct = vrun.final_structure
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # Energy and forces
    final_energy = float(vrun.final_energy)
    last = vrun.ionic_steps[-1]
    forces = np.array(last["forces"])

    # Stress: 3×3 kBar → Voigt 6-component eV/Å³
    # ASE convention: [xx, yy, zz, yz, xz, xy]
    s = np.array(last["stress"]) * 0.006242
    stress_voigt = np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])

    # Magnetic moment
    mag_total = None
    if incar.get("ISPIN") == 2:
        try:
            outcar = Outcar(str(vasp_dir / "OUTCAR"))
            if outcar.total_mag is not None:
                mag_total = float(outcar.total_mag)
        except Exception:
            pass

    calc = SinglePointCalculator(
        atoms,
        energy=final_energy,
        forces=forces,
        stress=stress_voigt,
    )
    atoms.calc = calc

    # Functional
    ldau = bool(incar.get("LDAU", False))
    ldauu = incar.get("LDAUU", [])
    effective_u = ldau and any(float(u) != 0 for u in ldauu)
    functional = "PBE+U" if effective_u else "PBE"

    kgrid = list(kpts.kpts[0]) if kpts.kpts else None

    # Space group
    try:
        sg_symbol, sg_number = struct.get_space_group_info()
    except Exception:
        sg_symbol, sg_number = "", 0

    kvp = dict(
        mp_id=meta["mp_id"] or "",
        chemical_system=meta["chemical_system"],
        reduced_formula=struct.composition.reduced_formula,
        geometry="bulk",
        source_database="Materials Project",
        functional=functional,
        encut=float(incar.get("ENCUT", 400)),
        kpoints=str(kgrid),
        spin_polarized=incar.get("ISPIN") == 2,
        hubbard_u=effective_u,
        vasp_version=_vasp_version(vasp_dir),
        space_group_symbol=sg_symbol,
        space_group_number=sg_number,
        converged=vrun.converged,
        total_energy_per_atom=round(final_energy / struct.num_sites, 8),
    )
    if mag_total is not None:
        kvp["total_magnetic_moment"] = round(mag_total, 6)

    db.write(atoms, **kvp)
    return meta["mp_id"] or material_dir.name


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <output.db> <material_dir> [material_dir ...]")
        sys.exit(1)

    db_path = sys.argv[1]
    material_dirs = sys.argv[2:]

    with connect(db_path) as db:
        ok, failed = [], []
        for d in material_dirs:
            try:
                mp_id = add_to_db(db, d)
                ok.append(mp_id)
            except Exception as e:
                print(f"  FAILED {Path(d).name}: {e}")
                failed.append(Path(d).name)

    print(f"\nDone: {len(ok)} written, {len(failed)} failed → {db_path}")
    if failed:
        print("Failed:", failed)
