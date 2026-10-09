# Dependencies

## Python Environment

Python 3.10+ required. Create and activate a virtual environment in your project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pymatgen numpy ase cathub jsonschema
```

---

## Required Libraries

### pymatgen
Used for all VASP file parsing (vasprun.xml, INCAR, KPOINTS, OUTCAR, POTCAR).

```bash
pip install pymatgen
```

**Known quirk**: `UnknownPotcarWarning` for Ir, Te, Ag, Au PAW potentials — pymatgen's internal POTCAR database does not include these symbols. This warning is harmless and does not affect parsing.

Key classes used:
- `pymatgen.io.vasp.Vasprun` — parses vasprun.xml (energies, forces, stress, structure, ionic steps)
- `pymatgen.io.vasp.Incar` — parses INCAR (ENCUT, ISPIN, LDAU*, ISMEAR, EDIFF, NSW, etc.)
- `pymatgen.io.vasp.Kpoints` — parses KPOINTS (mesh, scheme)
- `pymatgen.io.vasp.outputs.Outcar` — parses OUTCAR for total magnetic moment
- `pymatgen.io.ase.AseAtomsAdaptor` — converts pymatgen Structure → ASE Atoms

### numpy
Used for force/stress array operations.

```bash
pip install numpy
```

### ASE (Atomic Simulation Environment)
Required for `make_ase_db.py` and `ase_db_to_isaac.py`.

```bash
pip install ase
```

Minimum version: 3.28.0

Key classes used:
- `ase.db.connect` — SQLite database interface
- `ase.calculators.singlepoint.SinglePointCalculator` — attaches energy/forces/stress to Atoms
- `ase.geometry.cell_to_cellpar` — converts cell matrix to a, b, c, α, β, γ

### cathub
Required for CatHub database workflows (optional for ISAAC conversion).

```bash
pip install cathub
```

Note: `cathub folder2db` is designed for surface reaction data (slabs + adsorbates + gas references). For bulk DOS data, use the ASE db pipeline instead.

---

## Skills

### pymatgen skill
Installed at: `~/.claude/skills/pymatgen/`  
Source repo: https://github.com/K-Dense-AI/scientific-agent-skills/tree/main/skills/pymatgen

Provides reference docs for pymatgen core classes, IO formats, analysis modules, Materials Project API, and transformation workflows.

---

## Schema Validation

```bash
pip install jsonschema
```

See `references/isaac_schema.md` → "How to Update This Skill" for validation commands.
