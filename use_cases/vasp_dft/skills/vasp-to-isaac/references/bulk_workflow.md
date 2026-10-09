# Bulk Workflow — VASP to ISAAC

**Script**: `vasp_bulk_to_isaac.py` (in `scripts/` of this skill)  
**Use for**: Bulk crystal single-point calculations (DOS, static, band structure)

---

## Expected Directory Structure

```
<chem_system>_<formula>_<mp_id>/
  dos/
    vasprun.xml
    INCAR
    KPOINTS
    POSCAR
    CONTCAR
    OUTCAR
    POTCAR
    DOSCAR      (included in assets if present)
    CHGCAR      (included in assets if present)
    vasp.out
```

Example:
```
Ag-Cu-O_Cu2AgO4_mp-1225882/
  dos/
```

The script auto-detects the `dos/` subdirectory. You can also pass `dos/` directly.  
`mp_id`, `chemical_system`, and `formula` are parsed from the directory name pattern `{system}_{formula}_{mp-id}`.

---

## Usage

```bash
# Pass the material directory (auto-detects dos/ subdir)
python3 vasp_bulk_to_isaac.py <material_dir> [output.json]

# Example
python3 vasp_bulk_to_isaac.py Ag-Cu-O_Cu2AgO4_mp-1225882 out.json
```

---

## What Gets Extracted

| Field | Source |
|---|---|
| `sample.material` | Composition reduced formula |
| `sample.chemical_system` | First segment of dir name (e.g. `"Ag-Cu-O"`) |
| `sample.source_id` | mp-id from dir name (e.g. `"mp-1225882"`) |
| `sample.space_group_symbol/number` | pymatgen SpacegroupAnalyzer |
| `computation.functional` | PBE+U only if LDAU=True AND any LDAUU > 0 |
| `computation.hubbard_u_parameters` | Per-element L, U_eV, J_eV from INCAR |
| `computation.dos_settings` | NEDOS, LORBIT, projected_dos flag |
| `computation.calculation_type` | `"single_point"` (NSW=0) |
| `descriptors.total_energy_per_atom` | total_energy / num_atoms |
| `assets` | SHA256 + size for up to 9 files including DOSCAR, CHGCAR |

---

## PBE vs PBE+U Detection

`LDAU = .TRUE.` in INCAR does **not** mean PBE+U is active — the Hubbard correction is only applied when at least one element has `LDAUU > 0`. The script checks:

```python
effective_u = ldau and any(float(u) != 0 for u in ldauu)
functional = "PBE+U" if effective_u else "PBE"
```

Example: `Ag-Al-O_AlAgO2_mp-9631` has `LDAU=True` but all `LDAUU=0` → labelled `PBE`.  
Example: `Ag-Cu-O_Cu2AgO4_mp-1225882` has Cu with `LDAUU=3.0` → labelled `PBE+U`.

---

## Two-Step ASE DB Pipeline

For working from a pre-existing ASE database rather than raw VASP files:

**Step 1 — Build the ASE db** (`make_ase_db.py`):
```bash
python3 make_ase_db.py output.db <material_dir> [material_dir ...]
# or batch all:
python3 make_ase_db.py ternary_oxides.db /path/to/ternary_oxides_bulk_DFT/*/
```

Stores per-entry: Atoms object, energy/forces/stress (SinglePointCalculator), and key-value metadata (mp_id, functional, space_group, encut, kpoints, converged, etc.)

**Step 2 — Convert ASE db to ISAAC** (`ase_db_to_isaac.py`):
```bash
python3 ase_db_to_isaac.py ternary_oxides.db ./isaac_records/
```

Outputs one `{chemical_system}_{mp_id}_isaac.json` per entry.

---

## ASE Route vs Direct VASP Route — What's Lost

| Field | Direct VASP | ASE DB route |
|---|---|---|
| `smearing.method/sigma` | ✓ from INCAR | ✗ not stored in db |
| `dos_settings` (NEDOS, LORBIT) | ✓ | ✗ |
| `electronic_convergence` (EDIFF, NELM, NSW) | ✓ | ✗ |
| `hubbard_u_parameters` (per-element U/J/L) | ✓ | ✗ (only boolean flag) |
| `free_energy` descriptor | ✓ | ✗ (ASE stores total energy only) |
| `assets` (file paths + SHA256) | ✓ | ✗ |
| Structure, energy, forces, stress | ✓ | ✓ |
| Space group, lattice | ✓ | ✓ |
| mp_id, chemical_system, functional | ✓ | ✓ |

Use the direct VASP route when full INCAR detail matters. Use the ASE route when working from a pre-existing database.
