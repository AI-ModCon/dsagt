# Slab Workflow — VASP to ISAAC

> **Prerequisite**: If starting from raw VASP slab directories and needing a CatHub-organized ASE db, run `cathub organize` + `cathub folder2db` first. See `cathub_organize.md`.

**Script**: `vasp_slab_to_isaac.py` (in `scripts/` of this skill)  
**Use for**: Periodic surface slab calculations (IrOx OER, any facet/coverage)

---

## Expected Directory Structure

```
<facet>/
  <coverage>/
    vasprun.xml
    INCAR
    KPOINTS
    POSCAR
    CONTCAR
    OUTCAR
    POTCAR
    vasp.out        (optional, used for VASP version)
```

Example:
```
101/
  4O/      ← vasp_dir
  H_covered/
  Marzari/
  OH_O/
```

`surface_facet` defaults to `vasp_dir.parent.name` ("101")  
`surface_coverage` defaults to `vasp_dir.name` ("4O")

---

## Usage

```bash
# Minimal — facet/coverage derived from directory names
python3 vasp_slab_to_isaac.py <vasp_dir> [output.json]

# With explicit labels
python3 vasp_slab_to_isaac.py 101/4O out.json \
  --electrode-type anode \
  --surface-facet 101 \
  --surface-coverage 4O
```

**CLI arguments**:

| Argument | Default | Description |
|---|---|---|
| `vasp_dir` | required | Path to VASP calculation directory |
| `output` | stdout | Output JSON path |
| `--electrode-type` | `null` | `anode`, `cathode`, `reference`, `counter` |
| `--surface-facet` | parent dir name | Miller index string, e.g. `"101"` |
| `--surface-coverage` | dir name | Coverage label, e.g. `"4O"`, `"OH_O"` |

---

## What Gets Extracted

| Field | Source |
|---|---|
| `sample.material` | Composition reduced formula from vasprun.xml |
| `sample.surface_facet` | `--surface-facet` arg or parent dir name |
| `sample.surface_coverage` | `--surface-coverage` arg or dir name |
| `sample.slab_thickness_angstrom` | max(z) − min(z) of final structure |
| `sample.vacuum_thickness_angstrom` | lattice_c − slab_thickness |
| `computation.functional` | PBE+U if LDAU=True, else PBE |
| `computation.hubbard_u_parameters` | Per-element L, U_eV, J_eV from INCAR |
| `computation.ionic_relaxation` | IBRION, ISIF, NSW, EDIFFG, EDIFF |
| `computation.smearing` | ISMEAR mapped to name + SIGMA |
| `descriptors.total_energy` | vrun.final_energy |
| `descriptors.max_force` | max(|F|) over atoms, final ionic step |
| `descriptors.total_magnetic_moment` | Outcar.total_mag (only if ISPIN=2) |
| `descriptors.stress_tensor` | last ionic step stress, kBar → eV/Å³ |
| `measurement.channels` | Energy trajectory + per-atom forces |
| `assets` | SHA256 + size for 7 VASP files |

---

## Known Quirks

- **POTCAR warning**: `UnknownPotcarWarning` for Ir, Te, Ag — harmless, pymatgen's internal POTCAR database is incomplete for these PAW potentials
- **Magnetic moment**: Read from `Outcar.total_mag`, not from vasprun electronic steps (which don't store it as a simple float)
- **VASP version**: Parsed from `vasp.out` (second line) or `OUTCAR`; falls back to `"unknown"`
- **Smearing method map**: ISMEAR=0→Gaussian, 1→Methfessel-Paxton_1, 2→Methfessel-Paxton_2, -1→Fermi-Dirac, -4→tetrahedron, -5→tetrahedron_blochl
