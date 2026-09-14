---
name: vasp-to-isaac
description: Convert VASP DFT calculations (slab or bulk) to ISAAC AI-ready records (v1.05). Handles IrOx surface slabs and ternary oxide bulk DOS calculations on NERSC Perlmutter.
license: internal
metadata:
  version: "1.0"
---

# VASP to ISAAC Converter

Converts VASP DFT calculation directories to ISAAC v1.05 AI-ready JSON records. Two scripts handle two distinct cases — choose based on the geometry of the calculation.

---

## Which Script to Use?

| Calculation type | Geometry | Script |
|---|---|---|
| Surface slab, ionic relaxation | slab | `vasp_slab_to_isaac.py` |
| Bulk crystal, single-point / DOS | bulk | `vasp_bulk_to_isaac.py` |

**Quick check**: look at the INCAR.
- `NSW > 0` and slab structure → use slab script
- `NSW = 0` and bulk crystal → use bulk script

---

## Slab Script

**Script**: `vasp_slab_to_isaac.py` (located in `scripts/` directory of this skill)  
**Reference**: `references/slab_workflow.md`

```bash
# Activate your project virtual environment
source .venv/bin/activate

# Minimal — facet/coverage inferred from directory names
python3 vasp_slab_to_isaac.py <vasp_dir> [output.json]

# With explicit labels
python3 vasp_slab_to_isaac.py <vasp_dir> output.json \
  --electrode-type anode \
  --surface-facet 101 \
  --surface-coverage 4O
```

Extracts: energy trajectory, per-atom forces, stress, magnetic moment, Hubbard-U params, slab thickness, vacuum, all VASP assets with SHA256.

---

## Bulk Script

**Script**: `vasp_bulk_to_isaac.py` (located in `scripts/` directory of this skill)  
**Reference**: `references/bulk_workflow.md`

```bash
source .venv/bin/activate

# Pass the material directory (auto-detects dos/ subdir)
python3 vasp_bulk_to_isaac.py <material_dir> [output.json]
```

Extracts: space group, mp_id, PBE vs PBE+U (from actual LDAUU values), DOS settings, energy/forces/stress, all VASP assets including DOSCAR and CHGCAR.

### ASE DB Pipeline (bulk only)

```bash
# Step 1: build ASE db from raw VASP dirs
python3 make_ase_db.py output.db <dataset_dir>/*/

# Step 2: convert ASE db to ISAAC records
python3 ase_db_to_isaac.py output.db ./isaac_records/
```

Note: the ASE route loses INCAR-level detail (smearing, EDIFF, per-element U params). See `references/bulk_workflow.md` for the full comparison.

---

## Batch Conversion

```bash
# Slab — all configs in a facet
for config in <config1> <config2>; do
  python3 vasp_slab_to_isaac.py <facet>/$config isaac_ready/<facet>_${config}_isaac.json \
    --electrode-type anode
done

# Bulk — all materials in a dataset directory
for d in <dataset_dir>/*/; do
  name=$(basename $d)
  python3 vasp_bulk_to_isaac.py $d isaac_ready/${name}_isaac.json
done
```

---

## References

- `references/isaac_schema.md` — ISAAC v1.05 schema blocks, required fields, DFT method spec, how to pull latest from GitHub and update this skill
- `references/cathub_organize.md` — **prerequisite for slab ASE db**: how to organize raw VASP slabs with `cathub organize` and `cathub folder2db`
- `references/slab_workflow.md` — slab directory structure, CLI args, extracted fields, known quirks
- `references/bulk_workflow.md` — bulk directory naming, mp_id parsing, PBE+U detection, ASE db pipeline, route comparison
- `references/dependencies.md` — pymatgen, numpy, ASE, cathub install commands; venv location; POTCAR quirk

## Scripts

Scripts are in the `scripts/` directory of this skill (symlinked to the working copies in your project):
- `vasp_slab_to_isaac.py` — slab converter
- `vasp_bulk_to_isaac.py` — bulk converter
