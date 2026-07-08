# CatHub Organize — Slab Data Preparation

Prepares raw VASP slab calculations for ASE database generation using `cathub organize` and `cathub folder2db`. Run these steps before building the slab ASE db.

---

## Step 1 — Activate the CatHub Environment

```bash
source <project_dir>/.venv/bin/activate
```

Verify cathub is available:
```bash
cathub --version
```

---

## Step 2 — Build the Raw Input Folder

`cathub organize` expects a flat file layout grouped by facet. Gas references go in a shared `gas/` subfolder.

```bash
mkdir -p IrOx_raw/110 IrOx_raw/101 IrOx_raw/111 IrOx_raw/gas
```

Then copy your relaxed structure files (JSON, OUTCAR, or .traj — anything ASE can read with a total energy) into the appropriate subfolder:

| Subfolder | Contents |
|---|---|
| `IrOx_raw/110/` | Empty slab + every adsorbate slab for the 110 facet (e.g. `empty.json`, `4O.json`, `4OH.json`, `O4H12.json`, …). Also drop the bulk reference here — cathub auto-detects it by density. |
| `IrOx_raw/101/` | Same pattern for 101 facet |
| `IrOx_raw/111/` | Same pattern for 111 facet |
| `IrOx_raw/gas/` | Gas-phase references: `H2.json`, `H2O.json` |

**Important**: Do not create reaction subfolders manually — `cathub organize` builds those for you.

---

## Step 3 — Run `cathub organize`

Run once per facet. The output directory accumulates all facets across the loop.

```bash
OUT=IrOx_dataset_cathub_organized

for f in 110 101 111; do
  cathub organize "$f" \
    -c VASP-6.4.2 \
    -x PBE+U \
    -f "$f" \
    -S rutile \
    -a O4,O4H4,O4H8,O4H10,O4H12,O8,O8H8,O8H16,O8H4 \
    -d gas \
    -rtol 0.7 \
    -o "$OUT"
done
```

**Flag reference**:

| Flag | Value | Meaning |
|---|---|---|
| `"$f"` | `110` / `101` / `111` | Input subfolder for this facet |
| `-c` | `VASP-6.4.2` | DFT code and version |
| `-x` | `PBE+U` | Exchange-correlation functional |
| `-f` | `110` / `101` / `111` | Facet label written into the db |
| `-S` | `rutile` | Crystal structure prototype |
| `-a` | comma-separated list | Adsorbate species to match |
| `-d` | `gas` | Path to gas-phase reference files |
| `-rtol` | `0.7` | Reaction energy tolerance (eV) — filters unphysical energies |
| `-o` | `$OUT` | Output directory (created if absent) |

---

## Step 4 — Build the ASE Database

```bash
cathub folder2db "$OUT"
```

This scans `$OUT/` recursively, reads all structure files and energies, and writes a SQLite database at:

```
<OUT>/<OUT>.db
```

The `.db` file is a standard ASE database containing one row per structure, with key-value pairs for `name`, `state` (`star` or `gas`), `facet`, `species`, `n`, and `epot`.

---

## Next Steps

Once the `.db` file is built, convert to ISAAC records:

```bash
python3 ase_db_to_isaac.py <OUT>/<DBNAME>.db ./isaac_records/ --electrode-type anode
```

See `slab_workflow.md` for the full slab pipeline.

---

## Two Routes to an ASE Database

| Route | When to use |
|---|---|
| **cathub organize → folder2db** (this file) | Data is organized as surface reactions with gas references and adsorbate slabs. Produces CatHub-schema entries with reaction energies, `state`, `facet`, `species` key-value pairs. |
| **make_ase_db.py** (direct from VASP) | Raw VASP slab directories without a CatHub reaction structure. Useful for datasets not organized around specific reactions, or when you want richer INCAR-level metadata (smearing, convergence, Hubbard-U) in the db. |

Both routes produce an ASE SQLite `.db` file that `ase_db_to_isaac.py` can read directly.
