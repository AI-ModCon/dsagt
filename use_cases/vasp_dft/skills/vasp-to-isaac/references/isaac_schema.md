# ISAAC AI-Ready Record Schema — v1.05

**Repo**: https://github.com/ISAAC-DOE/isaac-ai-ready-record  
**Schema file**: `schema/isaac_record_v1.json` (JSON Schema draft 2020-12)  
**Normative reference**: GitHub Wiki at `docs/wiki/`

---

## How to Read the Docs

The authoritative documentation lives in the GitHub repo. Navigate it as follows:

| What you need | Where to look |
|---|---|
| Full JSON Schema | `schema/isaac_record_v1.json` |
| DFT method specification | `docs/wiki/Computation-DFT-Method-Specification.md` |
| Intent record guide | `docs/wiki/Record-Type-Intent.md` |
| Working examples | `examples/` directory |
| Validation tooling | `tools/` directory |

**To browse examples**:
```bash
curl -s https://api.github.com/repos/ISAAC-DOE/isaac-ai-ready-record/contents/examples \
  | python3 -c "import json,sys; [print(f['name']) for f in json.load(sys.stdin)]"
```

**To fetch and read a specific example**:
```bash
curl -s https://raw.githubusercontent.com/ISAAC-DOE/isaac-ai-ready-record/main/examples/<filename>.json | python3 -m json.tool | less
```

**To validate a record against the schema**:
```bash
pip install jsonschema
curl -s https://raw.githubusercontent.com/ISAAC-DOE/isaac-ai-ready-record/main/schema/isaac_record_v1.json -o /tmp/isaac_schema.json
python3 -c "
import json, jsonschema
schema = json.load(open('/tmp/isaac_schema.json'))
record = json.load(open('your_record.json'))
jsonschema.validate(record, schema)
print('Valid!')
"
```

---

## Required Top-Level Fields

| Field | Type | Allowed values |
|---|---|---|
| `isaac_record_version` | string | `"1.05"` |
| `record_id` | string | 26-char ULID |
| `record_type` | enum | `evidence`, `intent`, `synthesis` |
| `record_domain` | enum | `characterization`, `performance`, `simulation`, `theory`, `derived` |
| `source_type` | enum | `facility`, `laboratory`, `computation`, `literature`, `database`, `industrial` |
| `timestamps.created_utc` | string | ISO 8601, e.g. `"2026-06-02T12:00:00Z"` |

**For DFT slab/bulk simulations use**: `record_type=evidence`, `record_domain=simulation`, `source_type=computation`

---

## 8 Structural Blocks

### 1. `sample`
Material identity, physical form, and geometry.

Key fields: `material`, `reduced_formula`, `composition`, `num_atoms`, `geometry` (`slab`, `bulk`, `molecule`, `nanoparticle`), `electrode_type` (`anode`, `cathode`, `reference`, `counter`), `surface_facet`, `periodic_boundary_conditions`

### 2. `system`
Experimental or computational domain.

Key fields: `domain` (`simulation`, `experiment`), `technique` (`DFT`, `XRD`, `cyclic_voltammetry`, …), `code`, `code_version`

### 3. `context`
Environmental and electrochemical conditions.

Key fields: `temperature_K`, `electrochemistry` (reaction type, cell config, control mode), `simulation_assumptions`

**Boundary rule**: If changing the parameter would change the *physical system*, it belongs in `context` not `computation`. Applied potential → `context`. K-point mesh → `computation`.

### 4. `measurement`
Data series with channels and QC status.

Key fields: `qc_status` (`pass`, `fail`, `unknown`), `channels[]` (each with `name`, `data`, `x_axis`, `y_axis`, `role`)

### 5. `computation`
DFT/MD method specification (see full detail below).

Key sub-blocks: `method`, `slab_model`, `potential_method`, `output_quantity`, `transition_state`

### 6. `descriptors`
AI-ready scalar outputs. Each entry:

```json
{
  "name":        "total_energy",
  "kind":        "DFT_total_energy",
  "source":      "VASP/vasprun.xml",
  "value":       -541.767,
  "units":       "eV",
  "uncertainty": null
}
```

### 7. `links`
Relationships between records: `derived_from`, `validates`, `contradicts`, `related_to`

### 8. `assets`
External file references with cryptographic validation:

```json
{
  "name":       "vasprun.xml",
  "role":       "primary_output",
  "path":       "/path/to/file",
  "sha256":     "abc123...",
  "size_bytes": 12345678
}
```

---

## `computation` Block — DFT Method Specification

### `method` sub-block

| Property | Type | Description |
|---|---|---|
| `family` | enum | `DFT`, `DFT_U`, `hybrid_DFT`, `AIMD`, `classical_MD`, `CHE`, `microkinetic`, `machine_learning` |
| `functional_class` | enum | `LDA`, `GGA`, `meta_GGA`, `hybrid`, `double_hybrid`, `RPA` |
| `functional_name` | string | `PBE`, `RPBE`, `BEEF-vdW`, `HSE06`, `SCAN`, etc. |
| `basis_type` | enum | `planewave`, `LCAO`, `real_space`, `mixed` |
| `pseudopotential` | string | `PAW`, `ultrasoft`, `norm_conserving` |
| `cutoff_eV` | number | Planewave kinetic energy cutoff |
| `spin_treatment` | enum | `none`, `collinear`, `noncollinear`, `SOC` |
| `dispersion` | enum | `none`, `D2`, `D3`, `D3BJ`, `TS`, `MBD` |
| `kpoints` | string | e.g. `"4x4x1 Gamma"`, `"Gamma-only"` |
| `smearing` | object | `method` + `width_eV` |
| `convergence` | object | `energy_eV`, `force_eV_per_A`, `stress_kbar` |

### `slab_model` sub-block (slab calculations only)

| Property | Description |
|---|---|
| `surface_facet` | `"101"`, `"110"`, `"111"` |
| `supercell` | `"4x4"`, `"3x3"` |
| `layers` | Number of atomic layers |
| `vacuum_A` | Vacuum thickness in Å |
| `dipole_correction` | boolean |

### `potential_method` sub-block (electrochemical simulations)

| Value | Description |
|---|---|
| `vacuum` | No solvent, no potential |
| `CHE` | Computational Hydrogen Electrode (post-processing) |
| `implicit_solvent_PZC` | Implicit solvent at PZC |
| `fixed_NELECT` | Fixed electron count at target potential |
| `grand_canonical` | NELECT optimized for grand canonical energy |
| `constant_potential` | Self-consistent constant-potential (SJM, FHI-aims) |

### `output_quantity` sub-block

| Value | Description |
|---|---|
| `E_DFT` | Raw DFT total energy |
| `E_DFT_plus_ZPE` | DFT + zero-point energy |
| `delta_E` | Energy difference (adsorption, barrier) |
| `delta_G_CHE` | Free energy with CHE correction |
| `delta_G_grand_canonical` | Grand canonical free energy |
| `activation_energy_raw` | NEB/dimer barrier, raw DFT |
| `activation_free_energy` | Full free energy barrier |

**Critical rule**: Two records can only be directly compared if they share the same `output_quantity.quantity` and `potential_method.type`.

---

## How to Update This Skill

When a new version of the ISAAC schema is released, update this reference file as follows:

**Step 1 — Check for a new version**:
```bash
curl -s https://raw.githubusercontent.com/ISAAC-DOE/isaac-ai-ready-record/main/schema/isaac_record_v1.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('Version:', d.get('title',''), d.get('description',''))"
```

**Step 2 — Pull the updated schema**:
```bash
curl -s https://raw.githubusercontent.com/ISAAC-DOE/isaac-ai-ready-record/main/schema/isaac_record_v1.json \
  -o /tmp/isaac_record_v1.json
```

**Step 3 — Pull updated wiki docs**:
```bash
curl -s https://raw.githubusercontent.com/ISAAC-DOE/isaac-ai-ready-record/main/docs/wiki/Computation-DFT-Method-Specification.md
curl -s https://raw.githubusercontent.com/ISAAC-DOE/isaac-ai-ready-record/main/docs/wiki/Record-Type-Intent.md
```

**Step 4 — Check for new wiki pages**:
```bash
curl -s https://api.github.com/repos/ISAAC-DOE/isaac-ai-ready-record/contents/docs/wiki \
  | python3 -c "import json,sys; [print(f['name']) for f in json.load(sys.stdin)]"
```

**Step 5 — Update this file** with any new blocks, changed vocabularies, or new sub-blocks, then update the version string at the top.

**Step 6 — Re-validate existing records** against the new schema:
```bash
pip install jsonschema
python3 -c "
import json, jsonschema, glob
schema = json.load(open('/tmp/isaac_record_v1.json'))
for path in glob.glob('**/*_isaac.json', recursive=True):
    try:
        jsonschema.validate(json.load(open(path)), schema)
        print(f'OK  {path}')
    except jsonschema.ValidationError as e:
        print(f'FAIL {path}: {e.message}')
"
```
