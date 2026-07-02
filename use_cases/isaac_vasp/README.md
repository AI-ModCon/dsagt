# DSAgt Demo: VASP → ISAAC AI-Ready Record

> **Estimated time:** ~15 minutes — the NEB fixture data is bundled in this
> folder, so the only real cost is a one-time `pip install pymatgen`. No DFT run,
> no HPC.

**Goal:** register a converter as a DSAgt **code**, then have the agent run it
through `dsagt-run` to turn a VASP nudged-elastic-band (NEB) calculation into an
[ISAAC AI-Ready Record](https://github.com/ISAAC-DOE/isaac-ai-ready-record) —
with the execution captured in `trace_archive/`.

- Converter: [`vasp_neb_to_isaac.py`](vasp_neb_to_isaac.py) parses the NEB
  `OUTCAR`s with `pymatgen.io.vasp` and emits a v1.05 record.
- Data: the `neb/00..04/` subdirs are a small NEB fixture (5 images) copied from
  the pymatgen test suite — bundled here so the demo runs without a DFT code.
- Schema: [isaac_record_v1.json](https://github.com/ISAAC-DOE/isaac-ai-ready-record/blob/main/schema/isaac_record_v1.json).

For the **skill-management** counterpart of this workflow (the agent discovers,
installs, and authors the converter as a skill on tiny mock data), see the
sister demo [`isaac_skills_demo`](../isaac_skills_demo/).

## Prerequisites

- DSAgt installed (`uv sync --all-groups`) and an agent platform installed and
  **already authenticated** (BYOA — dsagt writes no credentials; the default
  local embedder needs no API key).
- `pymatgen` importable in the environment `dsagt` runs in
  (`uv pip install pymatgen`) — the converter uses `pymatgen.io.vasp`.

## Setup

```bash
uv pip install pymatgen                 # the converter's one real dependency
dsagt init isaac-vasp --agent claude
PROJ=~/dsagt-projects/isaac-vasp
mkdir -p "$PROJ/codes/scripts"
cp use_cases/isaac_vasp/vasp_neb_to_isaac.py "$PROJ/codes/scripts/"
cp -r use_cases/isaac_vasp/neb "$PROJ/data_neb"
dsagt start isaac-vasp
```

## Execution

Paste each prompt into the agent, one at a time.

### 1. Register the converter as a code

```text
Register a code named vasp-neb-to-isaac. Its executable is
`python codes/scripts/vasp_neb_to_isaac.py`, which takes a positional NEB
directory argument (containing 00/, 01/, ... image subdirs) and an optional
`--output` path. Run it with `--help` first to confirm the interface, then save
the code spec with the positional `neb_dir` and the `--output` option.
```

**Verify:** `Search the registry for the vasp-neb-to-isaac code.` →
`$PROJ/codes/vasp-neb-to-isaac/SKILL.md` should exist.

### 2. Run the conversion through dsagt-run

```text
Using the registered vasp-neb-to-isaac code, convert the NEB calculation in
data_neb/ and write the ISAAC record to data_neb/isaac_neb_record.json. Use the
exact dsagt-run command from the spec so the execution is recorded. Then tell me
the record's reaction-energy / barrier fields and how many images it summarized.
```

**Expect:** the agent runs `dsagt-run --code vasp-neb-to-isaac -- python
codes/scripts/vasp_neb_to_isaac.py data_neb/ --output data_neb/isaac_neb_record.json`,
pymatgen parses the five OUTCARs, and a v1.05 ISAAC record lands with the
`computation` / `measurement` blocks populated (5 NEB images). Compare against the
reference [`isaac_neb_record.json`](isaac_neb_record.json) shipped here.

### 3. Reconstruct the pipeline

```text
Reconstruct the pipeline from the execution records as a script.
```

## Post-Conditions

1. Code registry contains the `vasp-neb-to-isaac` spec (`codes/vasp-neb-to-isaac/SKILL.md`).
2. `trace_archive/` holds the conversion's provenance record.
3. `data_neb/isaac_neb_record.json` is a valid ISAAC v1.05 record matching the
   shape of the bundled `isaac_neb_record.json`.
4. MLflow traces (in the serverless `mlflow.db` store) capture the run —
   `mlflow ui --backend-store-uri sqlite:///$PROJ/mlflow.db`.

## Cleanup

```bash
dsagt rm isaac-vasp -y
```

## Notes

- The `neb/` OUTCARs are public pymatgen test fixtures vendored here for a
  self-contained demo. They are large (~32 MB total); a future revision may fetch
  them on demand instead of shipping them in-repo.
- The bundled `skills/vasp-to-isaac/` skill is a broader **slab/bulk** converter
  (a different pymatgen workflow that needs `vasprun.xml`-bearing slab/bulk data,
  not the NEB fixture here). It's shipped as a reference skill; the NEB converter
  above is what this folder's data actually exercises.
