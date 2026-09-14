# BlastNet → WELL format (example)

This folder is a **small, self-contained copy** of two BlastNet dataset families plus the full `convert_to_well_format.py` used for the COMB-FLOW-UNI conversion pipeline. Use it to demo conversion for an agentic data team or to smoke-test the script without touching the full `blastnet_data` tree.

## Layout

```
Data_conversion_example/
  blastnet_data/                    # input (mirrors blastnet_data/<dataset>/...)
    lifted_hydrogen_jet/
      hydrogen-jet-5000/
      nonreacting-hydrogen-jet-5000/
    nonreacting_channel_flow/
      channelflow-dns-Re544-eq-p000-020/
      channelflow-dns-Re544-eq-p021-041/
  well_output/                      # default output HDF5 files (created by the script)
  convert_to_well_format.py
  submit_convert_to_well_example.sh # optional PBS job (edit project/queue)
  README.md
```

## Original BlastNet layout (per trajectory)

Each **trajectory** is a directory (for example `hydrogen-jet-5000/` or `channelflow-dns-Re544-eq-p000-020/`) containing:

- **`info.json`** — metadata: `global.Nxyz` (grid size), snapshot count, variable list, paths to grid files, BC text, optional chemistry paths.
- **`data/*.dat`** — raw **float32** arrays, one file per field per time index (e.g. `P_Pa_id0000.dat`, `UX_ms-1_id0001.dat`). Names encode the physical quantity and units.
- **`grid/`** — `X_m.dat`, `Y_m.dat`, and for 3D cases `Z_m.dat` (float coordinates; the converter tries float32 then float64).

The converter groups files by trajectory directory and field name, reads `info.json` for dimensions and time, maps BlastNet field names to WELL names, and writes a single HDF5 per trajectory.

## WELL HDF5 layout (output)

Each output file is named **`{dataset_name}_{traj_key}.hdf5`** (for example `lifted_hydrogen_jet_traj_5000.hdf5`, `nonreacting_channel_flow_traj_eq_020.hdf5`).

Inside the file:

- **Root attributes** — `dataset_name`, `grid_type`, `n_spatial_dims`, `n_trajectories`, etc.
- **`dimensions/`** — `time` (1D array of snapshot times), and `x` / `y` / (`z`) coordinate 1D arrays when grid files are present (otherwise normalized \([0,1]\) placeholder).
- **`boundary_conditions/`** — groups per BC type (periodic / wall / open) with masks; types come from `BC_MAPPINGS` in the script for each dataset family.
- **`t0_fields/`** — scalar fields (pressure, density, temperature, species mass fractions, …).
- **`t1_fields/`** — **`velocity`** as a vector field (stacked `UX`, `UY`, `UZ` when present).
- **`t2_fields/`** — reserved (empty in these conversions).
- **`scalars/`** — placeholder group for scalar parameters.

Trajectory keys used in filenames (for `--only-trajectories`):

| Dataset | Example subdirectory | Trajectory key |
|--------|----------------------|----------------|
| `lifted_hydrogen_jet` | `hydrogen-jet-5000` | `traj_5000` |
| `lifted_hydrogen_jet` | `nonreacting-hydrogen-jet-5000` | `traj_nonreacting_5000` |
| `nonreacting_channel_flow` | `channelflow-dns-Re544-eq-p000-020` | `traj_eq_020` |
| `nonreacting_channel_flow` | `channelflow-dns-Re544-eq-p021-041` | `traj_eq_041` |

## Commands (interactive)

From this directory, defaults are `--base-path ./blastnet_data` and `--output-path ./well_output`.

**Dry run (no HDF5 written):**

```bash
cd /lus/flare/projects/COMB-FLOW-UNI/Kiran/Data_conversion_example

python3 convert_to_well_format.py --dataset lifted_hydrogen_jet --dry-run
python3 convert_to_well_format.py --dataset nonreacting_channel_flow --dry-run
```

**Convert lifted hydrogen jet (both trajectories in this example):**

```bash
python3 convert_to_well_format.py --dataset lifted_hydrogen_jet
```

**Convert channel flow (both trajectories):**

```bash
python3 convert_to_well_format.py --dataset nonreacting_channel_flow
```

**Optional: single trajectory by key:**

```bash
python3 convert_to_well_format.py --dataset lifted_hydrogen_jet --only-trajectories traj_5000
python3 convert_to_well_format.py --dataset nonreacting_channel_flow --only-trajectories traj_eq_020
```

**Full BlastNet tree (not this example):** pass explicit paths, same as the original script:

```bash
python3 convert_to_well_format.py --dataset premixed_flame_h2air \
  --base-path /lus/flare/projects/COMB-FLOW-UNI/blastnet_data \
  --output-path /lus/flare/projects/COMB-FLOW-UNI/blastnet_data/blastnet_data_well_format_complete
```

## PBS batch job

Edit account/queue paths in `submit_convert_to_well_example.sh`, then:

```bash
qsub submit_convert_to_well_example.sh
```

The job runs the two `--dataset` conversions sequentially and writes logs under this folder.

## Resource note

Channel-flow trajectories here are **very large** on disk (order \(10^2\) GB per case). Conversion loads time series into memory; run on a node with sufficient **RAM** and allow **long walltime**. Lifted-jet cases are smaller but still multi‑GB outputs.

## Upstream reference

The canonical copy of the converter (and the full converted HDF5 library) lives under:

`/lus/flare/projects/COMB-FLOW-UNI/blastnet_data/convert_to_well_format.py`  
`/lus/flare/projects/COMB-FLOW-UNI/blastnet_data/blastnet_data_well_format_complete/`

This example’s script is the same logic with **local default paths**, optional **`--only-trajectories`**, and the same **`--dataset`** / **`--dry-run`** interface.
