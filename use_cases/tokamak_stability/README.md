---
title: Tokamak Stability
domain: Fusion energy — M3D-C1 finite-element simulation data
summary: >-
  Register codes for reading, analyzing, and visualizing data from the M3D-C1
  finite-element code, then use DSAgt to explore an example linear-MHD dataset,
  produce plots and spectra, repackage secondary data products as HDF5, and
  reconstruct the session as a script that reruns on other datasets.
status: published
order: 40
---

# DSAgt Demo: Tokamak Stability

> **Estimated time:** involved / not a 10-minute demo. Setup is the cost:
> building the **fusion-io** C/C++ library from source (scripted, but its
> compilers and libraries must be installed first). Budget roughly an hour
> for first-time setup; the agent session itself is ~15 minutes once the
> dependencies and data are in place.

This guide uses DSAgt to investigate the stability properties of a tokamak
configuration from linear MHD simulation output produced by the
[M3D-C1](https://sites.google.com/pppl.gov/m3d-c1) unstructured-mesh
finite-element code. The Python modules under [`scripts/`](scripts/) — `hdf5.py`,
`m3dc1_tools.py`, `m3dc1_plots.py`, and the `m3dc1/` wrapper package — provide
the functions for reading M3D-C1 HDF5 output, evaluating fields from their
basis-function coefficients, computing equilibrium and spectral quantities, and
plotting. The agent registers those functions as codes under the guidance of
the [`m3dc1-skill`](skills/m3dc1-skill/SKILL.md), which documents the APIs and
the conventions the codes must follow.

The example dataset, a single M3D-C1 simulation output, is courtesy of Alvaro
Sanchez-Villar (PPPL). The session below has been tested with Claude Code.

## Prerequisites

- DSAgt installed with the `tokamak-stability` extra
  (`pip install "dsagt[tokamak-stability] @ git+https://github.com/AI-ModCon/dsagt.git"`),
  which brings `h5py` and `matplotlib`; `numpy` comes with dsagt.
- An agent platform installed and **already authenticated**.
- The [fusion-io](https://github.com/nferraro/fusion-io) library and its Python
  bindings, built from source by [`scripts/setup_env.sh`](scripts/setup_env.sh):

  ```bash
  bash use_cases/tokamak_stability/scripts/setup_env.sh
  ```

  The build needs git, cmake, pkg-config, C/C++/Fortran compilers, MPI, HDF5,
  NetCDF, and LAPACK already installed (macOS: `brew install cmake pkgconf gcc
  open-mpi hdf5 netcdf`; Debian: `apt install cmake pkg-config gfortran
  libopenmpi-dev libhdf5-dev libnetcdf-dev liblapack-dev`); the
  script names whatever is missing and stops. It installs under
  `~/dsagt-projects/.tools/tokamak_stability/fusion-io/` and prints the
  `FIO_INSTALL_DIR`, `PATH`, `PYTHONPATH`, and library-path exports to add to
  your shell before starting the session.


## Setup

```bash
dsagt init
```

At the menu, name the project `tokamak-stability` and pick your agent; the
defaults are fine for the rest. Then:

```bash
PROJ=~/dsagt-projects/tokamak-stability
mkdir -p "$PROJ/data" "$PROJ/skills"
# Demo data (one M3D-C1 simulation output) from the DSAgt use-case data folder:
# https://drive.google.com/drive/folders/1RWQAJeHaikIaD7CCf8ciJ71m55S1erp6
curl -L "https://drive.usercontent.google.com/download?id=1qo-ZG_GoGlZ_X2BjR1fE8zZW3s9K6mTu&export=download&confirm=t" \
  -o tokamak_stability.tar.gz
tar xzf tokamak_stability.tar.gz -C "$PROJ/data" --strip-components=1 tokamak_stability/m3dc1_data
cp -r use_cases/tokamak_stability/scripts "$PROJ/scripts"     # the modules, the m3dc1 package, and their tests
cp -r use_cases/tokamak_stability/skills/m3dc1-skill "$PROJ/skills/"
export PYTHONPATH=$PROJ/scripts:$PYTHONPATH
export M3DC1_DATA_DIR=$PROJ/data/m3dc1_data
python -m pytest "$PROJ/scripts/tests" -q      # all 91 pass with fusion-io and the data in place
dsagt start tokamak-stability                  # mirrors the skill into the agent's native skills dir
```

The tarball is also available from [OSF](https://osf.io/gak3v/files/). The
integration tests need fusion-io and the data directory named by
`M3DC1_DATA_DIR`; failures naming `fpy` or `write_neo_input` mean the
fusion-io install is not on the path.

## Execution

Paste these prompts one at a time. Plots go to `plots/` and new data products
to `processed_data/` under the project directory by default; the skill tells the
agent so, and you can redirect either in a prompt.

### 1. Register the M3D-C1 functions as codes

```text
The directory scripts/ contains three Python modules with functions for
M3D-C1 HDF5 datasets — hdf5.py, m3dc1_tools.py, and m3dc1_plots.py — plus the
m3dc1 wrapper package they use. Before using or registering any of them, read
the m3dc1-skill (skills/m3dc1-skill/SKILL.md) and its references for the
calling conventions, input/output formats, and examples. Then register the
functions as codes and print a summary here.
```

Creating and registering the codes may take several minutes. **Verify:**
`Search the registry for M3D-C1 codes.` The skill requires that codes wrapping
functions that call into fusion-io write their JSON result to an
`--output-json` file rather than stdout; check one such spec (for example the
Miller-geometry code) carries that option.

### 2. Explore the dataset

```text
Using your codes, tell me about the data in data/m3dc1_data/.
```

**Expect:** case metadata, the available time snapshots, and the scalar traces,
read through the registered codes (with `dsagt-run` records in `trace_archive/`).

### 3. Equilibrium quantities

```text
What are the Miller parameters for this configuration?
```

```text
What's the safety factor?
```

**Expect:** the agent runs the Miller-geometry and q-profile codes and reports
the parameters and q95.

### 4. Field plots

```text
Make plots of the t=1 fields of the electron temperature, all components of the
current density, and the perturbations of the density and magnetic flux.
```

**Expect:** PNG files under `plots/`, produced by evaluating the fields on a
grid rather than plotting raw coefficients (the skill says which the user means).

### 5. Spectra and energy trace

```text
Create plots of the standard poloidal spectra and the kinetic energy trace.
```

### 6. Repackage secondary data products

```text
Save the poloidal spectral data for the pressure field in an HDF5 file
processed_data/pressure_spectrum.h5.
```

```text
Extract the electron temperature and electron density data at t=1 and place
them in an HDF5 file processed_data/electrons.h5.
```

**Expect:** both files under `processed_data/`, written by registered codes
(the agent registers a spectrum-computing code for the first if none exists,
and uses the grid-evaluation code for the second), with the evaluated field
values rather than basis coefficients.

### 7. Reconstruct the session as a rerunnable script

```text
Reconstruct the pipeline from the execution records as a bash script, with the
data directory set as a variable at the top so it can be rerun on other M3D-C1
datasets. Save it as dsagt_session_script.sh.
```

**Expect:** `reconstruct_pipeline` renders the `trace_archive/` records in
dependency order. The skill asks the agent to check your default shell first,
since the fusion-io environment variables may be set only in that shell's
startup files.

## Post-Conditions

1. Code registry contains specs for the M3D-C1 reading, analysis, plotting, and
   HDF5 repackaging functions; those wrapping fusion-io calls take `--output-json`.
2. `plots/` holds the field, spectrum, and energy-trace plots.
3. `processed_data/` holds `pressure_spectrum.h5` and `electrons.h5`.
4. `trace_archive/` holds one execution record per code run, and no stray
   temporary JSON files remain in the project.
5. `dsagt_session_script.sh` reruns the session against a data directory set at
   the top of the script.
6. MLflow traces (in the serverless `mlflow.db` store) capture every code
   execution and agent turn — `dsagt traces tokamak-stability`.

## What This Tests

| DSAgt Capability | Steps |
|------------------|-------|
| Module tests as an environment check before the session | Setup |
| Skill-guided code creation from Python modules | 1 |
| Registry search | 1 |
| Code execution with provenance through `dsagt-run` | 2–6 |
| Handling codes whose stdout is unusable (file-based JSON results) | 1, 3 |
| Plot and data-product generation into project subdirectories | 4–6 |
| Pipeline reconstruction with a parameterized input | 7 |

## Cleanup

```bash
dsagt rm tokamak-stability -y
rm tokamak_stability.tar.gz
rm -rf ~/dsagt-projects/.tools/tokamak_stability      # the fusion-io build
```
