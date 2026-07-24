---
title: Tokamak Stability
domain: Fusion energy - explore finite-element simulation data
summary: >-
  Register tools for reading, analyzing, and visualizing data from the
  M3D-C1 finite-element code. Use DSAgt to explore an example dataset, 
  produce a range of plots, and generate and repackage secondary data products.
  Save the pipeline as a script that can be rerun across other similar datasets.
---


# Tokamak Stability - a fusion energy use case

> **Estimated time:** involved / not a 10-minute demo. Setup is the cost:
> building the **fusion-io** C/C++ library from source. Additionally, an **M3D-C1** 
> dataset is packaged with this use case's necessary text/source files on OSF.io. 
> Budget roughly an hour for first-time setup; the agent session itself is ~15 minutes once 
> dependencies and data are in place.

Here we're going to use dsagt to investigate the stability properties of a tokamak configuration. We'll be looking at linear MHD simulation data produced by the [M3D-C1](https://sites.google.com/pppl.gov/m3d-c1) unstructured-mesh finite-element code. The session below has been tested with Claude Code.


## Example data

A demonstration dataset, consisting of a single M3D-C1 simulation output, 
is available in [this OSF.io directory](https://osf.io/gak3v/files/) as
`tokamak_stability.tar.gz`; an account is not needed for access. 
The tarball also includes all necessary source files, so you can 
run this use case using a pip-installed dsagt without cloning the repository.
The dataset is courtesy of Alvaro Sanchez-Villar (asvillar@pppl.gov). 

Untar the .tar.gz file somewhere convenient; it will create a `tokamak_stability/`
directory.


## Dependencies

In addition to a standard dsagt installation you'll also need to build and install the fusion-io library from [https://github.com/nferraro/fusion-io](https://github.com/nferraro/fusion-io). The top commit of the main branch will work. Use these environment variables to point to the fusion-io installation:

```
export FIO_INSTALL_DIR=/path/to/your/fusion-io/install/
export PYTHONPATH=$FIO_INSTALL_DIR/lib:$PYTHONPATH
export DYLD_LIBRARY_PATH=$FIO_INSTALL_DIR/lib:$DYLD_LIBRARY_PATH
```

This use case also comes with a wrapper python module called `m3dc1`, which you may want to add to your PYTHONPATH:

```
export PYTHONPATH=/path/to/your/tokamak_stability:$PYTHONPATH
```


## Getting started

Install dsagt with pip:

```
python3 -m venv <chosen-venv-directory>
source <chosen-venv-directory>/bin/activate
pip install "git+https://github.com/AI-ModCon/dsagt.git"
```

Now create a new dsagt project and associated project directory: 

```
dsagt init 
```

Project directories are placed in `$HOME/dsagt-projects/` by default. If you prefer a different location either give a directory name (in which case the project directory is placed in the current directory or in an appropriate subdirectory) or a full absolute path; do not use `~` or `$HOME` as they will be interpreted literally. Select your agent of choice.

Every tool execution and agent turn is logged automatically to an SQLite MLflow store `mlflow.db`
in the project directory. We'll use this to reconstruct the session later. 

Now start the agent and dsagt's MCP server:

```
dsagt start <chosen-project-name>
```

Note that here we use the project name and not the name of the project directory, which you choose separately. 


## An example session

Inside the agent, enter these prompts one at a time, replacing the placeholder directory paths with the corresponding paths on your system:

-   Read the `AGENTS.md` file in the `/path/to/your/tokamak_stability` directory. This directory contains three python files with functions for dealing with HDF5 datasets produced by the M3D-C1 code: `hdf5.py`, `m3dc1_tools.py`, and `m3dc1_plots.py`. Register these functions as tools and print a summary here. 

Creating and registering the tools may take several minutes.

-   Using your tools, tell me about the data in the `/path/to/your/tokamak_stability/m3dc1_data/` directory.

-   What are the Miller parameters for this configuration?

-   What's the safety factor?

-   Make plots of the t=1 fields of the electron temperature, all components of the current density, and the perturbations of the density and magnetic flux.

Plots will go in a `plots/` subdirectory of your project directory by default. Instruct the agent if you prefer an alternative location.

-   Create plots of the standard poloidal spectra and the kinetic energy trace. 

-   Save the poloidal spectral data for the pressure field in an HDF5 file `pressure_spectrum.h5`.

New data products will go to a `processed_data/` subdirectory of your project directory by default.

-   Extract the electron temperature and electron density data at t=1 and place them in an HDF5 file `electrons.h5`.

-   Using the MLflow traces in the `mlflow.db` store, create a shell script that recreates this session's tool executions on a general data directory that is set at the top of the script. Save as `dsagt_session_script.sh`.

(You can browse those traces any time by running `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the project directory.)

Now exit the agent session as usual (e.g. with the `exit` command).

The project name is registered in `~/dsagt-projects/projects.yaml` (default location). You can unregister the project name with

```
dsagt rm <chosen-project-name>
```

You will be asked whether you want to delete the project directory.

