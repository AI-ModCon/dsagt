# Fusion energy use case

> **Estimated time:** advanced / not a 10-minute demo. Setup is the cost:
> building the **fusion-io** C/C++ library from source and downloading a
> Google-Drive-hosted **M3D-C1** dataset. Budget an hour+ for first-time setup;
> the agent session itself is ~15 minutes once dependencies and data are in
> place.

Here we're going to use dsagt to investigate the stability properties of a tokamak configuration. We'll be looking at linear MHD simulation data produced by the [M3D-C1](https://sites.google.com/pppl.gov/m3d-c1) unstructured-mesh finite-element code. The session below has been tested with Claude Code.


## Dependencies

In addition a standard dsagt installation you'll also need to build and install the fusion-io library from [https://github.com/nferraro/fusion-io](https://github.com/nferraro/fusion-io). The top commit of the main branch will work. Use these environment variables to point to the fusion-io installation:

```
export FIO_INSTALL_DIR=/path/to/your/fusion-io/install/
export PYTHONPATH=$FIO_INSTALL_DIR/lib:$PYTHONPATH
export DYLD_LIBRARY_PATH=$FIO_INSTALL_DIR/lib:$DYLD_LIBRARY_PATH
```

This use case also comes with a wrapper python module called `m3dc1`, which you may want to add to your PYTHONPATH:

```
export PYTHONPATH=/path/to/dsagt/use_cases/tokamak_stability:$PYTHONPATH
```

## Example data

You can download a [demonstration dataset](https://drive.google.com/file/d/1ZghND-G2SInuovLqrg-DVPyECEASPTSq/view?usp=sharing), consisting of a single M3D-C1 simulation output.
This dataset is courtesy of Alvaro Sanchez-Villar (asvillar@pppl.gov). Untar the .tar.gz file somewhere convenient.


## Getting started


With dsagt installed, activate the virtual environment from the repository root:

```
source .venv/bin/activate
```

Now create a new dsagt project and associated directory, here called `fusion-use-case`:

```
dsagt init fusion-use-case --agent claude --location ~/data/
```

Here we're using Claude Code; see the dsagt [documentation](https://ai-modcon.github.io/dsagt/) for guidelines for using other agents. The created project directory will be a subdirectory of `~/data/`; omitting this will place your project directories in `~/dsagt-projects/`.

Observability is serverless — there is nothing to start. Every code execution
and agent turn is logged automatically to a SQLite MLflow store at
`~/data/fusion-use-case/mlflow.db`, which we'll use to reconstruct the session
later. Now enter the project directory and start the agent:

```
cd ~/data/fusion-use-case
claude
```


## An example session

Inside the agent, enter these prompts one at a time, replacing the placeholder directory paths with the corresponding paths on your system:

-   There are three python files in the `/path/to/your/dsagt/use_cases/tokamak_stability` directory containing functions for dealing with HDF5 datasets produced by the M3D-C1 code: `hdf5.py`, `m3dc1_tools.py`, and `m3dc1_plots.py`. Register these functions as codes and print a summary here. Read the `AGENTS.md` file in that directory first.

Creating and registering the codes may take several minutes.

-   Using your codes, tell me about the data in the `/path/to/your/m3dc1_data` directory.

-   What are the Miller parameters for this configuration?

-   What's the safety factor?

-   Make plots of the t=1 fields of the electron temperature, all components of the current density, and the perturbations of the density and magnetic flux.

Plots will go in the `/path/to/fusion-use-case/plots/` subdirectory of your project directory by default. Instruct the agent if you prefer an alternative location.

-   Create plots of the standard poloidal spectra and the kinetic energy trace. 

-   Save the poloidal spectral data for the pressure field in an HDF5 file `pressure_spectrum.h5`.

New data products will go to a `processed_data/` subdirectory of your project directory by default.

-   Extract the electron temperature and electron density data at t=1 and place them in an HDF5 file `electrons.h5`.

-   Using the MLflow traces in the `mlflow.db` store, create a shell script that recreates this session's code executions on a general data directory that is set at the top of the script. Save as `dsagt_session_script.sh`.

(You can browse those traces any time with
`mlflow ui --backend-store-uri sqlite:///~/data/fusion-use-case/mlflow.db`.)

Now exit the agent session as usual (there is no server to stop — the store is
serverless).

The project name (here "fusion-use-case") is registered in `~/dsagt-projects/projects.yaml` (default location). You can unregister the project name with

```
dsagt rm fusion-use-case
```

You will be asked whether you want to delete the project directory.






