---
name: m3dc1-skill
description: Use this skill when working with the python modules `m3dc1_tools.py`, `m3dc1_plots.py`, `hdf5.py` and the tools created from the functions within. Triggers when working with M3D-C1 simulation data or repackaging general HDF5 files.
---

# m3dc1-skill


## Creating CLI tools from M3D-C1 python modules

Always read the relevant API guide before creating CLI tools --- see the `Reference material` section below.


IMPORTANT NOTE FOR AGENTS GENERATING NEW TOOLS FROM m3dc1_tools.py
------------------------------------------------------------------
The functions marked "requires m3dc1 + fpy" in the initial comment
in `m3dc1_tools.py` call into compiled C/Fortran code that writes diagnostic 
messages (e.g. "deleting simulation object", "period = 6.28318") directly 
to the OS-level stdout file descriptor. This output bypasses Python's 
sys.stdout entirely and cannot be suppressed or redirected from Python.

Consequence: any CLI tool that (1) calls one of these functions and (2) prints
its JSON result to stdout will produce contaminated output that cannot be parsed
as JSON when captured via shell redirect.

Required pattern for any new CLI tool wrapping these functions:
    - Accept an --output-json FILE argument.
    - Write the JSON result to that file instead of printing to stdout.
    - Document in the tool spec that the agent MUST use --output-json and read
      from the file rather than capturing stdout.

Functions that ARE affected:
    `compute_flux_average_profiles`, `compute_q95`, `compute_miller_geometry`,
    `compute_perturbed_fields`, `compute_poloidal_spectrum`, `compute_standard_spectra`
    `evaluate_field_on_grid`

Functions NOT affected (h5py / numpy only — no compiled stdout writes):
    `read_c1input`, `list_time_snapshots`, `read_snapshot_time`, `read_scalar_traces`,
    `read_case_metadata`, `read_mesh_vertices`, `make_evaluation_grid`,
    `compute_ke_growth_trace`, `compute_growth_rate`, `compute_q95`


## When to use these tools

Always use the M3D-C1 tools when processing or investigating datasets created by this code. Only create custom tools when your needs cannot be met by any of the existing tools. Be watchful for synonyms of alternative expressions from those used in the tool descriptions; for example, use of `repackage_hdf5` should also be triggered by calls to repack or reorganize HDF5 data, or to create a new file to hold a new dataset etc.

When using any tools that write a temporary JSON file (to circumvent the corruption of stdout by fpy as described above) that temporary file should be deleted as soon as it is no longer needed to connect tool calls. Do not leave unnecessary JSON files in the filesystem.

If the user's meaning or intent is unclear always ask a clarifying question --- do not silently fail or say that a required tool does not exist.


## Note on how M3D-C1 field data is stored

M3D-C1 stores fields using the coefficients of the basis functions rather than the pointwise field values themselves. The field values can be found using the `eval_field` function in the `m3dc1` python module; the `eval_field_on_grid` function in `m3dc1_tools.py` evaluates field values across a spatial grid and either returns a dict or writes an HDF5 file. Use these functions or tools derived from them to evaluate fields. If the user asks for field data (e.g. temperature, density, pressure, magnetic field or current density components) to be repackaged, repacked, extracted, moved, rewritten etc. assume that they are interested in these real evaluated field values rather than the raw coefficient values. Only repackage raw coefficients if the user specifically requests this. If the user's intent is unclear always ask a clarifying question.


## Where to store products

Unless explicitly requested by the user, products you create (e.g. plots or new files containing repacked or processed data) should be placed in the active dsagt project directory or its subdirectories. In particular, do not place any new files in the data directory in which the source M3D-C1 data files are located unless requested by the user. By default, place plots in a `plots/` subdirectory and new files containing data in a `processed_data/` subdirectory, creating them if necessary. 

If the user expresses a preference for new file locations attempt to maintain consistency with this choice for the remainder of the session unless new instructions are given; for example, if the user asks for a plot to go in a `new_images/` directory place subsequent plots in the same directory unless directed otherwise. If the user's intentions are unclear (e.g. plots have been placed in more the one directory in the same session) ask for clarification before proceeding.


## Generating shell scripts for session recreation

When asked to make a shell script always check the user's default shell. Create the script for the default shell unless the user requests a different shell, in which case ask for clarification. In particular, Mac users may informally ask for a "bash script" even if they have zsh set as their default shell. This may be a problem as environment variables required for finding libraries such as fusion-io may exist only in the default shell's dot files.


## Reference material

ALWAYS read the relevant API guide before attempting to create CLI tools from one of the python modules.

-  `references/m3dc1_output_guide.md`: explains M3D-C1's native output format in detail, including how the field coefficients are stored and evaluated.

-  `references/m3dc1_tools_api.md`: API and usage guide to the functions in `m3dc1_tools.py`. These are mostly wrappers of the `m3dc1` python module. 

-  `references/m3dc1_plots_api.md`: API and usage guide for plotting using the functions in `m3dc1_plots.py`.

-  `references/hdf5_api.md`: API and usage guide for the basic HDF5 functions in `hdf5.py`.

