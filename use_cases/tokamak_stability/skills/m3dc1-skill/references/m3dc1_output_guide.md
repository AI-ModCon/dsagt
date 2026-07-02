# M3D-C1 Output File Guide

This guide documents the structure of HDF5 output files produced by the M3D-C1
extended-MHD simulation code, based on direct inspection of the example dataset at
`/Users/kparfrey/data/asv/run1/m3dc1_data/` and the reference scripts in `existing/`.


## 1. Files in a Case Directory

A typical case directory contains the following files:

```
C1.h5               Main simulation output (HDF5)
equilibrium.h5      Standalone copy of the equilibrium state (HDF5)
time_000.h5         Standalone snapshot at the first saved timestep (HDF5)
time_001.h5         Standalone snapshot at the last saved timestep (HDF5)
C1input             Fortran namelist input parameters (ASCII)
geqdsk              Grad-Shafranov equilibrium in GEQDSK format (ASCII)
```

The `time_NNN.h5` files contain the same data as the corresponding `time_NNN/` groups
inside `C1.h5`. Similarly, `equilibrium.h5` is byte-for-byte identical to the
`equilibrium/` group in `C1.h5`. The standalone files are written as a convenience so
that individual snapshots can be loaded without opening the full aggregated file.


## 2. Simulation Units and Coordinates

M3D-C1 uses an internal normalization where the time unit is the Alfvén time τ_A.
Physical quantities stored in the HDF5 files are in these internal (normalized) units
unless the post-processing library performs a unit conversion.

The spatial coordinate system is cylindrical: `(R, φ, Z)` where R is the major radius
in metres, φ is the toroidal angle in radians, and Z is the vertical coordinate in
metres. For the example SPARC case:

- R range: 1.25 – 2.55 m
- Z range: −1.25 – 1.25 m
- Magnetic axis: R ≈ 1.855 m, Z ≈ 0.0 m
- Lower X-point:  R ≈ 1.492 m, Z ≈ −1.043 m
- ψ at magnetic axis (ψ_min): −1.438  (internal units)
- ψ at LCFS (ψ_LCFS):          +0.0235 (internal units)


## 3. The C1.h5 File

`C1.h5` is the master output file. It contains three top-level groups:

```
C1.h5
├── equilibrium/          n=0 equilibrium state (stored once)
│   ├── mesh/
│   └── fields/
├── time_NNN/             Perturbed state at each saved timestep (one group per snapshot)
│   ├── mesh/
│   └── fields/
└── scalars/              Time-history traces of global diagnostics
```

Top-level attributes on each time group and the equilibrium group:

```
Attribute        Type      Meaning
──────────────────────────────────────────────────────────────────────────────
version          int32     File format version (example: 45)
nspace           int32     Spatial dimensionality (2 = 2D poloidal mesh)
ntimestep        int32     Timestep number of this snapshot (0 for equilibrium)
time             float64   Simulation time of this snapshot in Alfvén times
iwall_regions    int32     Flag: whether wall-region elements are present
```


## 4. Mesh Structure

Both `equilibrium/mesh/` and every `time_NNN/mesh/` contain the same two datasets
describing the finite-element mesh on the poloidal cross-section. The mesh does not
change during the simulation.

### 4.1 `mesh/elements`  —  shape (nelms, 8)

Each row describes one triangular finite element. For the example case,
nelms = 18 555.

```
Column   Meaning
──────────────────────────────────────────────────────────────────────────────
0        ΔR component of the first edge vector (metres; may be negative)
1        ΔZ component of the first edge vector (metres; may be negative)
2        Length of the first edge: √(col0² + col1²) (metres; always positive)
3        Orientation angle α of the local coordinate frame (radians; −π to π)
4        R coordinate of the element reference vertex (metres)
5        Z coordinate of the element reference vertex (metres)
6        Zone/region flag (integer encoded as float: 0, 1, 2, or 4)
7        Constant equal to 1.0 (padding / normalisation flag)
```

Columns 0–3 define the affine mapping from a reference triangle to the physical
element, which is used internally by the FEM basis functions.

Zone codes (column 6):

```
Zone   Count    Meaning
──────────────────────────────────────────────────────────────────────────────
0      18 246   Interior plasma region (bulk of the mesh)
1          1    Magnetic axis element
2        286    Scrape-off layer / wall-adjacent region
4         22    Special boundary elements
```

Mesh group attributes:

```
Attribute        Type      Value (example)     Meaning
────────────────────────────────────────────────────────────────────────────────
nelms            int32     18 555              Number of triangular elements
nplanes          int32     1                   Number of toroidal planes (1 = 2D)
nperiods         int32     1                   Toroidal periodicity
period           float64   6.2832…             Toroidal period (2π for full torus)
height           float64   2.4996              Poloidal height of the mesh (m)
width            float64   1.2999              Radial width of the mesh (m)
3D               int32     0                   Flag: 0 = 2D mesh, 1 = 3D mesh
ifull_torus      int32     0                   Flag: full-torus (1) or wedge (0)
version          int32     45                  Format version
```

### 4.2 `mesh/adjacency`  —  shape (nelms, 3)

Each row lists the three element indices (0-based) of the neighbours sharing an edge
with this element. A value of −1 indicates a boundary with no neighbour on that side.

```
dtype:   int32
range:   −1 to nelms − 1
```


## 5. Field Arrays

### 5.1 Dataset shape and layout

Every field dataset inside `fields/` has shape **(nelms, 20)**.

- **First axis** (size nelms = 18 555): element index, matching the row order in
  `mesh/elements` and `mesh/adjacency`.
- **Second axis** (size 20): the 20 finite-element expansion coefficients stored per
  element. M3D-C1 uses high-order C1-continuous triangular elements; these 20 values
  are the degrees of freedom of the FEM basis within each element. They are not
  directly interpretable as pointwise field values — use the `m3dc1` Python library
  (specifically `eval_field()`) to evaluate the field at arbitrary (R, φ, Z) points.

All field datasets use dtype `float32`.

### 5.2 Real and imaginary parts: the `_i` suffix

Because M3D-C1 performs a Fourier decomposition in the toroidal direction, perturbed
fields are stored as complex amplitudes. The real and imaginary parts are stored as
separate datasets using the naming convention:

```
field      real part of the n = ntor Fourier component
field_i    imaginary part of the n = ntor Fourier component
```

The equilibrium (`n = 0`) fields have no meaningful imaginary part (they are
axisymmetric), but `field_i` datasets are still written for consistency.

The full 3D field at a given toroidal angle φ is reconstructed as:

```
f_total(R, φ, Z) = f_eq(R, Z) + f_real(R, Z) · cos(n·φ)
                                − f_imag(R, Z) · sin(n·φ)
```

where `n = ntor` (the toroidal mode number from `C1input`).

### 5.3 Equilibrium vs. perturbed fields

The `equilibrium/fields/` group stores the **axisymmetric (n = 0) equilibrium state**.
Each `time_NNN/fields/` group stores the **perturbed (n = ntor) component only** —
not the total field. The post-processing workflow explicitly separates the two:

```python
f_pert(R, Z) = eval_field(name, R, φ, Z, sim=sim_lin) - eval_field(name, R, φ, Z, sim=sim_eq)
```

In the example case (ntor = 9, linear run), `time_000/fields/psi` is effectively zero
(no initial psi perturbation), while `time_001/fields/psi` shows the fully grown n = 9
instability after 1000 Alfvén times.


## 6. Field Catalogue

### 6.1 Fields present only in the equilibrium group

These transport coefficient fields are time-independent and are only written once:

```
Name       Meaning
──────────────────────────────────────────────────────────────────────────────
eta        Resistivity
eta_J      Resistivity × J (Ohmic heating term)
kappa      Isotropic thermal conductivity
kappar     Field-aligned thermal conductivity
visc       Isotropic viscosity
visc_c     Compressional viscosity
denm       Neutral/minority species density
```

### 6.2 Fields present in all groups (equilibrium and all time snapshots)

#### Magnetic / electromagnetic

```
Name       Long name / description
──────────────────────────────────────────────────────────────────────────────
psi        Poloidal magnetic flux (stream function)
psi_i      Imaginary part of psi perturbation
f          Toroidal field function F = R·Bφ
f_i        Imaginary part of f perturbation
fp         Derivative of f with respect to ψ
fp_i       Imaginary part of fp perturbation
phi        Electrostatic / stream-function potential
phi_i      Imaginary part of phi
I          Parallel current
I_i        Imaginary part of I
E_R        Radial electric field component
E_R_i      Imaginary part of E_R
E_Z        Vertical electric field component
E_Z_i      Imaginary part of E_Z
E_PHI      Toroidal electric field component
E_PHI_i    Imaginary part of E_PHI
E_par      Parallel electric field
E_par_i    Imaginary part of E_par
jphi       Toroidal current density
jphi_i     Imaginary part of jphi
```

#### Kinetic / thermodynamic

```
Name       Long name / description
──────────────────────────────────────────────────────────────────────────────
P          Total pressure (ions + electrons)
P_i        Imaginary part of P
Pe         Electron pressure
Pe_i       Imaginary part of Pe
V          Flow velocity (or poloidal flux related)
V_i        Imaginary part of V
te         Electron temperature
te_i       Imaginary part of te
ti         Ion temperature
ti_i       Imaginary part of ti
ne         Electron number density
ne_i       Imaginary part of ne
den        Total mass density
den_i      Imaginary part of den
chi        Flux-surface label (related to poloidal flux)
chi_i      Imaginary part of chi
zeff       Effective ion charge Z_eff
zeff_i     Imaginary part of zeff
```

#### Transport / source terms

```
Name           Long name / description
──────────────────────────────────────────────────────────────────────────────
bdotgradp      B · ∇p  (parallel pressure gradient, drive term)
bdotgradp_i    Imaginary part of bdotgradp
bdotgradt      B · ∇T  (parallel temperature gradient)
bdotgradt_i    Imaginary part of bdotgradt
torque_em      Electromagnetic torque
torque_em_i    Imaginary part of torque_em
torque_ntv     Neoclassical toroidal viscosity torque
torque_ntv_i   Imaginary part of torque_ntv
```

#### Classification / geometry flags

```
Name            Meaning
──────────────────────────────────────────────────────────────────────────────
magnetic_region  0.0 = outside separatrix, 1.0 = inside LCFS
mesh_zone        Zone code, matching elements column 6
wall_dist        Distance to the nearest wall surface (metres)
```


## 7. Scalars: Global Time Traces

`scalars/` contains 97 datasets, each of shape **(ntimestep + 1,)** = **(1001,)** for
this case. The values are stored in the same internal (Alfvén-unit) normalisation as
the field arrays.

### 7.1 Time coordinate

```
Name   Description
──────────────────────────────────────────────────────────────────────────────
time   Simulation time at each output step (Alfvén times; 0.0 – 1000.0)
dt     Time-step size used at each step
```

The `scalars/` group itself carries `@ntimestep = 1000`, reflecting the total number of
steps taken.

### 7.2 Energy diagnostics

All energies are in internal (normalised) units.

```
Name         Meaning
──────────────────────────────────────────────────────────────────────────────
E_K3         Total 3D kinetic energy
E_K3D        3D kinetic energy (D-species component)
E_K3H        3D kinetic energy (H-species component)
E_KP         Poloidal kinetic energy
E_KPD        Poloidal kinetic energy (D-species)
E_KPH        Poloidal kinetic energy (H-species)
E_KT         Toroidal kinetic energy
E_KTD        Toroidal kinetic energy (D-species)
E_KTH        Toroidal kinetic energy (H-species)
E_MP         Perturbed magnetic energy
E_MPC        Perturbed magnetic energy (compressional)
E_MPD        Perturbed magnetic energy (D-species)
E_MPH        Perturbed magnetic energy (H-species)
E_MPV        Perturbed magnetic energy (vacuum region)
E_MT         Total magnetic energy (equilibrium + perturbation)
E_MTC        Total magnetic energy (compressional)
E_MTD        Total magnetic energy (D-species)
E_MTH        Total magnetic energy (H-species)
E_MTV        Total magnetic energy (vacuum)
E_P          Total pressure energy
E_PD         Pressure energy (density component)
E_PE         Electron pressure energy
E_PH         Pressure energy (H-component)
E_grav       Gravitational potential energy
W_M          Integrated perturbed magnetic energy (used for growth-rate diagnostics)
W_P          Integrated perturbed pressure energy
```

In the example linear run, `E_K3` grows from ~1.4 × 10⁻¹³ at t = 1 to ~9.7 × 10⁻⁸ at
t = 1000, giving an estimated growth rate γ ≈ 0.0067 τ_A⁻¹ for the n = 9 mode.

### 7.3 Geometry and equilibrium state

```
Name        Description
──────────────────────────────────────────────────────────────────────────────
xmag        R coordinate of the magnetic axis (metres)
zmag        Z coordinate of the magnetic axis (metres)
xnull       R coordinate of the primary X-point (metres)
znull       Z coordinate of the primary X-point (metres)
xnull2      R coordinate of the secondary X-point (metres)
znull2      Z coordinate of the secondary X-point (metres)
psimin      ψ at the magnetic axis (internal units; most negative value)
psi_lcfs    ψ at the last closed flux surface
psi0        Reference ψ value (axis or vacuum)
```

For the example case: magnetic axis at (1.855, ≈0) m; X-point at (1.492, −1.043) m.

### 7.4 Integral conservation quantities

```
Name                          Description
──────────────────────────────────────────────────────────────────────────────
toroidal_current              Total toroidal plasma current
toroidal_current_p            Toroidal current (plasma region only)
toroidal_current_w            Toroidal current (wall/SOL region)
toroidal_flux                 Total toroidal magnetic flux
toroidal_flux_p               Toroidal flux (plasma region)
angular_momentum              Total angular momentum
angular_momentum_p            Angular momentum (plasma region)
helicity                      Magnetic helicity
volume                        Total plasma volume
volume_p                      Plasma volume (inside LCFS)
volume_pd                     Plasma volume (D-species)
area                          Plasma cross-sectional area
area_p                        Cross-sectional area (plasma region)
particle_number               Total particle number (ions)
particle_number_p             Particle number (plasma region)
electron_number               Total electron count
circulation                   Poloidal flow circulation
loop_voltage                  Inductive loop voltage
```

### 7.5 Radiation and particle sources

```
Name                          Description
──────────────────────────────────────────────────────────────────────────────
radiation                     Total radiated power
brem_rad                      Bremsstrahlung radiation
line_rad                      Line radiation
reck_rad                      Recombination radiation (K channel)
recp_rad                      Recombination radiation (P channel)
kprad_n                       KPRAD impurity density
kprad_n0                      KPRAD initial impurity density
kprad_dt                      KPRAD time derivative
Particle_source               Particle source rate
Particle_Flux_convective      Convective particle flux across LCFS
Particle_Flux_diffusive       Diffusive particle flux across LCFS
ion_loss                      Ion particle loss rate
runaways                      Runaway electron count
temax                         Maximum electron temperature
```

### 7.6 Power and energy fluxes

```
Name                          Description
──────────────────────────────────────────────────────────────────────────────
power_injected                Total injected power
Flux_kinetic                  Kinetic energy flux across LCFS
Flux_poynting                 Poynting flux
Flux_pressure                 Pressure-driven energy flux
Flux_thermal                  Thermal conduction flux
Parallel_viscous_heating      Heating from parallel viscous dissipation
```

### 7.7 Torques

```
Name                 Description
──────────────────────────────────────────────────────────────────────────────
Torque_em            Electromagnetic (J × B) torque
Torque_visc          Viscous torque
Torque_ntv           Neoclassical toroidal viscosity torque
Torque_com           Combined torque
Torque_gyro          Gyroviscous torque
Torque_parvisc       Parallel viscosity torque
Torque_sol           Scrape-off-layer torque
```

### 7.8 Wall forces and Fourier diagnostics

```
Name                     Description
──────────────────────────────────────────────────────────────────────────────
Wall_Force_n0_x          n=0 wall force, x-component
Wall_Force_n0_x_halo     n=0 wall force (halo current), x-component
Wall_Force_n0_y          n=0 wall force, y-component
Wall_Force_n0_z          n=0 wall force, z-component
Wall_Force_n0_z_halo     n=0 wall force (halo current), z-component
Wall_Force_n1_x          n=1 wall force, x-component
Wall_Force_n1_y          n=1 wall force, y-component
IP_co                    Plasma current, cos-component of n=ntor Fourier mode
IP_sn                    Plasma current, sin-component of n=ntor Fourier mode
M_IZ                     Edge / wall current amplitude
M_IZ_co                  Edge current, cos-component
M_IZ_sn                  Edge current, sin-component
i_control%err_i          Current control integrator error
i_control%err_p_old      Current control proportional error (previous step)
n_control%err_i          Density control integrator error
n_control%err_p_old      Density control proportional error (previous step)
Ave_P                    Volume-averaged pressure
```


## 8. The Standalone Snapshot Files

### `equilibrium.h5`

Contains exactly the same data as `C1.h5/equilibrium/`. The top-level structure is:

```
equilibrium.h5
├── fields/          (same 56 datasets as C1.h5/equilibrium/fields/)
└── mesh/            (elements, adjacency)
```

Top-level attributes: same as the `equilibrium` group in `C1.h5`.

### `time_NNN.h5`

Contains exactly the same data as `C1.h5/time_NNN/`. The top-level structure is:

```
time_NNN.h5
├── fields/          (50 datasets: same as C1.h5/time_NNN/fields/)
└── mesh/            (elements, adjacency)
```

Top-level attributes: same as the `time_NNN` group in `C1.h5`, including `ntimestep`
and `time` reflecting the actual simulation time of this snapshot.

The files are named sequentially (`time_000.h5`, `time_001.h5`, …) corresponding to
the order in which snapshots were saved; the file index does not necessarily equal the
timestep number. The actual timestep number and simulation time are read from the
`ntimestep` and `time` attributes.


## 9. The `C1input` Parameter File

`C1input` is an ASCII Fortran namelist file (`&inputnl … /`) that controls the
simulation. Key parameters relevant to post-processing:

```
Parameter       Type     Example value    Meaning
─────────────────────────────────────────────────────────────────────────────────────
ntor            int      9                Toroidal mode number n of the perturbation
pscale          float    0.8150           Pressure normalisation scale factor
batemanscale    float    1.0634           Bateman transport scaling factor
dt              float    1.0              Time step (Alfvén times)
ntimemax        int      1000             Total number of time steps
ntimepr         int      1000             Steps between snapshot outputs
linear          int      1                0 = nonlinear run, 1 = linear run
numvar          int      3                Physics model: 1=2-field, 2=4-field, 3=6-field
idens           int      1                1 = evolve density equation
ion_mass        float    2.0              Ion mass in atomic mass units (2 = deuterium)
zeff            float    1.0              Effective charge number Z_eff
itor            int      1                1 = toroidal geometry
```

`pscale` and `batemanscale` are used to rescale pressure and transport from the GEQDSK
equilibrium to the M3D-C1 normalisation. `ntor` determines which Fourier component is
stored in the `field` / `field_i` datasets in the time-slice groups.


## 10. How to Access Data

A set of standalone post-processing tools is provided in `m3dc1_tools.py`.
These are designed for use in an agentic pipeline and cover the most common
operations. See `m3dc1_tools_api.md` in this directory for the full API reference.

### Recommended starting point — case summary

```python
from m3dc1_tools import read_case_metadata

meta = read_case_metadata("m3dc1_data")
# meta["params"]["ntor"]   → 9
# meta["snapshots"]        → [0, 1]
# meta["R_mag"]            → 1.855 m
```

### Listing and reading scalar time traces

```python
from m3dc1_tools import read_scalar_traces

traces = read_scalar_traces("m3dc1_data", names=["time", "E_K3"])
t   = traces["time"]   # Alfvén times, shape (1001,)
ke  = traces["E_K3"]   # kinetic energy
```

### Reading mesh vertices

```python
from m3dc1_tools import read_mesh_vertices

R, Z = read_mesh_vertices("m3dc1_data/C1.h5")
# R, Z are 1-D float32 arrays of unique mesh vertex positions
```

### Building evaluation grids for field interpolation

```python
from m3dc1_tools import read_mesh_vertices, make_evaluation_grid

R_v, Z_v = read_mesh_vertices("m3dc1_data/C1.h5")
R, Z, phi = make_evaluation_grid(R_v, Z_v, mode="mesh")
# or use a regular 200×200 Cartesian grid:
R, Z, phi = make_evaluation_grid(R_v, Z_v, mode="grid", grid_res=200)
```

### Growth rate

```python
from m3dc1_tools import compute_growth_rate

gamma = compute_growth_rate("m3dc1_data")   # → ~0.0067 τ_A⁻¹
```

### Listing all HDF5 variables

```python
from hdf5 import list_h5_variables

variables = list_h5_variables("m3dc1_data/C1.h5")
```

### Reading a single HDF5 dataset directly

```python
from hdf5 import read_h5_dataset, read_h5_attrs

psi = read_h5_dataset("m3dc1_data/C1.h5", "equilibrium/fields/psi")
# psi shape: (18555, 20)  — FEM coefficients, not pointwise values

attrs = read_h5_attrs("m3dc1_data/C1.h5", "equilibrium")
# {"version": 45, "nspace": 2, "ntimestep": 0, "time": 0.0}
```

### Evaluating a field at arbitrary (R, Z) points (requires m3dc1)

The FEM coefficient arrays must be evaluated through the `m3dc1` library;
they cannot be interpolated manually. Use:

```python
import fpy
from m3dc1.eval_field import eval_field

sim = fpy.sim_data("m3dc1_data/C1.h5", time=-1)   # time=-1 = equilibrium
# eval_field argument order: (name, R, phi, Z, ...)
psi_values = eval_field("psi", R, phi, Z, coord="scalar", sim=sim, time=sim.timeslice)
```

For perturbed-field evaluation across a full snapshot, use the higher-level
wrapper:

```python
from m3dc1_tools import compute_perturbed_fields

perts = compute_perturbed_fields("m3dc1_data", time_idx=1, R=R, Z=Z, phi=phi)
# perts["psi"]  → perturbed psi at each (R, Z, phi) point
```


## 11. Notes on Field Magnitudes and Units

The fields are stored in M3D-C1's internal normalisation. Absolute values are only
meaningful in context:

- **Equilibrium `psi`**: ranges ≈ −480 000 to +452 000 (internal flux units)
- **Equilibrium `te`**: ranges ≈ ±7.9 × 10⁶ (normalised temperature; includes FEM
  coefficients, not directly in eV)
- **Perturbed `psi` (time_001)**: max |ψ_pert| ≈ 1.8 × 10⁶ after 1000 τ_A of growth
- **Energies (scalars)**: in normalised units; E_K3 grows from ~10⁻¹³ to ~10⁻⁷ over
  1000 τ_A

The `m3dc1` Python library handles conversion to physical units (MKS) when `units="mks"`
is passed to evaluation functions.


## 12. Summary Table of HDF5 Datasets

```
Path                              Shape            dtype    Description
───────────────────────────────────────────────────────────────────────────────────
equilibrium/mesh/elements         (18555, 8)       float32  Element geometry
equilibrium/mesh/adjacency        (18555, 3)       int32    Element neighbours
equilibrium/fields/<name>         (18555, 20)      float32  Equilibrium field (56 fields)
time_NNN/mesh/elements            (18555, 8)       float32  Same mesh (unchanged)
time_NNN/mesh/adjacency           (18555, 3)       int32    Same adjacency
time_NNN/fields/<name>            (18555, 20)      float32  Perturbed field (50 fields)
time_NNN/fields/<name>_i          (18555, 20)      float32  Imaginary part (50 fields)
scalars/<name>                    (1001,)          float32  Global time traces (97 traces)
scalars/time                      (1001,)          float32  Time axis (Alfvén times)
```
