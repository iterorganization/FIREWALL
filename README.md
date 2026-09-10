# FIREWALL

This repository contains the Fast Integrated Runaway Electron WALL loads (FIREWALL) code.

**FIREWALL** is a surrogate model that enables a quick assessment of volumetric wall heating by runaway electrons, taking into account the energy and incidence angles of the incoming particles and realistic 3D wall geometry. Specifically, FIREWALL solves multiple time-dependent 1D heat diffusion equations for every mesh triangle in a realistic tokamak 3D wall geometry employing a Finite Volume Method with an implicit time integration scheme to handle the non-linear material properties and stiff source terms efficiently. FIREWALL can be coupled to the JOREK code or other runaway electron modelling codes.

The official FIREWALL paper is TO BE ADDED.

## Key Features

*   **Non-linear Material Properties**: Accurate modelling of Tungsten (W) properties (thermal conductivity, specific heat, density) as functions of temperature.
*   **Particle-based Source Terms**: Direct coupling with particle tracking outputs (HDF5 format) to compute volumetric energy deposition.
*   **Implicit Solver**: Robust tridiagonal solver for unconditional stability with adaptive time-stepping capability.
*   **Adaptive Grid**: Support for non-uniform spatial grids to resolve steep gradients near the surface.
*   **HDF5 Integration**: Efficient I/O for handling large particle datasets and storing simulation results.
*   **Parallel Execution**: OpenMP support for parallel processing of wall elements.

## Requirements

To compile FIREWALL, you need to have the following software installed:
*   **C++ Compiler**: Must support **C++20** (e.g. GCC 10+, Clang 13+).
*   **CMake**: Version 3.20 or later.
*   **HDF5 Library**: C component required (the serial build is sufficient).
*   **OpenMP**: Required. `<omp.h>` is used unconditionally, so a build without
    OpenMP will not link.

Additionally, to use the post-processing Python scripts, you need the following Python packages:
*   **scipy**
*   **h5py**
*   **numpy**
*   **matplotlib**
*   **pyvista**

## Getting the code

The documentation theme is a git submodule, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/iterorganization/FIREWALL.git
cd FIREWALL
```

If you have already cloned without `--recurse-submodules`, fetch the submodule with:

```bash
git submodule update --init --recursive
```

## Compilation

To compile FIREWALL, go to the root FIREWALL directory and run the following commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

## Usage

FIREWALL is run via the command line. The main executable is `firewall`. Other executables exist for benchmarking purposes. Execute FIREWALL using:

```bash
./firewall [options]
```
In the command line, add the following arguments:
### Arguments

*   `--config <path>`: Path to the configuration file (default: `../examples/config.txt`).
*   `--wall <path>`: Path to the wall geometry HDF5 file (default: `./data/wall.h5`).
*   `--part <path>`: Path to the particles HDF5 file (default: `./data/particles.h5`).
*   `--interp <path>`: Path to the interpolation data HDF5 file (default: `./data/interpolation.h5`).
*   `--out <path>`: Path to the output HDF5 file (default: `results.h5`).
*   `--walls <list>`: Comma-separated list of wall IDs to process (default: process all).
*   `--full_profile_walls <list>`: Comma-separated list of wall IDs to store full in-depth temperature profiles for (default: none).
*   `--help`, `-h`: Display help information.

### Example

```bash
./firewall --config my_config.txt --wall geometry.h5 --part particles.h5 --out run1_results.h5
```

The post-processing Python scripts are run via the command line in a similar way.

## Testing

After building, run the smoke tests with:

```bash
ctest --test-dir build --output-on-failure
```

The full end-to-end regression downloads the test dataset from Zenodo, runs the
pipeline and compares the surface temperatures against a committed reference
result. It needs network access, `h5diff` (from `hdf5-tools`) and the Python
requirements, and is run with:

```bash
./tests/run_pipeline.sh
```

## Documentation

The API documentation is published at
<https://iterorganization.github.io/FIREWALL/>.

To build it locally you need Doxygen (and Graphviz for the diagrams). Paths in
`docs/Doxyfile` are relative to the repository root, so run it from there:

```bash
doxygen docs/Doxyfile
```

This writes the HTML into `docs/html`; open `docs/html/index.html` to view it.

## Citing FIREWALL

If you use FIREWALL in your publications, please cite the FIREWALL article: TO BE ADDED

## Development

The code is developed and coordinated by the research unit Magneto-hydrodynamics and fast particles [MHD](https://www.ipp.mpg.de/5035213/mhd) at the Max Planck Institute for Plasma Physics.

The original idea for FIREWALL was proposed by [Svetlana Ratynskaia](https://www.kth.se/profile/srat?l=en) and [Matthias Hölzl](https://www.ipp.mpg.de/person/139800). Victor Johan Svensson developed the first version of the code.
