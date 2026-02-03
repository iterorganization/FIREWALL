# FIREWALL

**FIREWALL** is a high-performance C++ simulation tool designed for thermal analysis of plasma-facing components in fusion devices. It solves the 1D non-linear heat equation to predict surface temperature evolution under intense particle energy deposition.

## Overview

The code simulates the thermal response of wall materials (specifically Tungsten) when subjected to heat fluxes derived from incident particle data. It employs a Finite Volume Method (FVM) with an implicit time integration scheme to handle the non-linear material properties and stiff source terms efficiently.

## Key Features

*   **Non-linear Material Properties**: Accurate modelling of Tungsten (W) properties (thermal conductivity, specific heat, density) as functions of temperature.
*   **Particle-based Source Terms**: Direct coupling with particle tracking outputs (HDF5 format) to compute volumetric energy deposition.
*   **Implicit Solver**: Robust tridiagonal solver for unconditional stability with adaptive time-stepping capability.
*   **Adaptive Grid**: Support for non-uniform spatial grids to resolve steep gradients near the surface.
*   **HDF5 Integration**: Efficient I/O for handling large particle datasets and storing simulation results.
*   **Parallel Execution**: OpenMP support for parallel processing of wall elements.

## Build Instructions

FIREWALL uses CMake for build configuration.

### Prerequisites

*   **C++ Compiler**: Must support **C++20**.
*   **CMake**: Version 3.20 or later.
*   **HDF5 Library**: C component required.
*   **OpenMP**: Recommended for parallel execution.

### Steps

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

The application is run via the command line. The main executable is `firewall`.

```bash
./firewall [options]
```

### Arguments

*   `--config <path>`: Path to the configuration file (default: `../examples/config.txt`).
*   `--wall <path>`: Path to the wall geometry HDF5 file (default: `./data/wall.h5`).
*   `--part <path>`: Path to the particles HDF5 file (default: `./data/particles.h5`).
*   `--interp <path>`: Path to the interpolation data HDF5 file (default: `./data/interpolation.h5`).
*   `--out <path>`: Path to the output HDF5 file (default: `results.h5`).
*   `--walls <list>`: Comma-separated list of wall IDs to process (default: process all).
*   `--help`, `-h`: Display help information.

### Example

```bash
./firewall --config my_config.txt --wall geometry.h5 --part particles.h5 --out run1_results.h5
```

## Documentation

Full API documentation is available in the `docs/html` directory. Open `index.html` in your browser to view it.