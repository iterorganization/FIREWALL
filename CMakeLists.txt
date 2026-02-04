cmake_minimum_required(VERSION 3.20)

# Declare project
project(HeatSimulation LANGUAGES CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Dependencies
# HDF5 often requires C language enabled to find the C component correctly
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP)

# -----------------------------------------------------------------------------
# Core Library
# (Compiles the logic once, shared by all executables)
# -----------------------------------------------------------------------------
add_library(core_lib STATIC
    src/Interpolator.cpp
    src/Solver.cpp
    src/Material.cpp
    src/Utils.cpp
    src/ConfigParser.cpp
    src/ArgParser.cpp
)

# Set C++ Standard specifically for this target
target_compile_features(core_lib PUBLIC cxx_std_23)

# Include directories
# PUBLIC = accessible to core_lib AND anything that links to it
target_include_directories(core_lib PUBLIC 
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link HDF5
if(TARGET hdf5::hdf5)
    target_link_libraries(core_lib PUBLIC hdf5::hdf5)
else()
    # Fallback for older HDF5 installations
    target_include_directories(core_lib PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(core_lib PUBLIC ${HDF5_LIBRARIES})
endif()

# Link OpenMP
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel execution.")
    target_link_libraries(core_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(core_lib PUBLIC MPI::MPI_CXX)

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

# 1. Main Simulation Executable
add_executable(firewall src/main.cpp)
target_link_libraries(firewall PRIVATE core_lib)

# 2. Benching Executable
add_executable(benching tests/bench.cpp)
target_link_libraries(benching PRIVATE core_lib)

# 3. Benching workflow Executable
add_executable(benching_workflow tests/bench_workflow.cpp)
target_link_libraries(benching_workflow PRIVATE core_lib)