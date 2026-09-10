cmake_minimum_required(VERSION 3.20)

# Declare project
project(FIREWALL LANGUAGES CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Dependencies
# HDF5 often requires C language enabled to find the C component correctly
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED)

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

# Set C++ Standard specifically for this target.
# C++20 is what the sources actually require (std::span, designated
# initialisers); it is also understood by the minimum CMake declared above.
target_compile_features(core_lib PUBLIC cxx_std_20)

# Warnings are on by default but never fatal, so downstream packagers and
# older/newer compilers cannot be blocked by a new diagnostic.
add_library(firewall_warnings INTERFACE)
if(MSVC)
    target_compile_options(firewall_warnings INTERFACE /W4)
else()
    target_compile_options(firewall_warnings INTERFACE -Wall -Wextra)
endif()
target_link_libraries(core_lib PRIVATE firewall_warnings)

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
message(STATUS "OpenMP found, enabling parallel execution.")
target_link_libraries(core_lib PUBLIC OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

# 1. Main Simulation Executable
add_executable(firewall src/main.cpp)
target_link_libraries(firewall PRIVATE core_lib firewall_warnings)

# 2. Benching Executable
add_executable(benching tests/bench.cpp)
target_link_libraries(benching PRIVATE core_lib firewall_warnings)

# 3. Benching workflow Executable
add_executable(benching_workflow tests/bench_workflow.cpp)
target_link_libraries(benching_workflow PRIVATE core_lib firewall_warnings)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
# BUILD_TESTING is defined by CTest and defaults to ON. Guarding on it lets
# downstream packagers configure with -DBUILD_TESTING=OFF.
include(CTest)

if(BUILD_TESTING)
    # Smoke tests: these need no input data, so they run anywhere (including CI)
    # and catch the most common breakages -- a target that does not link, or an
    # executable that cannot start.
    add_test(NAME firewall_help COMMAND firewall --help)
    add_test(NAME benching_help COMMAND benching --help)
    add_test(NAME benching_workflow_help COMMAND benching_workflow --help)

    # End-to-end regression against the committed reference result. It needs the
    # test dataset from Zenodo, so it is opt-in via -DFIREWALL_ENABLE_PIPELINE_TEST=ON.
    option(FIREWALL_ENABLE_PIPELINE_TEST
           "Run the full download-and-compare pipeline test (requires network and h5diff)" OFF)
    if(FIREWALL_ENABLE_PIPELINE_TEST)
        add_test(NAME pipeline
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_pipeline.sh
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(pipeline PROPERTIES TIMEOUT 3600)
    endif()
endif()
