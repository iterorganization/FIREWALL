cmake_minimum_required(VERSION 3.10)
project(HeatSimulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel execution.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(include)
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(heat_sim
    apps/main.cpp
    src/Interpolator.cpp
    src/Solver.cpp
    src/Material.cpp
    src/Utils.cpp
    src/ConfigParser.cpp
    src/ArgParser.cpp
)

target_link_libraries(heat_sim PRIVATE ${HDF5_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(heat_sim PRIVATE OpenMP::OpenMP_CXX)
endif()
