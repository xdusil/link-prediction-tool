cmake_minimum_required(VERSION 3.12)

project(LinkPredictionApp VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Generate compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

option(
    LINK_PREDICTION_ENABLE_PREDICTION_THRESHOLD_DIAGNOSTICS
    "Enable diagnostic threshold search on prediction ground truth."
    OFF
)

# Set paths to external libraries
set(MLPACK_INCLUDE_DIR "" CACHE PATH "Path to mlpack include directory")
set(ENSMALLEN_INCLUDE_DIR "" CACHE PATH "Path to ensmallen include directory")
set(CEREAL_INCLUDE_DIR "" CACHE PATH "Path to cereal include directory")

# Add source directory definition
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# Find packages
find_package(Boost 1.86 REQUIRED)
find_package(Torch 2.5 REQUIRED)
find_package(Armadillo 14.2 REQUIRED)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    # Add OpenMP flags to compiler options
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    
    # Add a definition to tell the code to use OpenMP
    add_definitions(-DMLPACK_USE_OPENMP)
    add_definitions(-DUSE_OPENMP)
endif()

# Find paths to include directories
find_path(MLPACK_INCLUDE_DIR_FOUND
    NAMES mlpack/core.hpp
    PATHS ${MLPACK_INCLUDE_DIR}
    REQUIRED
)
find_path(ENSMALLEN_INCLUDE_DIR_FOUND
    NAMES ensmallen.hpp
    PATHS ${ENSMALLEN_INCLUDE_DIR}
    REQUIRED
)
find_path(CEREAL_INCLUDE_DIR_FOUND
    NAMES cereal/cereal.hpp
    PATHS ${CEREAL_INCLUDE_DIR}
    REQUIRED
)

# Recursively find all source files in the src directory
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${SRC_DIR}/*.cpp"
)

# Add the executable
add_executable(LinkPredictionApp
    ${SOURCES}
)

target_compile_definitions(LinkPredictionApp PRIVATE ARMA_DONT_USE_WRAPPER)

if(LINK_PREDICTION_ENABLE_PREDICTION_THRESHOLD_DIAGNOSTICS)
    target_compile_definitions(
        LinkPredictionApp
        PRIVATE LINK_PREDICTION_ENABLE_PREDICTION_THRESHOLD_DIAGNOSTICS
    )
endif()

# Include directories
target_include_directories(LinkPredictionApp PUBLIC
    ${SRC_DIR}
    SYSTEM ${Boost_INCLUDE_DIRS}
    SYSTEM ${TORCH_INCLUDE_DIRS}
    SYSTEM ${ARMADILLO_INCLUDE_DIRS}
    SYSTEM ${MLPACK_INCLUDE_DIR_FOUND}
    SYSTEM ${ENSMALLEN_INCLUDE_DIR_FOUND}
    SYSTEM ${CEREAL_INCLUDE_DIR_FOUND}
)

# Link libraries
target_link_libraries(LinkPredictionApp PUBLIC
    ${TORCH_LIBRARIES}
)

# Ensure C++20 standard
set_property(TARGET LinkPredictionApp PROPERTY CXX_STANDARD 20)
