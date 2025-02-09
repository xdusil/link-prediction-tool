# LinkPredictionApp

## Overview
This project builds a C++ application for link prediction of device dependencies in a network.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)

## Dependencies

### Required
- **[CMake](https://cmake.org/)** (>= 3.18)
- **C++ Compiler** with C++20 support (e.g., GCC >= 10, Clang >= 10)
- **[Boost](https://www.boost.org/)** (>= 1.86)
- **[Torch](https://pytorch.org/)** (>= 2.5, CPU version)  
  *This project uses the LibTorch C++ API for CPU computations.*
- **[Armadillo](https://arma.sourceforge.net/)** (>= 14.2)
- **[mlpack](https://www.mlpack.org/index.html)** (>= 4.5)
- **[ensmallen](https://ensmallen.org/)** (>= 2.22)  
  *Used internally by mlpack; the required version depends on your mlpack version.*
- **[cereal](https://uscilab.github.io/cereal/)** (>= 1.3)  
  *Used internally by mlpack; the required version depends on your mlpack version.*

### Directory/Path Variables
- `MLPACK_INCLUDE_DIR` — Path to the mlpack include directory (must contain `mlpack/core.hpp`)
- `ENSMALLEN_INCLUDE_DIR` — Path to the ensmallen include directory (must contain `ensmallen.hpp`)
- `CEREAL_INCLUDE_DIR` — Path to the cereal include directory (must contain `cereal/cereal.hpp`)

### Environment Variables
- `CMAKE_PREFIX_PATH` — Required by CMake to find Boost, Torch, and Armadillo.

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://gitlab.fi.muni.cz/xdusil/link-prediction.git
   cd link-prediction
   ```
2. **Create a Build Directory and Enter It**  
   ```bash
   mkdir build
   cd build
   ```
3. **Run CMake**
You must specify paths to libraries via -D flags and also set the CMake prefix path for dependencies like Boost, Torch, and Armadillo. For example:
```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="/path/to/boost;/path/to/libtorch/share/cmake/Torch;/path/to/armadillo" \
  -DMLPACK_INCLUDE_DIR="/path/to/mlpack/include" \
  -DENSMALLEN_INCLUDE_DIR="/path/to/ensmallen/include" \
  -DCEREAL_INCLUDE_DIR="/path/to/cereal/include"
```
- Adjust the paths accordingly:
   - Replace /path/to/... with the correct locations on your system.
- Make sure to include all the required libraries in your CMAKE_PREFIX_PATH.

4. **Build the Project**
If the configuration step completes successfully, you can build:
```bash
make -j4
```
This will produce an executable named LinkPredictionApp in your build directory.

5. **Run the Application**
Once the build is complete, run the executable:
```bash
./LinkPredictionApp
```

## Usage
The application requires `data_bt1.json` file to be present in the build directory. This file contains the network data in JSON format. The application reads the data, performs link prediction, and outputs the results to the console.