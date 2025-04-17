# LinkPredictionApp

## Overview
LinkPredictionApp is a high‑performance C++ application for detecting dependencies between devices in a network. It combines graph‑based analysis, network flow processing, and machine learning to identify and predict device communication dependencies with high accuracy.

## Features
- **Three Operational Modes**: Training, prediction, and ground truth extraction  
- **Graph‑based Network Analysis**: Build network graphs from raw flow data  
- **Machine Learning for Dependency Prediction**: Uses Random Forest classification with feature engineering  
- **Highly Configurable**: Runtime configuration via JSON config files or command‑line options  
- **Performance Optimized**: Multi‑threaded operation to handle large-scale networks  
- **Comprehensive Evaluation**: Built‑in metrics for model performance assessment  

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Dependencies](#dependencies)  
  - [Required](#required)  
  - [Directory/Path Variables](#directorypath-variables)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Command‑Line Interface](#command-line-interface)  
  - [Training Mode](#training-mode)  
  - [Prediction Mode](#prediction-mode)  
  - [Ground Truth Extraction Mode](#ground-truth-extraction-mode)  
- [Configuration](#configuration)  
  - [Example Configuration File](#example-configuration-file)  
  - [Example Blocked or Internal IPs File](#example-blocked-or-internal-ips-file)
  - [Example Flow Data File](#example-flow-data-file)
- [Input/Output File Formats](#inputoutput-file-formats)  
- [Performance Considerations](#performance-considerations)

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
make -j$(nproc)
```

# Usage

### Command-Line Interface
The application supports three main modes:

```text
  -t, --training       Training mode: build & train a classifier
  -p, --prediction     Prediction mode: use an existing classifier
  -x, --extract        Ground truth extraction mode
  -c, --config PATH    JSON configuration file
  -v, --verbose        Enable detailed output & timing
```

For a full list of options, run:
```bash
./LinkPredictionApp --help
```

### Training Mode
Build and train a dependency prediction model.

```bash
./LinkPredictionApp -t \
  --data flow_data.json \
  --classifier model.bin \
  [--ground-truth-in truth.csv] \
  [--ground-truth-out truth_out.csv] \
  [--blocked-ips blocked.txt] \
  [--internal-ips internal.txt] \
  [--config config.json] \
  [--verbose]
```

- `--data PATH` *(required)*: Input flow data (JSON)  
- `--classifier PATH` *(required)*: Where to save the trained model  
- `--ground-truth-in PATH`: Use existing ground truth instead of recalculating  
- `--ground-truth-out PATH`: Save extracted ground truth  
- `--blocked-ips PATH`: List of IPs to ignore (TXT)  
- `--internal-ips PATH`: List of internal network IPs (TXT)  
- `--config PATH`: JSON configuration file  
- `--verbose`: Detailed output & timing information  

### Prediction Mode
Use a pre‑trained model to predict dependencies.

```bash
./LinkPredictionApp -p \
  --classifier model.bin \
  --data flow_data.json \
  --predictions-out preds.csv \
  [--ground-truth-in truth.csv] \
  [--blocked-ips blocked.txt] \
  [--internal-ips internal.txt] \
  [--config config.json] \
  [--verbose]
```

- `--classifier PATH` *(required)*: Load classifier model  
- `--data PATH` *(required)*: Input flow data (JSON)  
- `--predictions-out PATH` *(required)*: Save predictions (CSV)  
- Other options as in Training Mode  

### Ground Truth Extraction Mode
Extract ground truth dependencies without training.

```bash
./LinkPredictionApp -x \
  --data flow_data.json \
  --ground-truth-out truth.csv \
  [--blocked-ips blocked.txt] \
  [--config config.json] \
  [--verbose]
```

- `--data PATH` *(required)*: Input flow data (JSON)  
- `--ground-truth-out PATH` *(required)*: Save ground truth (CSV)  
- Optional: `--blocked-ips`, `--config`, `--verbose`  

## Configuration
The application can be configured via a JSON file passed with `--config`. This controls pipeline behavior, embedding generation, and classifier settings.

### Example Configuration File
```json
{
    "COUNT_EXTERNAL": 100,
    "COUNT_INTERNAL": 50,
    "MAX_EDGES": 500,
    "N_OCCURRENCES": 10,
    "EPSILON": 1000,
    "N_APPEARANCES": 10,
    "EPSILON_REV": 1000,
    "EMBEDDING_DIM": 64,
    "WALK_LENGTH": 5,
    "CONTEXT_SIZE": 4,
    "NUM_NEGATIVE_SAMPLES": 1,
    "EPOCHS": 15,
    "NUM_THREADS": 30,
    "LEARNING_RATE": 0.01,
    "CLASSIFIER_THRESHOLD": 0.5,
    "USE_WEIGHTS": false,
    "USE_SCALING": true,
    "USE_GRID_SEARCH": false,
    "USE_THRESHOLD_CALIBRATION": false,
    "RF_PARAMS": {
        "num_trees": 50,
        "min_leaf_size": 1,
        "min_gain_split": 0.0,
        "max_depth": 30
    },
    "GRID_PARAMS": {
        "num_trees": [
            10,
            20,
            50,
            100
        ],
        "min_leaf_size": [
            1,
            3,
            5
        ],
        "min_gain_split": [
            0.0,
            1e-7,
            1e-5
        ],
        "max_depth": [
            0,
            10,
            20,
            30
        ],
        "validation_size": 0.25
    },
    "FEATURE_CONFIG": {
        "cosine_similarity": false,
        "euclidean_distance": false,
        "dot_product": false,
        "hadamard_sum": false,
        "hadamard_mean": false,
        "l1_distance": false,
        "common_neighbors": false,
        "jaccard_coefficient": false,
        "node_degree": false,
        "embed_std": false,
        "adamic_adar": false,
        "preferential_attachment": false,
        "resource_allocation": false,
        "embedding_ratio": false,
        "embedding_abs_mean": false,
        "element_wise_product": true
    }
}
```

### Example blocked or internal IPs file
```text
4.122.55.21/32
10.1.4.46
10.1.4.47
```

### Example flow data file
```json
{"biFlowEndMilliseconds":1553069758864,"biFlowStartMilliseconds":1553069758864,"destinationIPv4Address":"9.66.11.12","destinationTransportPort":1914,"flowEndMilliseconds":1553069758864,"flowStartMilliseconds":1553069758864,"protocolIdentifier":6,"sourceIPv4Address":"4.122.55.221","sourceTransportPort":49581, "timestamp":1553069758864}
{"biFlowEndMilliseconds":1553069758864,"biFlowStartMilliseconds":1553069758864,"destinationIPv4Address":"9.66.11.12","destinationTransportPort":801,"flowEndMilliseconds":1553069758864,"flowStartMilliseconds":1553069758864,"protocolIdentifier":6,"sourceIPv4Address":"4.122.55.221","sourceTransportPort":49581,"timestamp":1553069758864}
```

## Input/Output File Formats
- **Input Flow Data**: JSON array of flow records  
- **Blocked IPs**: Plain‑text, one IP or CIDR per line (TXT) 
- **Internal IPs**: Plain‑text, one IP or CIDR per line (TXT) 
- **Ground Truth & Predictions**: CSV with columns `src_ip,dst_ip,type`

## Performance Considerations
- If `NUM_THREADS` is not set, the application will use all available CPU cores
- It is recommended to use OpenMP for multi-threading - ensure it is installed on your system.
- Use `--verbose` to monitor processing times and identify bottlenecks.
