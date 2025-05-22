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
  - [Platform Support](#platform-support)
  - [Tested Versions](#tested-versions)  
  - [CMake Configuration Variables](#cmake-configuration-variables)  
    - [Required Variables](#required-variables)  
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Build steps](#build-steps)
  - [Troubleshooting](#troubleshooting)
- [Usage](#usage)  
  - [Command‑Line Interface](#command-line-interface)  
  - [Training Mode](#training-mode)  
  - [Prediction Mode](#prediction-mode)  
  - [Ground Truth Extraction Mode](#ground-truth-extraction-mode)  
- [Configuration](#configuration)  
  - [Example Configuration File](#example-configuration-file)  
- [Input/Output File Formats](#inputoutput-file-formats) 
  - [Flow Data Fields](#flow-data-fields)
  - [Example Flow Data File](#example-flow-data-file)
  - [Example Blocked or Internal IPs File](#example-blocked-or-internal-ips-file)
  - [Example Ground Truth File](#example-ground-truth-file)
- [Performance Considerations](#performance-considerations)
- [Documentation](#documentation)  
  - [Generating Documentation](#generating-documentation)
  - [For ZIP Submission Reviewers](#for-zip-submission-reviewers)
- [Academic Research](#academic-research)
- [License](#license)
- [Contact](#contact)


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

### Platform Support
This application is primarily designed for Linux/Unix-like systems.

### Tested Versions
The application was tested using the following versions:
- CMake 3.18.3
- GCC 13.1.0
- Boost 1.86.0
- LibTorch 2.5.1 (CPU)
- Armadillo 14.2.1
- mlpack 4.5.1
- ensmallen 2.22.0
- cereal 1.3.2

While other versions may work, these configurations have been verified for compatibility.

The software has been successfully tested on the following computing environments:
- [Aisa](https://www.fi.muni.cz/tech/unix/aisa.html.cs) (Faculty of Informatics, Masaryk University)
- [Nymfe01](https://www.fi.muni.cz/tech/unix/nymfe.html.cs) (Faculty of Informatics, Masaryk University)

### CMake Configuration Variables
All dependency paths must be provided as command-line arguments to CMake using `-D` flags:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="/path/to/boost;/path/to/libtorch/share/cmake/Torch;/path/to/armadillo" \
  -DMLPACK_INCLUDE_DIR="/path/to/mlpack/include" \
  -DENSMALLEN_INCLUDE_DIR="/path/to/ensmallen/include" \
  -DCEREAL_INCLUDE_DIR="/path/to/cereal/include"
```

#### Required Variables
- `CMAKE_PREFIX_PATH` — Path list where CMake searches for Boost, Torch, and Armadillo packages (semicolon-separated)
- `MLPACK_INCLUDE_DIR` — Path to the mlpack include directory (must contain `mlpack/core.hpp`)
- `ENSMALLEN_INCLUDE_DIR` — Path to the ensmallen include directory (must contain `ensmallen.hpp`)
- `CEREAL_INCLUDE_DIR` — Path to the cereal include directory (must contain `cereal/cereal.hpp`)

## Installation

### Prerequisites
Ensure all [dependencies](#dependencies) are installed on your system before proceeding.

### Build steps

1. **Clone the Repository**  
   ```bash
   # If you already have the repository, skip this step
   # Otherwise, clone it using:
   git clone https://github.com/xdusil/link-prediction-tool.git
   cd link-prediction-tool
   ```
   
2. **Create a Build Directory and Enter It**  
   ```bash
   mkdir build
   cd build
   ```

3. **Run CMake**

   You must specify paths to libraries via -D flags and also set the CMake prefix path for dependencies like Boost, Torch, and Armadillo. See the [CMake Configuration Variables](#cmake-configuration-variables) section for details.

   For example:
    ```bash
    cmake .. \
    -DCMAKE_PREFIX_PATH="/path/to/boost;/path/to/libtorch/share/cmake/Torch;/path/to/armadillo" \
    -DMLPACK_INCLUDE_DIR="/path/to/mlpack/include" \
    -DENSMALLEN_INCLUDE_DIR="/path/to/ensmallen/include" \
    -DCEREAL_INCLUDE_DIR="/path/to/cereal/include"
    ```
  - Adjust the paths accordingly:
     - Replace /path/to/... with the correct locations on your system.
  - Make sure to include all the required libraries in your `CMAKE_PREFIX_PATH`.

4. **Build the Project**

    If the configuration step completes successfully, you can build:
    ```bash
    make -j$(nproc)
    ```

5. **Verify the Build**
   
   After the build completes, you should see an executable named `LinkPredictionApp` in the `build` directory. Try running it:
    ```bash
    ./LinkPredictionApp --help
    ```
   This should display the help message with available options.

### Troubleshooting
- **CMake can't find a package**:  
  If CMake fails to find a package, ensure that the paths provided in `CMAKE_PREFIX_PATH` are correct and that the required libraries are installed at those locations.
- **Missing header files**:
  If you encounter issues with missing header files, check the include directory paths provided in `MLPACK_INCLUDE_DIR`, `ENSMALLEN_INCLUDE_DIR`, and `CEREAL_INCLUDE_DIR` are correct and ensure  that they contain the expected header files.

# Usage

### Command-Line Interface
The application supports three main modes:

```text
  -t, --training       Training mode: build & train a classifier
  -p, --prediction     Prediction mode: use an existing classifier
  -x, --extract        Ground truth extraction mode
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
The application can be configured via a JSON file passed with `--config`. This controls pipeline behavior, embedding generation, classifier settings and feature selection.

For detailed information about configuration parameters, explanations of their functionality, and how they affect the system's behavior, please refer to the thesis mentioned in the [Academic Research](#academic-research) section.

### Example Configuration File
```json
{
    "COUNT_EXTERNAL": 100,
    "COUNT_INTERNAL": 50,
    "MAX_EDGES": 500,
    "N_OCCURRENCES": 10,
    "N_APPEARANCES": 10,
    "EPSILON": 1000,
    "EPSILON_REV": 1000,
    "EMBEDDING_DIM": 64,
    "WALK_LENGTH": 5,
    "CONTEXT_SIZE": 4,
    "NUM_NEGATIVE_SAMPLES": 1,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.01,
    "USE_WEIGHTS": true,
    "USE_SCALING": true,
    "USE_GRID_SEARCH": false,
    "USE_THRESHOLD_CALIBRATION": false,
    "CLASSIFIER_THRESHOLD": 0.5,
    "METRIC_TO_OPTIMIZE": "f1",
    "RF_PARAMS": {
        "num_trees": 50,
        "min_leaf_size": 1,
        "min_gain_split": 0.0,
        "max_depth": 30
    },
    "GRID_PARAMS": {
        "num_trees": [
            40,
            50,
            70
        ],
        "min_leaf_size": [
            1,
            3,
            5
        ],
        "min_gain_split": [
            0E0,
            1E-7,
            1E-5
        ],
        "max_depth": [
            0,
            30,
            45
        ],
        "validation_size": 2.5E-1
    },
    "FEATURE_CONFIG": {
        "cosine_similarity": true,
        "dot_product": false,
        "l1_distance": false,
        "l2_distance": true,
        "embedding_std": false,
        "embedding_abs_mean": false,
        "embedding_norm_ratio": true,
        "hadamard_product_sum": true,
        "hadamard_product_mean": false,
        "hadamard_product_components": false,
        "common_neighbors_count": true,
        "jaccard_coefficient": true,
        "adamic_adar_index": true,
        "preferential_attachment": true,
        "resource_allocation_index": false,
        "node_degree": true
    }
}
```

## Input/Output File Formats
- **Input Flow Data**: JSON file of flow records  
- **Blocked IPs**: Plain‑text, one IP or CIDR per line (TXT) 
- **Internal IPs**: Plain‑text, one IP or CIDR per line (TXT) 
- **Ground Truth**: CSV file with columns `src_ip`, `dst_ip`, `dependency_type`
- **Predictions**: CSV with columns `ip1`, `ip2`, that contains the predicted dependencies between IP addresses.

### Flow Data Fields
The application processes network flow records with the following fields:

- **IP Addresses** (required):
  - `sourceIPv4Address`: Source IP address
  - `destinationIPv4Address`: Destination IP address

- **Transport Ports** (optional):
  - `sourceTransportPort`: Source port
  - `destinationTransportPort`: Destination port
  
- **Protocol** (required):
  - `protocolIdentifier`: Protocol number (e.g., 6 for TCP, 17 for UDP)

- **Timing Information** (at least one pair required):
  - Forward flow: `flowStartMilliseconds` and `flowEndMilliseconds`
  - Bidirectional flow: `biFlowStartMilliseconds` and `biFlowEndMilliseconds`
  - Reverse flow: `flowStartMilliseconds_Rev` and `flowEndMilliseconds_Rev`
  - Reverse bidirectional flow: `biFlowStartMilliseconds_Rev` and `biFlowEndMilliseconds_Rev`

The application handles different flow types as follows:

- For training and prediction, it processes unidirectional flows normally, and only considers reverse flows when reverse data is explicitly available
- For ground truth extraction, all flows are treated as bidirectional - if dedicated reverse fields aren't present, it falls back to using bidirectional fields, and if those aren't available, it uses forward flow timestamps as a last resort to ensure complete dependency analysis

### Example flow data file
```json
{"biFlowEndMilliseconds":1553069758864,"biFlowStartMilliseconds":1553069758864,"destinationIPv4Address":"9.66.11.12","destinationTransportPort":1914,"flowEndMilliseconds":1553069758864,"flowStartMilliseconds":1553069758864,"protocolIdentifier":6,"sourceIPv4Address":"4.122.55.221","sourceTransportPort":49581, "timestamp":1553069758864}
{"biFlowEndMilliseconds":1553069758864,"biFlowStartMilliseconds":1553069758864,"destinationIPv4Address":"9.66.11.12","destinationTransportPort":801,"flowEndMilliseconds":1553069758864,"flowStartMilliseconds":1553069758864,"protocolIdentifier":6,"sourceIPv4Address":"4.122.55.221","sourceTransportPort":49581,"timestamp":1553069758864}
```

### Example blocked or internal IPs file
```text
4.122.55.21/32
10.1.4.46
10.1.4.47
```

### Example ground truth file
```csv
src_ip,dst_ip,dependency_type
77.51.161.30,9.66.11.14,DD
77.51.161.30,9.66.11.12,DD
```

## Performance Considerations
- If `NUM_THREADS` is not set, the application will use all available CPU cores
- It is recommended to use OpenMP for multi-threading - ensure it is installed on your system.
- Use `--verbose` to monitor processing times and identify bottlenecks.

## Documentation

This project includes comprehensive API documentation generated with Doxygen.

### Generating Documentation

To generate the documentation:

1. **Install Doxygen and Graphviz**:
   ```bash
   # For Ubuntu/Debian-based systems
   sudo apt-get install doxygen graphviz
    ```
2. **Run Doxygen** from the project root directory:
   ```bash
   doxygen
   ```
   This will create a `docs/html` directory containing the generated documentation.
3. **View the Documentation**:
   Open `docs/html/index.html` in your web browser to view the documentation.

The generated documentation includes:
- Class and function descriptions
- Namespace descriptions
- Diagrams (class hierarchy, collaboration)
- Detailed descriptions of API functions and parameters

### For ZIP Submission Reviewers
If you're reviewing a ZIP submission, the complete documentation is included in the `docs` directory. You can view it by opening `docs/html/index.html` in your web browser.

## Academic Research

This software was developed as part of a Bachelor's thesis:

**"Automatic Identification of Dependencies in Computer Network"**  
Author: Jakub Dusil  
Institution: Masaryk University, Faculty of Informatics  
Available from: https://is.muni.cz/th/t0dn4/

The thesis provides comprehensive documentation about:
- Theoretical foundations of network dependency detection
- Description of the algorithm and its implementation
- Detailed explanation of all configuration parameters and their effects
- Experimental evaluation and performance analysis

Users seeking a deeper understanding of the algorithm and parameter tuning are encouraged to consult the thesis for detailed explanations.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions, suggestions, or contributions:

**Jakub Dusil**
- Email: 536566@mail.muni.cz
- GitHub: [xdusil](https://github.com/xdusil)
