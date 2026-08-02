# LinkPrediction Tool

LinkPrediction is a C++20 tool for inferring directed dependencies between IP
endpoints from flow-level network telemetry. It builds a retained directed graph
from network flows, learns directional graph embeddings, generates interpretable
pairwise features, and predicts dependencies with a binary Random Forest classifier.

The tool is designed for practical dependency analysis under incomplete
observability. It is useful when experts need to understand which systems depend
on which other systems, but only flow records are available.

Typical use cases:

- Incident response: identify systems that may depend on a compromised host.
- Impact analysis: estimate what breaks if a server fails.
- Network understanding: recover dependency structure in undocumented networks.
- Security analysis: discover hidden service relationships and lateral movement paths.
- Monitoring gap filling: infer missing dependency information when tracing is absent.

The tool is built around ranking, configurability, reproducibility, and
explainability so experts can inspect and trust the results.

## Table of Contents

- [At a Glance](#at-a-glance)
- [Why This Tool Exists](#why-this-tool-exists)
- [Architecture](#architecture)
- [Build](#build)
  - [Dependencies](#dependencies)
  - [Generic Build](#generic-build)
  - [Build With Explicit Dependency Paths](#build-with-explicit-dependency-paths)
  - [Build Options](#build-options)
- [Command Line Usage](#command-line-usage)
  - [Ground Truth Extraction](#ground-truth-extraction)
  - [Training](#training)
  - [Prediction](#prediction)
- [Input Data](#input-data)
- [Configuration](#configuration)
- [Outputs](#outputs)
  - [Training Artifacts](#training-artifacts)
  - [Prediction Artifacts](#prediction-artifacts)
- [Metrics](#metrics)
- [Explainability](#explainability)
- [Reproducibility](#reproducibility)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Academic Context](#academic-context)
- [License](#license)
- [Contact](#contact)

## At a Glance

| Area | Description |
|---|---|
| Input | JSON Lines flow records |
| Main task | Directed dependency inference, `dependent_ip -> dependency_ip` |
| Graph | Directed Boost graph over retained endpoints |
| Sampling | Structured endpoint retention plus temporal per-pair edge sampling |
| Embeddings | Directional source/destination embeddings trained with LibTorch |
| Features | Embedding, topology, temporal, observed-flow, and network-role features |
| Classifier | Binary mlpack Random Forest |
| Main output | Positive dependency predictions as CSV |
| Ranking output | Optional all-pair score CSV |
| Evaluation | Classification and ranking metrics when ground truth is supplied |
| Explainability | Local group ablation and global permutation importance |
| Reproducibility | Configs, logs, manifests, metrics JSON, score files, explanation JSONL |

## Why This Tool Exists

Flow telemetry is widely available, but it is incomplete and indirect. It tells
us who talked to whom, when, and how often. It does not directly tell us whether
one system depends on another.

This tool treats dependency inference as an expert-facing decision-support
problem:

- It predicts candidate dependencies.
- It ranks candidates so experts can inspect the most likely ones first.
- It exposes a configurable coverage/confidence tradeoff.
- It explains predictions using feature groups that map to network concepts.

The intended claim is therefore practical:

> Given flow-level telemetry, the tool infers, ranks, and explains likely endpoint
> dependencies in a configurable retained graph universe.

## Architecture

The pipeline has three operational modes: ground-truth extraction, training, and
prediction.

High-level flow:

1. Parse JSON flow records.
2. Filter invalid, blocked, or out-of-scope IPs.
3. Retain endpoints using configurable graph-size limits.
4. Sample temporal flow evidence per directed pair and temporal bucket.
5. Build a directed graph.
6. Train directional source and destination embeddings.
7. Generate pairwise features for retained directed pairs.
8. Train or load a binary Random Forest classifier.
9. Predict dependencies, produce scores, compute metrics, and write explanations.

Feature groups:

| Family | Purpose |
|---|---|
| Embedding | Directional embedding similarity, embedding asymmetry, and Hadamard aggregate evidence |
| Topology | Directed degree, common-neighbor, path, hierarchy, and attachment evidence |
| Temporal | Duration, interarrival, regularity, direction bias, initiation order, cross-correlation, and spike-timing evidence |
| Observed flow | Response time, request ratio, direction asymmetry, causality, and flow-concentration evidence |
| Network role | Protocol role, port role, and top destination-port evidence |

## Build

### Dependencies

| Dependency | Version constraint | Notes |
|---|---|---|
| [CMake](https://cmake.org/) | `>= 3.18` | Build configuration |
| C++ compiler | C++20 support, for example GCC `>= 10` or Clang `>= 10` | Linux/Unix-like toolchain |
| [Boost](https://www.boost.org/) | `>= 1.86` | Graphs, JSON support, and utilities |
| [LibTorch](https://pytorch.org/) | `>= 2.5`, CPU build | C++ API used for CPU embedding training |
| [Armadillo](https://arma.sourceforge.net/) | `>= 14.2` | Matrix representation and numerical operations |
| [mlpack](https://www.mlpack.org/) | `>= 4.5` | Random Forest classifier and ML utilities |
| [ensmallen](https://ensmallen.org/) | `>= 2.22` | Used by mlpack; compatibility depends on the selected mlpack version |
| [cereal](https://uscilab.github.io/cereal/) | `>= 1.3` | Used by mlpack; compatibility depends on the selected mlpack version |

This application is designed for Linux and other Unix-like systems.

Tested versions:

| Component | Version |
|---|---|
| CMake | 3.18.3 |
| GCC | 13.1.0 |
| Boost | 1.86.0 |
| LibTorch | 2.5.1 CPU |
| Armadillo | 14.2.1 |
| mlpack | 4.5.1 |
| ensmallen | 2.22.0 |
| cereal | 1.3.2 |

While other versions may work, these configurations have been verified for compatibility.

Known tested environments include Faculty of Informatics Linux servers such as
[Aisa](https://www.fi.muni.cz/tech/unix/aisa.html.cs) and
[Nymfe](https://www.fi.muni.cz/tech/unix/nymfe.html.cs).

### Generic Build

```bash
mkdir -p build
cd build

cmake .. \
  -DCMAKE_PREFIX_PATH="/path/to/boost;/path/to/libtorch/share/cmake/Torch;/path/to/armadillo" \
  -DMLPACK_INCLUDE_DIR="/path/to/mlpack/include" \
  -DENSMALLEN_INCLUDE_DIR="/path/to/ensmallen/include" \
  -DCEREAL_INCLUDE_DIR="/path/to/cereal/include"

make -j"$(nproc)"
```

Fresh single-config builds default to `Release` when `CMAKE_BUILD_TYPE` is not
set.

```bash
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Important CMake variables:

| Variable | Description |
|---|---|
| `CMAKE_PREFIX_PATH` | Semicolon-separated package search path for Boost, LibTorch, and Armadillo |
| `MLPACK_INCLUDE_DIR` | Include directory that contains `mlpack/core.hpp` |
| `ENSMALLEN_INCLUDE_DIR` | Include directory that contains `ensmallen.hpp` |
| `CEREAL_INCLUDE_DIR` | Include directory that contains `cereal/cereal.hpp` |

### Build With Explicit Dependency Paths

```bash
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/boost-1.86;/opt/libtorch/share/cmake/Torch;/opt/armadillo-14.2" \
  -DMLPACK_INCLUDE_DIR="/opt/mlpack-4.5/include" \
  -DENSMALLEN_INCLUDE_DIR="/opt/ensmallen-2.22/include" \
  -DCEREAL_INCLUDE_DIR="/opt/cereal/include"

make -j"$(nproc)"
```

### Build Options

Prediction-set threshold diagnostics are guarded by:

```text
LINK_PREDICTION_ENABLE_PREDICTION_THRESHOLD_DIAGNOSTICS
```

Keep this option disabled for reported metrics. When compiled in, prediction
threshold diagnostics may search thresholds using prediction labels; this is
useful for internal analysis, but not for unbiased evaluation.

## Command Line Usage

Show CLI help:

```bash
./build/LinkPredictionApp --help
```

Show build/version metadata:

```bash
./build/LinkPredictionApp --version
```

### Ground Truth Extraction

```bash
./build/LinkPredictionApp \
  -x \
  -f configs/defaultCfg.json \
  -d flows.json \
  -G ground_truth.csv \
  -v
```

Ground truth output:

```csv
src_ip,dst_ip,dependency_type
```

Known dependency types:

- `DD`: direct dependency
- `TD2`: two-stage transitive dependency
- `RR2`: two-stage remote-remote dependency
- `TD3`: three-stage transitive dependency
- `RR3`: three-stage remote-remote dependency

The extracted ground truth is generated using the configured temporal and
occurrence rules.

### Training

```bash
./build/LinkPredictionApp \
  -t \
  -f configs/defaultCfg.json \
  -c model.rf \
  -d train_flows.json \
  -g train_ground_truth.csv \
  -v
```

Useful training options:

| Option | Description |
|---|---|
| `-d, --data PATH` | Input flow JSON Lines file |
| `-c, --classifier PATH` | Output classifier path |
| `-f, --config PATH` | JSON configuration file |
| `-g, --ground-truth-in PATH` | Load existing ground truth |
| `-G, --ground-truth-out PATH` | Save calculated ground truth |
| `-b, --blocked-ips PATH` | File with blocked IPs or CIDR ranges |
| `-i, --internal-ips PATH` | File with internal IPs or CIDR ranges |
| `-F, --feature-importance` | Calculate permutation feature importance |
| `-v, --verbose` | Enable verbose timing/logging |

### Prediction

```bash
./build/LinkPredictionApp \
  -p \
  -f configs/defaultCfg.json \
  -c model.rf \
  -d test_flows.json \
  -g test_ground_truth.csv \
  -o predictions.csv \
  -s \
  -e \
  -v
```

Useful prediction options:

| Option | Description |
|---|---|
| `-c, --classifier PATH` | Trained classifier to load |
| `-d, --data PATH` | Input flow JSON Lines file |
| `-o, --predictions-out PATH` | Main prediction CSV path |
| `-g, --ground-truth-in PATH` | Optional reference labels for evaluation |
| `-s, --scores` | Write all evaluated pair scores |
| `-e, --explanations` | Write local group ablation explanations |
| `-v, --verbose` | Enable verbose timing/logging |

The main prediction CSV contains positive predictions only. Use `--scores` to
write the complete evaluated pair universe.

## Input Data

Input flow data is JSON Lines: one JSON object per line.

| Field | Required | Description |
|---|---|---|
| `sourceIPv4Address` | Yes | Source IP address |
| `destinationIPv4Address` | Yes | Destination IP address |
| `protocolIdentifier` | Yes | Transport protocol number, for example TCP `6` or UDP `17` |
| `flowStartMilliseconds`, `flowEndMilliseconds` | Forward timestamp option | Forward flow start/end |
| `biFlowStartMilliseconds`, `biFlowEndMilliseconds` | Forward timestamp option | Forward/biflow fallback start/end |
| `flowStartMilliseconds_Rev`, `flowEndMilliseconds_Rev` | Optional | Reverse flow start/end |
| `biFlowStartMilliseconds_Rev`, `biFlowEndMilliseconds_Rev` | Optional | Reverse biflow start/end |
| `sourceTransportPort` | Optional | Source port |
| `destinationTransportPort` | Optional | Destination port |

At least one forward timestamp pair is required. Reverse timestamps are used
when available by features and by ground-truth rules that require response
timing.

Example:

```json
{"sourceIPv4Address":"10.0.0.10","destinationIPv4Address":"10.0.0.20","protocolIdentifier":6,"sourceTransportPort":53000,"destinationTransportPort":443,"flowStartMilliseconds":1553069758000,"flowEndMilliseconds":1553069758864}
```

Blocked and internal IP files are plain text, one IP or CIDR range per line:

```text
10.0.0.0/8
192.168.1.10
```

## Configuration

The default configuration is:

[configs/defaultCfg.json](configs/defaultCfg.json)

Important configuration groups:

| Group | Parameters |
|---|---|
| Endpoint retention | `MAX_INTERNAL_ENDPOINTS`, `MAX_EXTERNAL_ENDPOINTS` |
| Temporal edge sampling | `MAX_EDGES_PER_PAIR_TEMPORAL_BUCKET`, `TEMPORAL_BUCKETS` |
| Ground truth | `REFERENCE_MIN_OCCURRENCES`, `TIMING_EPSILON_MS` |
| Random walks | `WALK_MIN_TARGET_APPEARANCES`, `TIMING_EPSILON_MS`, `TIMING_REVERSE_EPSILON_MS`, `WALK_LENGTH`, `WALKS_PER_VERTEX` |
| Embeddings | `EMBEDDING_DIM`, `CONTEXT_SIZE`, `NUM_NEGATIVE_SAMPLES`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE` |
| Classifier | `USE_CLASS_WEIGHTS`, `USE_FEATURE_SCALING`, `USE_GRID_SEARCH`, `USE_THRESHOLD_CALIBRATION`, `RF_PARAMS`, `GRID_PARAMS`, `METRIC_TO_OPTIMIZE` |
| Reproducibility | `SEED`, `NUM_THREADS`, `WRITE_RUN_MANIFESTS` |
| Features | `FEATURE_CONFIG` |
| Service enrichment | `SERVICE_CONFIG` |

The current cap is per directed pair and
per temporal bucket:

```json
{
  "MAX_EDGES_PER_PAIR_TEMPORAL_BUCKET": 4,
  "TEMPORAL_BUCKETS": 4
}
```

## Outputs

### Training Artifacts

| Artifact | Description |
|---|---|
| `model.rf` | Serialized binary Random Forest classifier |
| `model.rf.feature_baselines.json` | Feature names and training medians for explanations |
| `model.rf.run_manifest.json` | Training metadata when manifests are enabled |

### Prediction Artifacts

| Artifact | Description |
|---|---|
| `predictions.csv` | Positive predictions only |
| `predictions.csv.scores.csv` | All evaluated pair scores when `--scores` is used |
| `predictions.csv.explanations.jsonl` | Per-pair local explanations when `--explanations` is used |
| `predictions.csv.metrics.json` | Metrics when ground truth is supplied |
| `predictions.csv.run_manifest.json` | Prediction metadata when manifests are enabled |

Main prediction columns:

```csv
dependent_ip,dependency_ip,score
```

When service enrichment is enabled:

```csv
dependent_ip,dependency_ip,score,service,service_conf,service_topk
```

All-pair score columns:

```csv
dependent_ip,dependency_ip,score,predicted_label,label
```

Explanation JSONL contains one JSON object per evaluated pair and can therefore
be large.

## Metrics

When ground truth is supplied in prediction mode, the tool reports:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- average precision
- mean reciprocal rank
- precision@10 / recall@10
- precision@50 / recall@50
- precision@100 / recall@100

Score-based ranking metrics treat label `1` as the positive dependency class.

## Explainability

The tool supports two explainability modes:

- Global permutation feature importance during training with `--feature-importance`.
- Local group ablation during prediction with `--explanations`.

Training writes:

```text
<classifier>.feature_baselines.json
```

This sidecar stores exact feature names and training medians. Prediction with
`--explanations` loads it and validates that feature names and order match the
active feature configuration.

Local group contribution:

```text
contribution = original_dependency_score - score_after_group_replaced_by_training_medians
```

Feature groups:

- `embedding`
- `topology`
- `temporal`
- `observed_flow`
- `network`

These are sensitivity explanations, not SHAP values and not causal guarantees.
They are intentionally group-based because individual engineered features can be
correlated. Group ablation is more stable and easier for domain experts to interpret.

## Reproducibility

For serious experiments, preserve:

- exact config JSON,
- command log,
- train and prediction logs,
- model file,
- run manifests,
- metrics JSON,
- score CSV,
- explanation JSONL if explanations were enabled,
- ground-truth CSV and the config used to generate it.

## Performance Notes

- Use `Release` or `RelWithDebInfo` builds for measured runs.
- Use `NUM_THREADS` to control thread count.
- Increasing `MAX_INTERNAL_ENDPOINTS` or `MAX_EXTERNAL_ENDPOINTS` increases the retained graph and evaluated pair count.
- Increasing `MAX_EDGES_PER_PAIR_TEMPORAL_BUCKET` or `TEMPORAL_BUCKETS` retains more temporal evidence.
- `--explanations` runs additional classifier passes, one per active feature group.
- `--feature-importance` can be expensive because it performs repeated feature permutations.

## Troubleshooting

| Problem | Suggested fix |
|---|---|
| CMake finds an older Boost | Pass Boost 1.86 through `CMAKE_PREFIX_PATH` |
| Explanations fail with missing baseline | Retrain the classifier so `<classifier>.feature_baselines.json` is created |
| Explanations reject the baseline | Ensure prediction uses the same feature configuration and feature order as training |
| Score-based metrics fail or are unavailable | Check that ground truth and positive-class scores are both available and have matching dimensions |
| F1 is unexpectedly low | Check retained graph size, ground-truth parameters, classifier threshold, and positive coverage |
| Results differ across runs | Check `SEED`, `NUM_THREADS`, build type, config, and generated ground truth |

## Documentation

The repository includes a `Doxyfile` for generating API documentation with
[Doxygen](https://www.doxygen.nl/). Graphviz is recommended for diagrams.

To generate the documentation:

1. Install Doxygen and Graphviz:
   ```bash
   # For Ubuntu/Debian-based systems
   sudo apt-get install doxygen graphviz
   ```
2. Run Doxygen from the project root directory:
   ```bash
   doxygen
   ```
   This will create a `docs/html` directory containing the generated documentation.
3. View the documentation:
   Open `docs/html/index.html` in your web browser to view the documentation.

The generated documentation includes:
- Class and function descriptions
- Namespace descriptions
- Diagrams (class hierarchy, collaboration)
- Detailed descriptions of API functions and parameters

## Academic Context

This project originates from the Bachelor's thesis:

**Automatic Identification of Dependencies in Computer Network**

- Author: Jakub Dusil
- Institution: Masaryk University, Faculty of Informatics
- Thesis URL: https://is.muni.cz/th/t0dn4/

The implementation has continued beyond the original thesis toward a more
stable, configurable, and explainable dependency inference tool.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact

- Jakub Dusil
- Email: 536566@mail.muni.cz
- GitHub: [xdusil](https://github.com/xdusil)
