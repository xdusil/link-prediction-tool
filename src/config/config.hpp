#pragma once
#include "classifier/generic/IRandomForestClassifier.hpp"
#include "generators/feature/FeatureConfig.hpp"
#include "service/EdgeServiceClassifier.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace config {

/**
 * @brief Configuration parameters
 */
struct Config {
    int COUNT_EXTERNAL = 100;
    int COUNT_INTERNAL = 50;
    int MAX_EDGES_PER_PAIR_TEMPORAL_BUCKET = 4;
    int TEMPORAL_BUCKETS = 4;
    int N_OCCURRENCES = 10;
    int EPSILON = 1000;
    int N_APPEARANCES = 10;
    int EPSILON_REV = 1000;
    int EMBEDDING_DIM = 64;
    int WALK_LENGTH = 5;
    int WALKS_PER_VERTEX = 10;
    int BATCH_SIZE = 32;
    int CONTEXT_SIZE = 4;
    int NUM_NEGATIVE_SAMPLES = 1;
    int EPOCHS = 15;
    double LEARNING_RATE = 0.01;
    std::uint32_t SEED = 123;
    std::optional<int> NUM_THREADS;
    std::optional<double> CLASSIFIER_THRESHOLD;
    std::string METRIC_TO_OPTIMIZE = "f1";

    bool USE_WEIGHTS = false;
    bool USE_SCALING = true;
    bool USE_GRID_SEARCH = false;
    bool USE_THRESHOLD_CALIBRATION = false;
    bool WRITE_RUN_MANIFESTS = true;

    RandomForestParams RF_PARAMS = {200, 5, 1e-7, 0};

    GridSearchParams GRID_PARAMS = {
        {70, 200, 500}, {3, 5, 10}, {0.0, 1e-7, 1e-5}, {0, 30, 45}, 0.25};

    FeatureConfig FEATURE_CONFIG{};

    service::ServiceClassificationConfig SERVICE_CONFIG{};
};

/**
 * @brief Load configuration from a file
 *
 * @param filename The name of the file to load the configuration from
 * @return The loaded configuration
 * @throws ConfigurationException if the file cannot be read or
 *                                   the parsing fails or
 *                                   the validation fails
 */
Config load(const std::string& filename);

} // namespace config
