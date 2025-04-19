#pragma once
#include "classifier/generic/IRandomForestClassifier.hpp"
#include "generators/feature/FeatureConfig.hpp"
#include <cstddef>
#include <optional>
#include <vector>

namespace config {

/**
 * @brief Configuration parameters
 */
struct Config {
    int COUNT_EXTERNAL = 100;
    int COUNT_INTERNAL = 50;
    int MAX_EDGES = 500;
    int N_OCCURRENCES = 10;
    int EPSILON = 1000;
    int N_APPEARANCES = 10;
    int EPSILON_REV = 1000;
    int EMBEDDING_DIM = 64;
    int WALK_LENGTH = 5;
    int CONTEXT_SIZE = 4;
    int NUM_NEGATIVE_SAMPLES = 1;
    int EPOCHS = 15;
    double LEARNING_RATE = 0.01;
    std::optional<int> NUM_THREADS;
    std::optional<double> CLASSIFIER_THRESHOLD;
    std::string METRIC_TO_OPTIMIZE = "f1";

    bool USE_WEIGHTS = false;
    bool USE_SCALING = true;
    bool USE_GRID_SEARCH = false;
    bool USE_THRESHOLD_CALIBRATION = false;

    RandomForestParams RF_PARAMS = {50, 1, 0.0, 30};

    GridSearchParams GRID_PARAMS = {
        {10, 20, 50, 100}, {1, 3, 5}, {0.0, 1e-7, 1e-5}, {0, 10, 20, 30}, 0.25};

    FeatureConfig FEATURE_CONFIG{};
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
Config load(const std::string &filename);

} // namespace config