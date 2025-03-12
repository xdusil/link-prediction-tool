#pragma once
#include <vector>
#include <cstddef>

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
    int CONTEXT_SIZE = 2;
    int WALKS_PER_NODE = 10;
    int NUM_NEGATIVE_SAMPLES = 1;
    int EPOCHS = 15;
    int NUM_THREADS = 4;
    double LEARNING_RATE = 0.01;

    // Random Forest parameters
    int NUM_TREES = 50;
    int MIN_LEAF_SIZE = 1;
    double MIN_GAIN_SPLIT = 0.0;
    int MAX_DEPTH = 30;
    
    // Grid search parameters
    bool GRID_SEARCH_ENABLED = false;
    std::vector<std::size_t> GRID_NUM_TREES = {10, 20, 50, 100};
    std::vector<std::size_t> GRID_MIN_LEAF_SIZE = {1, 3, 5};
    std::vector<double> GRID_MIN_GAIN_SPLIT = {0.0, 1e-7, 1e-5};
    std::vector<std::size_t> GRID_MAX_DEPTH = {0, 10, 20, 30};
    double GRID_VALIDATION_SIZE = 0.25;
};

} // namespace config