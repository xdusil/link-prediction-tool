#pragma once

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
};