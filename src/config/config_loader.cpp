#include "config_loader.hpp"
#include "../exceptions/exceptions.hpp"
#include "../io/FileReader.hpp"
#include "../json/JsonHelper.hpp"
#include "boost/json/error.hpp"
#include <cstddef>
#include <exception>
#include <fstream>
#include <sstream>

namespace config {

namespace {

Config parse_json(const std::string &json_content) {
    Config config;

    try {
        json::object json_obj = JsonHelper::parse_json(json_content);

        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "COUNT_EXTERNAL"))
            config.COUNT_EXTERNAL = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "COUNT_INTERNAL"))
            config.COUNT_INTERNAL = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "MAX_EDGES"))
            config.MAX_EDGES = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "N_OCCURRENCES"))
            config.N_OCCURRENCES = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "EPSILON"))
            config.EPSILON = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "N_APPEARANCES"))
            config.N_APPEARANCES = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "EPSILON_REV"))
            config.EPSILON_REV = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "EMBEDDING_DIM"))
            config.EMBEDDING_DIM = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "WALK_LENGTH"))
            config.WALK_LENGTH = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "CONTEXT_SIZE"))
            config.CONTEXT_SIZE = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "WALKS_PER_NODE"))
            config.WALKS_PER_NODE = static_cast<int>(*val);
        if (auto val =
                JsonHelper::extract_value<int64_t>(json_obj, "NUM_NEGATIVE_SAMPLES"))
            config.NUM_NEGATIVE_SAMPLES = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "EPOCHS"))
            config.EPOCHS = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "NUM_THREADS"))
            config.NUM_THREADS = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<double>(json_obj, "LEARNING_RATE"))
            config.LEARNING_RATE = *val;
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "NUM_TREES"))
            config.NUM_TREES = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "MIN_LEAF_SIZE"))
            config.MIN_LEAF_SIZE = static_cast<int>(*val);
        if (auto val = JsonHelper::extract_value<double>(json_obj, "MIN_GAIN_SPLIT"))
            config.MIN_GAIN_SPLIT = *val;
        if (auto val = JsonHelper::extract_value<int64_t>(json_obj, "MAX_DEPTH"))
            config.MAX_DEPTH = static_cast<int>(*val);

        // Load grid search enabled flag
        if (auto val = JsonHelper::extract_value<bool>(json_obj, "GRID_SEARCH_ENABLED"))
            config.GRID_SEARCH_ENABLED = *val;

        // Load grid search parameters
        if (auto arr = JsonHelper::extract_array<int64_t>(json_obj, "GRID_NUM_TREES")) {
            config.GRID_NUM_TREES.clear();
            for (const auto &elem : *arr) {
                config.GRID_NUM_TREES.push_back(static_cast<std::size_t>(elem));
            }
        }

        if (auto arr =
                JsonHelper::extract_array<int64_t>(json_obj, "GRID_MIN_LEAF_SIZE")) {
            config.GRID_MIN_LEAF_SIZE.clear();
            for (const auto &elem : *arr) {
                config.GRID_MIN_LEAF_SIZE.push_back(static_cast<std::size_t>(elem));
            }
        }

        if (auto arr =
                JsonHelper::extract_array<double>(json_obj, "GRID_MIN_GAIN_SPLIT")) {
            config.GRID_MIN_GAIN_SPLIT = *arr;
        }

        if (auto arr = JsonHelper::extract_array<int64_t>(json_obj, "GRID_MAX_DEPTH")) {
            config.GRID_MAX_DEPTH.clear();
            for (const auto &elem : *arr) {
                config.GRID_MAX_DEPTH.push_back(static_cast<std::size_t>(elem));
            }
        }

        if (auto val =
                JsonHelper::extract_value<double>(json_obj, "GRID_VALIDATION_SIZE"))
            config.GRID_VALIDATION_SIZE = *val;

    } catch (const std::exception &e) {
        throw ConfigurationException("Error parsing config file: " +
                                     std::string(e.what()));
    }

    return config;
}

void validate(const Config &config) {
    if (config.COUNT_EXTERNAL <= 0)
        throw ConfigurationException("COUNT_EXTERNAL must be positive");
    if (config.COUNT_INTERNAL <= 0)
        throw ConfigurationException("COUNT_INTERNAL must be positive");
    if (config.MAX_EDGES <= 0)
        throw ConfigurationException("MAX_EDGES must be positive");
    if (config.N_OCCURRENCES <= 0)
        throw ConfigurationException("N_OCCURRENCES must be positive");
    if (config.EPSILON <= 0)
        throw ConfigurationException("EPSILON must be positive");
    if (config.N_APPEARANCES <= 0)
        throw ConfigurationException("N_APPEARANCES must be positive");
    if (config.EPSILON_REV <= 0)
        throw ConfigurationException("EPSILON_REV must be positive");
    if (config.EMBEDDING_DIM <= 0)
        throw ConfigurationException("EMBEDDING_DIM must be positive");
    if (config.WALK_LENGTH <= 0)
        throw ConfigurationException("WALK_LENGTH must be positive");
    if (config.CONTEXT_SIZE <= 0)
        throw ConfigurationException("CONTEXT_SIZE must be positive");
    if (config.WALKS_PER_NODE <= 0)
        throw ConfigurationException("WALKS_PER_NODE must be positive");
    if (config.NUM_NEGATIVE_SAMPLES < 0)
        throw ConfigurationException("NUM_NEGATIVE_SAMPLES must be non-negative");
    if (config.EPOCHS <= 0)
        throw ConfigurationException("EPOCHS must be positive");
    if (config.NUM_THREADS <= 0)
        throw ConfigurationException("NUM_THREADS must be positive");
    if (config.LEARNING_RATE <= 0 || config.LEARNING_RATE > 1)
        throw ConfigurationException("LEARNING_RATE must be between 0 and 1");
    if (config.NUM_TREES <= 0)
        throw ConfigurationException("NUM_TREES must be positive");
    if (config.MIN_LEAF_SIZE <= 0)
        throw ConfigurationException("MIN_LEAF_SIZE must be positive");
    if (config.MIN_GAIN_SPLIT < 0)
        throw ConfigurationException("MIN_GAIN_SPLIT must be non-negative");
    if (config.MAX_DEPTH < 0)
        throw ConfigurationException("MAX_DEPTH must be non-negative");
}
} // anonymous namespace

Config load(const std::string &filename) {
    FileReader reader(filename);

    std::string json_content;

    // Read the entire file
    try {
        reader.read_all(json_content);
    } catch (const std::exception &e) {
        std::throw_with_nested(ConfigurationException("Could not read config file"));
    }

    // Parse JSON
    Config config = parse_json(json_content);

    // Validate
    validate(config);

    return config;
}
} // namespace config