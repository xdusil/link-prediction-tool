#include "tag_invokes.hpp"
#include "config/config.hpp"
#include "exceptions/exceptions.hpp"
#include "utils/validators/simple_validators.hpp"

namespace json = boost::json;
using namespace validators;

using namespace config;

/**
 * @brief Common function to set a value in a place if it is valid
 *
 * @tparam PlaceType The type of the place
 * @tparam Inner The type of the inner value
 * @param obj The JSON object
 * @param key The key of the value
 * @param place The place to set the value
 * @param validator The validation function
 * @param extractor The extractor function
 */
template <typename PlaceType, typename Inner>
void set_validated_common(
    const json::object &obj, const std::string &key, PlaceType &place,
    std::function<std::pair<bool, std::string>(const Inner &)> validator,
    std::function<std::optional<Inner>(const json::object &, const std::string &,
                                       std::function<bool(const Inner &)>,
                                       const std::string &)>
        extractor) {

    try {
        const auto &error_msg = validator(Inner{}).second;
        auto v = [&validator](const Inner &val) { return validator(val).first; };
        auto val_opt = extractor(obj, key, v, error_msg);
        if (val_opt) {
            place = *val_opt;
        }
    } catch (const std::exception &e) {
        std::throw_with_nested(
            ConfigurationException("Error while getting value for key: " + key));
    }
}

/**
 * @brief Set a value in a place if it is valid
 *
 * @tparam T The type of the value
 * @param obj The JSON object
 * @param key The key of the value
 * @param place The place to set the value
 * @param validator The validation function
 */
template <typename T>
void set_validated(const json::object &obj, const std::string &key, T &place,
                   std::function<std::pair<bool, std::string>(const T &)> validator) {
    set_validated_common<T, T>(obj, key, place, validator,
                               [](const json::object &o, const std::string &k,
                                  std::function<bool(const T &)> v,
                                  const std::string &msg) {
                                   return JsonHelper::extract_validated<T>(o, k, v, msg);
                               });
}

/**
 * @brief Set an optional value in a place if it is valid
 *
 * @tparam T The type of the value
 * @param obj The JSON object
 * @param key The key of the value
 * @param place The place to set the value
 * @param validator The validation function
 */
template <typename T>
void set_validated_opt(
    const json::object &obj, const std::string &key, std::optional<T> &place,
    std::function<std::pair<bool, std::string>(const std::optional<T> &)> validator) {
    set_validated_common<std::optional<T>, T>(
        obj, key, place, validator,
        [](const json::object &o, const std::string &k, std::function<bool(const T &)> v,
           const std::string &msg) {
            return JsonHelper::extract_validated<T>(o, k, v, msg);
        });
}

/**
 * @brief Set a vector in a place if it is valid
 *
 * @tparam T The type of the vector elements
 * @param obj The JSON object
 * @param key The key of the vector
 * @param place The place to set the vector
 * @param validator The validation function
 */
template <typename T>
void set_validated_vec(
    const json::object &obj, const std::string &key, std::vector<T> &place,
    std::function<std::pair<bool, std::string>(const std::vector<T> &)> validator) {
    set_validated_common<std::vector<T>, std::vector<T>>(
        obj, key, place, validator,
        [](const json::object &o, const std::string &k,
           std::function<bool(const std::vector<T> &)> v, const std::string &msg) {
            return JsonHelper::extract_validated_array<T>(o, k, v, msg);
        });
}

RandomForestParams tag_invoke(json::value_to_tag<RandomForestParams>,
                              const json::value &jv) {
    RandomForestParams params;
    const auto &obj = JsonHelper::parse_json_value(jv);

    set_validated<std::size_t>(obj, "num_trees", params.num_trees, is_positive{});
    set_validated<std::size_t>(obj, "min_leaf_size", params.min_leaf_size, is_positive{});
    set_validated<double>(obj, "min_gain_split", params.min_gain_split,
                          is_non_negative{});
    set_validated<std::size_t>(obj, "max_depth", params.max_depth, is_non_negative{});

    return params;
}

void tag_invoke(json::value_from_tag, json::value &jv, const RandomForestParams &params) {
    json::object obj;

    obj["num_trees"] = json::value_from(params.num_trees);
    obj["min_leaf_size"] = json::value_from(params.min_leaf_size);
    obj["min_gain_split"] = json::value_from(params.min_gain_split);
    obj["max_depth"] = json::value_from(params.max_depth);

    jv = std::move(obj);
}

GridSearchParams tag_invoke(json::value_to_tag<GridSearchParams>, const json::value &jv) {
    GridSearchParams params;
    const auto &obj = JsonHelper::parse_json_value(jv);

    set_validated_vec<std::size_t>(obj, "num_trees", params.num_trees,
                                   is_not_empty_positive_vector{});
    set_validated_vec<std::size_t>(obj, "min_leaf_size", params.min_leaf_size,
                                   is_not_empty_positive_vector{});
    set_validated_vec<double>(obj, "min_gain_split", params.min_gain_split,
                              is_not_empty_non_negative_vector{});
    set_validated_vec<std::size_t>(obj, "max_depth", params.max_depth,
                                   is_not_empty_non_negative_vector{});

    set_validated<double>(obj, "validation_size", params.validation_size,
                          is_open_unit_interval{});

    return params;
}

void tag_invoke(json::value_from_tag, json::value &jv, const GridSearchParams &params) {
    json::object obj;

    obj["num_trees"] = json::value_from(params.num_trees);
    obj["min_leaf_size"] = json::value_from(params.min_leaf_size);
    obj["min_gain_split"] = json::value_from(params.min_gain_split);
    obj["max_depth"] = json::value_from(params.max_depth);
    obj["validation_size"] = json::value_from(params.validation_size);

    jv = std::move(obj);
}

FeatureConfig tag_invoke(json::value_to_tag<FeatureConfig>, const json::value &jv) {
    FeatureConfig config;
    const auto &obj = JsonHelper::parse_json_value(jv);

    // 1. Directional Embedding Similarity
    set_validated<bool>(obj, "emb_dot_src_dst", config.emb_dot_src_dst, always_true{});
    set_validated<bool>(obj, "emb_cosine_src_dst", config.emb_cosine_src_dst, always_true{});
    set_validated<bool>(obj, "emb_l1_src_dst", config.emb_l1_src_dst, always_true{});
    set_validated<bool>(obj, "emb_l2_src_dst", config.emb_l2_src_dst, always_true{});

    // 2. Embedding Asymmetry
    set_validated<bool>(obj, "emb_src_norm", config.emb_src_norm, always_true{});
    set_validated<bool>(obj, "emb_dst_norm", config.emb_dst_norm, always_true{});
    set_validated<bool>(obj, "emb_norm_ratio", config.emb_norm_ratio, always_true{});

    // 3. Hadamard Aggregates
    set_validated<bool>(obj, "emb_hadamard_sum", config.emb_hadamard_sum, always_true{});
    set_validated<bool>(obj, "emb_hadamard_mean", config.emb_hadamard_mean, always_true{});

    // 4. Structural Graph Features
    set_validated<bool>(obj, "struct_in_degree_src", config.struct_in_degree_src, always_true{});
    set_validated<bool>(obj, "struct_out_degree_src", config.struct_out_degree_src, always_true{});
    set_validated<bool>(obj, "struct_in_degree_dst", config.struct_in_degree_dst, always_true{});
    set_validated<bool>(obj, "struct_out_degree_dst", config.struct_out_degree_dst, always_true{});
    set_validated<bool>(obj, "struct_degree_ratio", config.struct_degree_ratio, always_true{});
    set_validated<bool>(obj, "struct_common_neighbors", config.struct_common_neighbors, always_true{});
    set_validated<bool>(obj, "struct_jaccard_coefficient", config.struct_jaccard_coefficient, always_true{});
    set_validated<bool>(obj, "struct_adamic_adar_index", config.struct_adamic_adar_index, always_true{});
    set_validated<bool>(obj, "struct_preferential_attachment", config.struct_preferential_attachment, always_true{});
    set_validated<bool>(obj, "struct_resource_allocation", config.struct_resource_allocation, always_true{});
    set_validated<bool>(obj, "struct_transitive_reachability", config.struct_transitive_reachability, always_true{});
    set_validated<bool>(obj, "struct_shortest_path", config.struct_shortest_path, always_true{});
    set_validated<bool>(obj, "struct_hierarchy_diff", config.struct_hierarchy_diff, always_true{});

    // 5. Temporal Causality
    set_validated<bool>(obj, "time_avg_duration", config.time_avg_duration, always_true{});
    set_validated<bool>(obj, "time_avg_interarrival", config.time_avg_interarrival, always_true{});
    set_validated<bool>(obj, "time_regularity", config.time_regularity, always_true{});
    set_validated<bool>(obj, "time_direction_bias", config.time_direction_bias, always_true{});
    set_validated<bool>(obj, "time_initiation_order", config.time_initiation_order, always_true{});
    set_validated<bool>(obj, "time_crosscorr_peak", config.time_crosscorr_peak, always_true{});
    set_validated<bool>(obj, "time_spike_score", config.time_spike_score, always_true{});

    // 6. Bidirectional Flow
    set_validated<bool>(obj, "flow_response_time", config.flow_response_time, always_true{});
    set_validated<bool>(obj, "flow_request_ratio", config.flow_request_ratio, always_true{});
    set_validated<bool>(obj, "flow_direction_asymmetry", config.flow_direction_asymmetry, always_true{});
    set_validated<bool>(obj, "flow_causality_score", config.flow_causality_score, always_true{});

    // 7. Protocol/Port Role
    set_validated<bool>(obj, "net_protocol_role", config.net_protocol_role, always_true{});
    set_validated<bool>(obj, "net_port_role", config.net_port_role, always_true{});

    return config;
}

void tag_invoke(json::value_from_tag, json::value &jv, const FeatureConfig &config) {
    json::object obj;

    // 1. Directional Embedding Similarity
    obj["emb_dot_src_dst"] = json::value_from(config.emb_dot_src_dst);
    obj["emb_cosine_src_dst"] = json::value_from(config.emb_cosine_src_dst);
    obj["emb_l1_src_dst"] = json::value_from(config.emb_l1_src_dst);
    obj["emb_l2_src_dst"] = json::value_from(config.emb_l2_src_dst);

    // 2. Embedding Asymmetry
    obj["emb_src_norm"] = json::value_from(config.emb_src_norm);
    obj["emb_dst_norm"] = json::value_from(config.emb_dst_norm);
    obj["emb_norm_ratio"] = json::value_from(config.emb_norm_ratio);

    // 3. Hadamard Aggregates
    obj["emb_hadamard_sum"] = json::value_from(config.emb_hadamard_sum);
    obj["emb_hadamard_mean"] = json::value_from(config.emb_hadamard_mean);

    // 4. Structural Graph Features
    obj["struct_in_degree_src"] = json::value_from(config.struct_in_degree_src);
    obj["struct_out_degree_src"] = json::value_from(config.struct_out_degree_src);
    obj["struct_in_degree_dst"] = json::value_from(config.struct_in_degree_dst);
    obj["struct_out_degree_dst"] = json::value_from(config.struct_out_degree_dst);
    obj["struct_degree_ratio"] = json::value_from(config.struct_degree_ratio);
    obj["struct_common_neighbors"] = json::value_from(config.struct_common_neighbors);
    obj["struct_jaccard_coefficient"] = json::value_from(config.struct_jaccard_coefficient);
    obj["struct_adamic_adar_index"] = json::value_from(config.struct_adamic_adar_index);
    obj["struct_preferential_attachment"] = json::value_from(config.struct_preferential_attachment);
    obj["struct_resource_allocation"] = json::value_from(config.struct_resource_allocation);
    obj["struct_transitive_reachability"] = json::value_from(config.struct_transitive_reachability);
    obj["struct_shortest_path"] = json::value_from(config.struct_shortest_path);
    obj["struct_hierarchy_diff"] = json::value_from(config.struct_hierarchy_diff);

    // 5. Temporal Causality
    obj["time_avg_duration"] = json::value_from(config.time_avg_duration);
    obj["time_avg_interarrival"] = json::value_from(config.time_avg_interarrival);
    obj["time_regularity"] = json::value_from(config.time_regularity);
    obj["time_direction_bias"] = json::value_from(config.time_direction_bias);
    obj["time_initiation_order"] = json::value_from(config.time_initiation_order);
    obj["time_crosscorr_peak"] = json::value_from(config.time_crosscorr_peak);
    obj["time_spike_score"] = json::value_from(config.time_spike_score);

    // 6. Bidirectional Flow
    obj["flow_response_time"] = json::value_from(config.flow_response_time);
    obj["flow_request_ratio"] = json::value_from(config.flow_request_ratio);
    obj["flow_direction_asymmetry"] = json::value_from(config.flow_direction_asymmetry);
    obj["flow_causality_score"] = json::value_from(config.flow_causality_score);

    // 7. Protocol/Port Role
    obj["net_protocol_role"] = json::value_from(config.net_protocol_role);
    obj["net_port_role"] = json::value_from(config.net_port_role);

    jv = std::move(obj);
}

namespace config {

Config tag_invoke(json::value_to_tag<Config>, const json::value &jv) {
    Config config;
    const auto &obj = JsonHelper::parse_json_value(jv);

    // Integer parameters
    set_validated<int>(obj, "COUNT_EXTERNAL", config.COUNT_EXTERNAL, is_positive{});
    set_validated<int>(obj, "COUNT_INTERNAL", config.COUNT_INTERNAL, is_positive{});
    set_validated<int>(obj, "MAX_EDGES", config.MAX_EDGES, is_positive{});
    set_validated<int>(obj, "N_OCCURRENCES", config.N_OCCURRENCES, is_positive{});
    set_validated<int>(obj, "EPSILON", config.EPSILON, is_positive{});
    set_validated<int>(obj, "N_APPEARANCES", config.N_APPEARANCES, is_positive{});
    set_validated<int>(obj, "EPSILON_REV", config.EPSILON_REV, is_positive{});
    set_validated<int>(obj, "EMBEDDING_DIM", config.EMBEDDING_DIM, is_positive{});
    set_validated<int>(obj, "WALK_LENGTH", config.WALK_LENGTH, is_positive{});
    set_validated<int>(obj, "CONTEXT_SIZE", config.CONTEXT_SIZE, is_positive{});
    set_validated<int>(obj, "NUM_NEGATIVE_SAMPLES", config.NUM_NEGATIVE_SAMPLES,
                       is_non_negative{});
    set_validated<int>(obj, "EPOCHS", config.EPOCHS, is_positive{});
    set_validated_opt<int>(obj, "NUM_THREADS", config.NUM_THREADS, is_positive{});

    // Double parameters
    set_validated<double>(obj, "LEARNING_RATE", config.LEARNING_RATE, is_unit_interval{});
    set_validated_opt<double>(obj, "CLASSIFIER_THRESHOLD", config.CLASSIFIER_THRESHOLD,
                              is_unit_interval{});

    // String parameters
    set_validated<std::string>(
        obj, "METRIC_TO_OPTIMIZE", config.METRIC_TO_OPTIMIZE, [](const std::string &val) {
            return std::pair{val == "accuracy" || val == "f1" || val == "precision" ||
                                 val == "recall",
                             "Invalid metric: " + val +
                                 ". Valid options are: accuracy, f1, precision, recall."};
        });

    // Boolean parameters
    set_validated<bool>(obj, "USE_WEIGHTS", config.USE_WEIGHTS, always_true{});
    set_validated<bool>(obj, "USE_SCALING", config.USE_SCALING, always_true{});
    set_validated<bool>(obj, "USE_GRID_SEARCH", config.USE_GRID_SEARCH, always_true{});
    set_validated<bool>(obj, "USE_THRESHOLD_CALIBRATION",
                        config.USE_THRESHOLD_CALIBRATION, always_true{});

    // Nested objects
    set_validated<RandomForestParams>(obj, "RF_PARAMS", config.RF_PARAMS, always_true{});
    set_validated<GridSearchParams>(obj, "GRID_PARAMS", config.GRID_PARAMS,
                                    always_true{});
    set_validated<FeatureConfig>(obj, "FEATURE_CONFIG", config.FEATURE_CONFIG,
                                 always_true{});

    return config;
}

void tag_invoke(json::value_from_tag, json::value &jv, const Config &config) {
    // Create empty object first
    json::object obj;

    // Add primitive values with proper conversion
    obj["COUNT_EXTERNAL"] = json::value_from(config.COUNT_EXTERNAL);
    obj["COUNT_INTERNAL"] = json::value_from(config.COUNT_INTERNAL);
    obj["MAX_EDGES"] = json::value_from(config.MAX_EDGES);
    obj["N_OCCURRENCES"] = json::value_from(config.N_OCCURRENCES);
    obj["EPSILON"] = json::value_from(config.EPSILON);
    obj["N_APPEARANCES"] = json::value_from(config.N_APPEARANCES);
    obj["EPSILON_REV"] = json::value_from(config.EPSILON_REV);
    obj["EMBEDDING_DIM"] = json::value_from(config.EMBEDDING_DIM);
    obj["WALK_LENGTH"] = json::value_from(config.WALK_LENGTH);
    obj["CONTEXT_SIZE"] = json::value_from(config.CONTEXT_SIZE);
    obj["NUM_NEGATIVE_SAMPLES"] = json::value_from(config.NUM_NEGATIVE_SAMPLES);
    obj["EPOCHS"] = json::value_from(config.EPOCHS);
    obj["NUM_THREADS"] = json::value_from(config.NUM_THREADS);
    obj["LEARNING_RATE"] = json::value_from(config.LEARNING_RATE);
    obj["USE_WEIGHTS"] = json::value_from(config.USE_WEIGHTS);
    obj["USE_SCALING"] = json::value_from(config.USE_SCALING);
    obj["USE_GRID_SEARCH"] = json::value_from(config.USE_GRID_SEARCH);
    obj["USE_THRESHOLD_CALIBRATION"] = json::value_from(config.USE_THRESHOLD_CALIBRATION);
    obj["CLASSIFIER_THRESHOLD"] = json::value_from(config.CLASSIFIER_THRESHOLD);
    obj["METRIC_TO_OPTIMIZE"] = json::value_from(config.METRIC_TO_OPTIMIZE);

    // Add nested objects
    obj["RF_PARAMS"] = json::value_from(config.RF_PARAMS);
    obj["GRID_PARAMS"] = json::value_from(config.GRID_PARAMS);
    obj["FEATURE_CONFIG"] = json::value_from(config.FEATURE_CONFIG);

    // Assign the filled object to the output value
    jv = std::move(obj);
}
} // namespace config