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

    // Process all boolean feature flags
    set_validated<bool>(obj, "cosine_similarity", config.cosine_similarity,
                        always_true{});
    set_validated<bool>(obj, "dot_product", config.dot_product, always_true{});
    set_validated<bool>(obj, "l1_distance", config.l1_distance, always_true{});
    set_validated<bool>(obj, "l2_distance", config.l2_distance, always_true{});

    set_validated<bool>(obj, "embedding_std", config.embedding_std, always_true{});
    set_validated<bool>(obj, "embedding_abs_mean", config.embedding_abs_mean,
                        always_true{});
    set_validated<bool>(obj, "embedding_norm_ratio", config.embedding_norm_ratio,
                        always_true{});

    set_validated<bool>(obj, "hadamard_product_sum", config.hadamard_product_sum,
                        always_true{});
    set_validated<bool>(obj, "hadamard_product_mean", config.hadamard_product_mean,
                        always_true{});
    set_validated<bool>(obj, "hadamard_product_components",
                        config.hadamard_product_components, always_true{});

    set_validated<bool>(obj, "common_neighbors_count", config.common_neighbors_count,
                        always_true{});
    set_validated<bool>(obj, "jaccard_coefficient", config.jaccard_coefficient,
                        always_true{});

    set_validated<bool>(obj, "adamic_adar_index", config.adamic_adar_index,
                        always_true{});
    set_validated<bool>(obj, "preferential_attachment", config.preferential_attachment,
                        always_true{});
    set_validated<bool>(obj, "resource_allocation_index",
                        config.resource_allocation_index, always_true{});

    set_validated<bool>(obj, "node_degree", config.node_degree, always_true{});
    
    // Temporal edge features
    set_validated<bool>(obj, "temporal_avg_duration", config.temporal_avg_duration,
                        always_true{});
    set_validated<bool>(obj, "temporal_avg_inter_arrival", config.temporal_avg_inter_arrival,
                        always_true{});
    set_validated<bool>(obj, "temporal_var_inter_arrival", config.temporal_var_inter_arrival,
                        always_true{});
    set_validated<bool>(obj, "temporal_regularity", config.temporal_regularity,
                        always_true{});
    set_validated<bool>(obj, "temporal_concentration", config.temporal_concentration,
                        always_true{});
    
    // Bidirectional flow features
    set_validated<bool>(obj, "bidirectional_has_flows", config.bidirectional_has_flows,
                        always_true{});
    set_validated<bool>(obj, "bidirectional_response_time",
                        config.bidirectional_response_time, always_true{});
    set_validated<bool>(obj, "bidirectional_request_ratio",
                        config.bidirectional_request_ratio, always_true{});
    set_validated<bool>(obj, "bidirectional_asymmetry", config.bidirectional_asymmetry,
                        always_true{});
    
    return config;
}

void tag_invoke(json::value_from_tag, json::value &jv, const FeatureConfig &config) {
    json::object obj;

    obj["cosine_similarity"] = json::value_from(config.cosine_similarity);
    obj["dot_product"] = json::value_from(config.dot_product);
    obj["l1_distance"] = json::value_from(config.l1_distance);
    obj["l2_distance"] = json::value_from(config.l2_distance);

    obj["embedding_std"] = json::value_from(config.embedding_std);
    obj["embedding_abs_mean"] = json::value_from(config.embedding_abs_mean);
    obj["embedding_norm_ratio"] = json::value_from(config.embedding_norm_ratio);

    obj["hadamard_product_sum"] = json::value_from(config.hadamard_product_sum);
    obj["hadamard_product_mean"] = json::value_from(config.hadamard_product_mean);
    obj["hadamard_product_components"] =
        json::value_from(config.hadamard_product_components);

    obj["common_neighbors_count"] = json::value_from(config.common_neighbors_count);
    obj["jaccard_coefficient"] = json::value_from(config.jaccard_coefficient);
    obj["adamic_adar_index"] = json::value_from(config.adamic_adar_index);
    obj["preferential_attachment"] = json::value_from(config.preferential_attachment);
    obj["resource_allocation_index"] = json::value_from(config.resource_allocation_index);

    obj["node_degree"] = json::value_from(config.node_degree);

    // Temporal edge features
    obj["temporal_avg_duration"] = json::value_from(config.temporal_avg_duration);
    obj["temporal_avg_inter_arrival"] = json::value_from(config.temporal_avg_inter_arrival);
    obj["temporal_var_inter_arrival"] = json::value_from(config.temporal_var_inter_arrival);
    obj["temporal_regularity"] = json::value_from(config.temporal_regularity);
    obj["temporal_concentration"] = json::value_from(config.temporal_concentration);
    
    // Bidirectional flow features
    obj["bidirectional_has_flows"] = json::value_from(config.bidirectional_has_flows);
    obj["bidirectional_response_time"] = json::value_from(config.bidirectional_response_time);
    obj["bidirectional_request_ratio"] = json::value_from(config.bidirectional_request_ratio);
    obj["bidirectional_asymmetry"] = json::value_from(config.bidirectional_asymmetry);

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