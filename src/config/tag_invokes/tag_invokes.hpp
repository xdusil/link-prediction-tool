#pragma once

#include "classifier/generic/RandomForestClassifier.hpp"
#include "config/config.hpp"
#include "generators/feature/FeatureConfig.hpp"
#include "json/JsonHelper.hpp"

/**
 * @brief Tag invoke for RandomForestParams
 *
 * @param jv The JSON value to parse
 * @return The parsed RandomForestParams object
 */
RandomForestParams tag_invoke(json::value_to_tag<RandomForestParams>,
                              const json::value &jv);

/**
 * @brief Tag invoke for serializing RandomForestParams
 *
 * @param jv The JSON value to serialize to
 * @param params The RandomForestParams object to serialize
 */
void tag_invoke(json::value_from_tag, json::value &jv, const RandomForestParams &params);

/**
 * @brief Tag invoke for GridSearchParams
 *
 * @param jv The JSON value to parse
 * @return The parsed GridSearchParams object
 */
GridSearchParams tag_invoke(json::value_to_tag<GridSearchParams>, const json::value &jv);

/**
 * @brief Tag invoke for serializing GridSearchParams
 *
 * @param jv The JSON value to serialize to
 * @param params The GridSearchParams object to serialize
 */
void tag_invoke(json::value_from_tag, json::value &jv, const GridSearchParams &params);

/**
 * @brief Tag invoke for FeatureConfig
 *
 * @param jv The JSON value to parse
 * @return The parsed FeatureConfig object
 */
FeatureConfig tag_invoke(json::value_to_tag<FeatureConfig>, const json::value &jv);

/**
 * @brief Tag invoke for serializing FeatureConfig
 *
 * @param jv The JSON value to serialize to
 * @param config The FeatureConfig object to serialize
 */
void tag_invoke(json::value_from_tag, json::value &jv, const FeatureConfig &config);

// Specializations for JsonHelper
template <>
inline bool JsonHelper::is_type<RandomForestParams>(const json::value &jv) {
    return jv.is_object();
}

template <>
inline bool JsonHelper::is_type<GridSearchParams>(const json::value &jv) {
    return jv.is_object();
}

template <>
inline bool JsonHelper::is_type<FeatureConfig>(const json::value &jv) {
    return jv.is_object();
}

template <>
inline bool JsonHelper::is_type<service::ServiceClassificationConfig>(const boost::json::value &jv) {
    return jv.is_object();
}

namespace service {
// Tag invokes for service::ServiceClassificationConfig must be in service namespace for ADL

/**
 * @brief Tag invoke for ServiceClassificationConfig
 *
 * @param jv The JSON value to parse
 * @return The parsed ServiceClassificationConfig object
 */
ServiceClassificationConfig tag_invoke(boost::json::value_to_tag<ServiceClassificationConfig>,
                                      const boost::json::value &jv);

/**
 * @brief Tag invoke for serializing ServiceClassificationConfig
 *
 * @param jv The JSON value to serialize to
 * @param config The ServiceClassificationConfig object to serialize
 */
void tag_invoke(boost::json::value_from_tag, boost::json::value &jv,
                const ServiceClassificationConfig &config);

} // namespace service

// Main namespace for Config
namespace config {

/**
 * @brief Tag invoke for Config
 *
 * @param jv The JSON value to parse
 * @return The parsed Config object
 */
Config tag_invoke(json::value_to_tag<Config>, const json::value &jv);

/**
 * @brief Tag invoke for serializing Config
 *
 * @param jv The JSON value to serialize to
 * @param config The Config object to serialize
 */
void tag_invoke(json::value_from_tag, json::value &jv, const Config &config);

} // namespace config