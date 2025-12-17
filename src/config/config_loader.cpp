#include "config.hpp"
#include "../exceptions/exceptions.hpp"
#include "../io/FileReader.hpp"
#include "../json/JsonHelper.hpp"
#include "tag_invokes/tag_invokes.hpp"

namespace json = boost::json;

namespace config {

/**
 * @brief Check if configuration parameter combinations are valid
 * 
 * @param config The configuration to validate
 * @throws ConfigurationException if any parameter combination is invalid
 */
void validate_parameter_combinations(const Config& config) {
    // Check walk length and context size relationship
    if (config.WALK_LENGTH <= config.CONTEXT_SIZE) {
        throw ConfigurationException("WALK_LENGTH must be greater than CONTEXT_SIZE");
    }
    
    // Check that at least one feature is enabled
    bool any_feature_enabled = config.FEATURE_CONFIG.get_dimension() > 0;
    if (!any_feature_enabled) {
        throw ConfigurationException("At least one feature must be enabled in FEATURE_CONFIG");
    }
    
    // Check that the classifier threshold and threshold calibration are not used together
    if (config.USE_THRESHOLD_CALIBRATION && config.CLASSIFIER_THRESHOLD) {
        throw ConfigurationException(
            "CLASSIFIER_THRESHOLD cannot be used with USE_THRESHOLD_CALIBRATION");
    }
}

Config parse_json(const std::string &json_content) {
    try {
        json::value jv = JsonHelper::parse_json(json_content, json::parse_options{.allow_comments = true});
        Config config = json::value_to<Config>(jv);

        return config;
    } catch (const std::exception &e) {
        std::throw_with_nested(
            ConfigurationException("Error while parsing JSON configuration"));
    }
}

Config load(const std::string &filename) {
    FileReader reader(filename);
    std::string json_content;

    try {
        reader.read_all(json_content);
    } catch (const std::exception &e) {
        std::throw_with_nested(ConfigurationException("Could not read config file" + filename));
    }

    Config config = parse_json(json_content);
    validate_parameter_combinations(config);
    return config;
}
} // namespace config