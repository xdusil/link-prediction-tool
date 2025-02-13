#pragma once

#include "config.hpp"
#include <string>

/**
 * @brief Configuration loader class
 *
 * Loads configuration from a JSON file.
 */
class ConfigLoader {
public:

    /**
     * @brief Load configuration from a file
     *
     * @param filename The name of the file to load the configuration from
     * @return The loaded configuration
     */
    static Config load(const std::string& filename);
    
private:

    /**
     * @brief Validate the configuration
     *
     * @param config The configuration to validate
     */
    static void validate(const Config& config);

    /**
     * @brief Parse a JSON string into a Config object
     *
     * @param json_content The JSON string to parse
     * @return The parsed Config object
     */
    static Config parse_json(const std::string& json_content);
};