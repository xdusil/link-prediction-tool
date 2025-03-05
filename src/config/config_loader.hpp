#pragma once

#include "config.hpp"
#include <string>

namespace config {

/**
 * @brief Load configuration from a file
 *
 * @param filename The name of the file to load the configuration from
 * @return The loaded configuration
 */
Config load(const std::string &filename);

} // namespace config