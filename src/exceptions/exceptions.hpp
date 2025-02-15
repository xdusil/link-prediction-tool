#pragma once

#include "io/FileReader.hpp"
#include <stdexcept>

/**
 * @brief Configuration exception class
 */
class ConfigurationException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};


/**
 * @brief File reader exception class
 */
class FileReaderException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief Random forest exception class
 */
class RandomForestException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};