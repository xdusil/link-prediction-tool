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
 * @brief File writer exception class
 */
class FileWriterException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief File exception class
 */
class FileException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief Random forest exception class
 */
class RandomForestException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Exception will be thrown when parsing cmd arguments go to default branch
class UnknownOptionException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
