#pragma once

#include "io/FileReader.hpp"
#include <stdexcept>

/**
 * @brief Base exception class for the application
 */
class ApplicationException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief Configuration exception class
 */
class ConfigurationException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief File reader exception class
 */
class FileReaderException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief File writer exception class
 */
class FileWriterException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief File exception class
 */
class FileException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Random forest exception class
 */
class RandomForestException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Unknown option exception class
 *
 * Exception will be thrown when parsing cmd arguments and unknown option is found
 */
class UnknownOptionException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Missing argument exception class
 *
 * Exception will be thrown when parsing cmd arguments and missing argument is found
 */
class MissingArgumentException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief CLI validation exception class
 *
 * Exception will be thrown when error occurs during CLI validation
 */
class CliValidationException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Component not initialized exception class
 *
 * Exception will be thrown when a component is not initialized
 */
class ComponentNotInitializedException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Not supported exception class
 *
 * Exception will be thrown when a feature is not supported
 */
class NotSupportedException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief JSON exception class
 *
 * Exception will be thrown when an error occurs during JSON parsing or manipulation with
 * JSON objects
 */
class JSONException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Feature generator exception class
 *
 * Exception will be thrown when an error occurs during feature generation
 */
class FeatureGeneratorException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Flow processor exception class
 *
 * Exception will be thrown when an error occurs during flow processing
 */
class FlowProcessorException : public ApplicationException {
    using ApplicationException::ApplicationException;
};

/**
 * @brief Graph empty exception class
 *
 * Exception will be thrown when the graph is empty
 */
class GraphEmptyException : public ApplicationException {
    using ApplicationException::ApplicationException;
};