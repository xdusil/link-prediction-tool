#pragma once

#include <optional>
#include <string>

/**
 * @brief Command line argument parsing and validation
 *
 * This namespace provides functions to parse and validate command line arguments.
 * It includes options for training, prediction, and ground truth modes,
 * as well as paths to various files needed for the application.
 */
namespace cli {

struct cmd_args {
    // Flags
    bool training_mode;
    bool prediction_mode;
    bool ground_truth_mode;
    bool verbose;
    bool help;

    // Paths
    std::optional<std::string> blocked_ips_path; // Path to blocked IPs file
    std::optional<std::string> classifier_path;  // Path to classifier model
    std::optional<std::string> config_path;      // Path to configuration file
    std::optional<std::string> data_path;        // Path to data file
    std::optional<std::string>
        ground_truth_input_path; // File containing existing ground truth
    std::optional<std::string>
        ground_truth_output_path;                 // File to save ground truth results
    std::optional<std::string> internal_ips_path; // Path to internal IPs file
    std::optional<std::string>
        predictions_output_path; // File to save predicted dependencies
};

/**
 * @brief Parse command line arguments
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @return Parsed command line arguments
 */
cmd_args parse_args(int argc, char *argv[]);

/**
 * @brief Check command line arguments for validity
 *
 * @param args The command line arguments to check
 * @throws CliValidationException if the arguments are invalid
 */
void validate_args(const cmd_args &args);

/**
 * @brief Print help message for command line arguments
 */
void print_help();

} // namespace cli