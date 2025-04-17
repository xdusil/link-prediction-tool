#include "app/LinkPredictionApp.hpp"
#include "exceptions/exceptions.hpp"
#include <boost/json/src.hpp> // for boost json - has to be included only once in the project
#include <cstdio>
#include <cstdlib>
#include <exception>
#include "utils/utils.hpp"
#include <getopt.h>
#include <iostream>
#include <optional>
#include <string>
#include <unistd.h>

static void print_help() {
    std::cout
        << "Usage: ./LinkPrediction [MODE] [OPTIONS]\n"
        << "\nModes (one required):\n"
        << "  -t, --training              Run in training mode\n"
        << "  -p, --prediction            Run in prediction mode\n"
        << "  -x, --extract               Extract ground truth only\n"
        << "\nRequired options for training mode:\n"
        << "  -c, --classifier PATH       Path to classifier model file (to save)\n"
        << "  -d, --data PATH             Path to input data file\n"
        << "\nRequired options for prediction mode:\n"
        << "  -c, --classifier PATH       Path to classifier model file (to load)\n"
        << "  -d, --data PATH             Path to input data file\n"
        << "  -o, --predictions-out PATH  Path to save predicted dependencies\n"
        << "\nOptional options for training mode:\n"
        << "  -G, --ground-truth-out PATH Path to save ground truth results\n"
        << "\nOptional options for both modes:\n"
        << "  -f, --config PATH           Path to configuration file\n"
        << "  -b, --blocked-ips PATH      Path to blocked IPs file\n"
        << "  -i, --internal-ips PATH     Path to internal IPs file\n"
        << "  -g, --ground-truth-in PATH  Path to existing ground truth file\n"
        << "  -v, --verbose               Enable verbose output\n"
        << "\nHelp:\n"
        << "  -h, --help                  Display this help message\n";
}

static struct option long_options[] = {
    {"blocked-ips", required_argument, nullptr, 'b'},
    {"classifier", required_argument, nullptr, 'c'},
    {"config", required_argument, nullptr, 'f'},
    {"data", required_argument, nullptr, 'd'},
    {"extract", no_argument, nullptr, 'x'},
    {"ground-truth-in", required_argument, nullptr, 'g'},
    {"ground-truth-out", required_argument, nullptr, 'G'},
    {"help", no_argument, nullptr, 'h'},
    {"internal-ips", required_argument, nullptr, 'i'},
    {"prediction", no_argument, nullptr, 'p'},
    {"predictions-out", required_argument, nullptr, 'o'},
    {"training", no_argument, nullptr, 't'},
    {"verbose", no_argument, nullptr, 'v'},
    {0, 0, 0, 0}};

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

void check_training_mode(const cmd_args &args) {
    if (args.prediction_mode) {
        throw CliValidationException("Cannot specify both training and prediction mode");
    }

    if (args.ground_truth_mode) {
        throw CliValidationException("Cannot combine training with ground-truth-only mode");
    }

    // Validate required files
    if (!args.classifier_path) {
        throw CliValidationException("Missing required argument --classifier");
    }

    if (!args.data_path) {
        throw CliValidationException("Missing required argument --data");
    }
}

void check_prediction_mode(const cmd_args &args) {

    if (args.training_mode) {
        throw CliValidationException("Cannot specify both training and prediction mode");
    }
    if (args.ground_truth_mode) {
        throw CliValidationException("Cannot combine prediction with ground-truth-only mode");
    }

    // Validate required files
    if (!args.classifier_path) {
        throw CliValidationException("Missing required argument --classifier");
    }

    if (!args.data_path) {
        throw CliValidationException("Missing required argument --data");
    }

    if (!args.predictions_output_path) {
        throw CliValidationException("Missing required argument --predictions-out");
    }

    if (args.ground_truth_output_path) {
        throw CliValidationException(
            "Cannot specify --ground-truth-out in prediction mode");
    }
}

void check_ground_truth_mode(const cmd_args &args) {
    if (args.training_mode || args.prediction_mode) {
        throw CliValidationException("Cannot combine ground-truth-only with other modes");
    }

    if (!args.data_path) {
        throw CliValidationException("Missing required argument --data");
    }

    if (!args.ground_truth_output_path) {
        throw CliValidationException("Missing required argument --ground-truth-out");
    }
    
    if (args.classifier_path) {
        throw CliValidationException("Classifier not needed in ground-truth-only mode");
    }
    
    if (args.predictions_output_path) {
        throw CliValidationException("Predictions output not used in ground-truth-only mode");
    }

    if (args.ground_truth_input_path) {
        throw CliValidationException("Ground truth input not used in ground-truth-only mode");
    }

    if (args.internal_ips_path) {
        throw CliValidationException("Internal IPs not used in ground-truth-only mode");
    }
}

void check_cli_args(const cmd_args &args) {
    if (!args.training_mode && !args.prediction_mode && !args.ground_truth_mode) {
        throw CliValidationException("Must specify a mode: training, prediction, or ground-truth-only");
    }
    
    if (args.training_mode) {
        check_training_mode(args);
    } else if (args.prediction_mode) {
        check_prediction_mode(args);
    } else {
        check_ground_truth_mode(args);
    }
}

cmd_args parse_cmd_args(int argc, char *argv[]) {
    cmd_args args{}; // Initialize all fields
    int opt;

    const char *short_opts = ":b:c:f:d:g:G:hi:po:txv";

    while ((opt = getopt_long(argc, argv, short_opts, long_options, nullptr)) != -1) {
        switch (opt) {
        // Flag options
        case 't':
            args.training_mode = true;
            break;
        case 'p':
            args.prediction_mode = true;
            break;
        case 'x':
            args.ground_truth_mode = true;
            break;
        case 'h':
            args.help = true;
            break;

        // Path options
        case 'b':
            args.blocked_ips_path = optarg;
            break;
        case 'c':
            args.classifier_path = optarg;
            break;
        case 'f':
            args.config_path = optarg;
            break;
        case 'd':
            args.data_path = optarg;
            break;
        case 'g':
            args.ground_truth_input_path = optarg;
            break;
        case 'G':
            args.ground_truth_output_path = optarg;
            break;
        case 'i':
            args.internal_ips_path = optarg;
            break;
        case 'o':
            args.predictions_output_path = optarg;
            break;
        case 'v':
            args.verbose = true;
            break;

        // Error handling
        case '?':
            throw UnknownOptionException("Unknown option " +
                                         std::string(argv[optind - 1]));
        case ':':
            throw MissingArgumentException("Missing argument for option " +
                                           std::string(argv[optind - 1]));
        default:
            throw CliValidationException("Unhandled option " +
                                         std::string(argv[optind - 1]));
        }
    }

    return args;
}

int main(int argc, char *argv[]) {
    try {
        cmd_args args = parse_cmd_args(argc, argv);

        if (args.help) {
            print_help();
            return EXIT_SUCCESS;
        }

        check_cli_args(args);

        LinkPredictionApp app(args.config_path, args.verbose);

        if (args.training_mode) {
            app.run_training_mode(*args.classifier_path, *args.data_path,
                                  args.ground_truth_input_path,
                                  args.ground_truth_output_path, args.blocked_ips_path,
                                  args.internal_ips_path);
        } else if (args.prediction_mode) {
            app.run_prediction_mode(*args.classifier_path, *args.predictions_output_path,
                                    *args.data_path, args.ground_truth_input_path,
                                    args.blocked_ips_path, args.internal_ips_path);
        } else {
            app.run_ground_truth_mode(*args.data_path, *args.ground_truth_output_path,
                                           args.blocked_ips_path);
        }

    } catch (const std::exception &e) {
        utils::print_exception(e);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}