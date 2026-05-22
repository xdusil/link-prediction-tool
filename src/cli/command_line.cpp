#include "command_line.hpp"
#include "exceptions/exceptions.hpp"
#include "utils/utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <getopt.h>
#include <iostream>
#include <optional>
#include <string>
#include <unistd.h>

namespace cli {

void print_help() {
    std::cout
        << "Usage: ./LinkPrediction [MODE] [OPTIONS]\n"
        << "\nModes (one required):\n"
        << "  -t, --training              Run in training mode (build model)\n"
        << "  -p, --prediction            Run in prediction mode (use existing model)\n"
        << "  -x, --extract               Extract ground truth only (no model needed)\n"
        << "\nRequired options for training mode:\n"
        << "  -c, --classifier PATH       Path to classifier model file (to save)\n"
        << "  -d, --data PATH             Path to input data file\n"
        << "\nRequired options for prediction mode:\n"
        << "  -c, --classifier PATH       Path to classifier model file (to load)\n"
        << "  -d, --data PATH             Path to input data file\n"
        << "  -o, --predictions-out PATH  Path to save predicted dependencies\n"
        << "\nRequired options for ground truth extraction:\n"
        << "  -d, --data PATH             Path to input data file\n"
        << "  -G, --ground-truth-out PATH Path to save ground truth results\n"
        << "\nOptional options by mode:\n"
        << "  Training mode:\n"
        << "    -g, --ground-truth-in PATH  Use existing ground truth instead of "
           "calculating\n"
        << "    -G, --ground-truth-out PATH Save extracted ground truth\n"
        << "    -b, --blocked-ips PATH      Path to list of IPs to ignore\n"
        << "    -i, --internal-ips PATH     Path to list of internal network IPs\n"
        << "    -F, --feature-importance    Calculate feature importance analysis\n"
        << "  Prediction mode:\n"
        << "    -g, --ground-truth-in PATH  Path to ground truth for evaluation\n"
        << "    -s, --scores-out PATH       Path to save all evaluated pair scores\n"
        << "    -b, --blocked-ips PATH      Path to list of IPs to ignore\n"
        << "    -i, --internal-ips PATH     Path to list of internal network IPs\n"
        << "  Ground truth mode:\n"
        << "    -b, --blocked-ips PATH      Path to list of IPs to ignore\n"
        << "\nGeneral options:\n"
        << "  -f, --config PATH           Path to configuration file\n"
        << "  -v, --verbose               Enable verbose output and timing\n"
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
    {"scores-out", required_argument, nullptr, 's'},
    {"training", no_argument, nullptr, 't'},
    {"verbose", no_argument, nullptr, 'v'},
    {"feature-importance", no_argument, nullptr, 'F'},
    {0, 0, 0, 0}};

void check_training_mode(const cmd_args &args) {
    if (args.prediction_mode) {
        throw CliValidationException("Cannot specify both training and prediction mode");
    }

    if (args.ground_truth_mode) {
        throw CliValidationException(
            "Cannot combine training with ground-truth-only mode");
    }

    // Validate required files
    if (!args.classifier_path) {
        throw CliValidationException("Missing required argument --classifier");
    }

    if (!args.data_path) {
        throw CliValidationException("Missing required argument --data");
    }

    if (args.scores_output_path) {
        throw CliValidationException("Scores output is only used in prediction mode");
    }
}

void check_prediction_mode(const cmd_args &args) {

    if (args.training_mode) {
        throw CliValidationException("Cannot specify both training and prediction mode");
    }
    if (args.ground_truth_mode) {
        throw CliValidationException(
            "Cannot combine prediction with ground-truth-only mode");
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
        throw CliValidationException(
            "Predictions output not used in ground-truth-only mode");
    }
    if (args.scores_output_path) {
        throw CliValidationException(
            "Scores output not used in ground-truth-only mode");
    }

    if (args.ground_truth_input_path) {
        throw CliValidationException(
            "Ground truth input not used in ground-truth-only mode");
    }

    if (args.internal_ips_path) {
        throw CliValidationException("Internal IPs not used in ground-truth-only mode");
    }
}

void validate_args(const cmd_args &args) {
    if (!args.training_mode && !args.prediction_mode && !args.ground_truth_mode) {
        throw CliValidationException(
            "Must specify a mode: training, prediction, or ground-truth-only");
    }

    if (args.training_mode) {
        check_training_mode(args);
    } else if (args.prediction_mode) {
        check_prediction_mode(args);
    } else {
        check_ground_truth_mode(args);
    }
}

cmd_args parse_args(int argc, char *argv[]) {
    cmd_args args{}; // Initialize all fields
    int opt;

    const char *short_opts = ":b:c:f:d:g:G:hi:po:s:txvF";

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
        case 's':
            args.scores_output_path = optarg;
            break;
        case 'v':
            args.verbose = true;
            break;
        case 'F':
            args.feature_importance = true;
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

} // namespace cli
