#include "app/LinkPredictionApp.hpp"
#include "cli/command_line.hpp"
#include "utils/utils.hpp"
#include <boost/json/src.hpp> // for boost json - has to be included only once in the project

using namespace cli;

int main(int argc, char *argv[]) {
    try {
        cmd_args args = parse_args(argc, argv);

        if (args.help) {
            print_help();
            return EXIT_SUCCESS;
        }

        validate_args(args);

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
        utils::print_exception(e, 0, "Error");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}