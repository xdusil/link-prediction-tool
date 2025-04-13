#pragma once

#include "Types.hpp"
#include "classifier/generic/RandomForestClassifier.hpp"
#include "classifier/binary/BinaryRandomForestClassifier.hpp"
#include "config/config.hpp"
#include "constrained_collections/counters/EvictingCounter.hpp"
#include "constrained_collections/reservoirs/CapacityLimitedReservoir.hpp"
#include "data_preprocessing/FlowProcessor.hpp"
#include "generators/context/SlidingWindowContextGenerator.hpp"
#include "generators/dependency/CandidateDependencyGenerator.hpp"
#include "graph/network/NetworkGraphManager.hpp"
#include "ground_truth/DependencyAnalyser.hpp"
#include "model/SkipGramModel.hpp"
#include "model/data/DataLoader.hpp"
#include "model/optimizer/Optimizer.hpp"
#include "model/trainer/SkipGramTrainer.hpp"
#include "random_walk/logic/custom/CustomRandomWalkLogic.hpp"
#include "random_walk/manager/RandomWalkManager.hpp"
#include "statistics/metrics.hpp"
#include "utils/ip/AllowedIPChecker.hpp"
#include "utils/ip/BoostIPHandler.hpp"
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

/**
 * @brief Main application class for link prediction
 *
 * This class orchestrates the link prediction workflow, supporting both
 * training and prediction modes.
 */
class LinkPredictionApp {
public:
    /**
     * @brief Construct a new LinkPredictionApp
     * @param config_path Path to the configuration file (optional)
     */
    LinkPredictionApp(const std::optional<std::string> &config_path = std::nullopt);

    LinkPredictionApp(const LinkPredictionApp &) = delete;
    LinkPredictionApp &operator=(const LinkPredictionApp &) = delete;
    LinkPredictionApp(LinkPredictionApp &&) = delete;
    LinkPredictionApp &operator=(LinkPredictionApp &&) = delete;

    /**
     * @brief Destroy the LinkPredictionApp
     */
    ~LinkPredictionApp() = default;

    /**
     * @brief Run the training mode
     *
     * @param classifier_path Path to the classifier model file
     * @param data_path Path to the input data file
     * @param ground_truth_path Path to the existing ground truth file (optional)
     * @param ground_truth_output_path Path to save the calculated ground truth (optional)
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     * @param internal_ips_path Path to the internal IPs file (optional)
     */
    void run_training_mode(
        const std::string &classifier_path, const std::string &data_path,
        const std::optional<std::string> &ground_truth_path = std::nullopt,
        const std::optional<std::string> &ground_truth_output_path = std::nullopt,
        const std::optional<std::string> &blocked_ips_path = std::nullopt,
        const std::optional<std::string> &internal_ips_path = std::nullopt);

    /**
     * @brief Run the prediction mode
     *
     * @param classifier_path Path to the classifier model file
     * @param predictions_output_path Path to save the predicted dependencies
     * @param data_path Path to the input data file
     * @param ground_truth_path Path to the ground truth file for evaluation (optional)
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     * @param internal_ips_path Path to the internal IPs file (optional)
     */
    void run_prediction_mode(
        const std::string &classifier_path, const std::string &predictions_output_path,
        const std::string &data_path,
        const std::optional<std::string> &ground_truth_path = std::nullopt,
        const std::optional<std::string> &blocked_ips_path = std::nullopt,
        const std::optional<std::string> &internal_ips_path = std::nullopt);

    /**
     * @brief Run the ground truth mode
     *
     * This mode calculates the ground truth dependencies only.
     *
     * @param data_path Path to the input data file
     * @param ground_truth_output_path Path to save the calculated ground truth
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     */
    void run_ground_truth_mode(const std::string &data_path,
                               const std::string &ground_truth_output_path,
                               const std::optional<std::string> &blocked_ips_path);

private:
    /**
     * @brief Perform common operations for training and prediction modes
     *
     * @param data_path Path to the input data file
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     * @param internal_ips_path Path to the internal IPs file (optional)
     */
    void common_training_or_prediction(std::string data_path,
                                       std::optional<std::string> blocked_ips_path,
                                       std::optional<std::string> internal_ips_path);

    /**
     * @brief Perform common startup operations
     *
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     * @param internal_ips_path Path to the internal IPs file (optional)
     */
    void common_startup(std::optional<std::string> blocked_ips_path,
                        std::optional<std::string> internal_ips_path);

    /**
     * @brief Process the input data
     *
     * This method reads the input data file, processes the flows.
     * It fills the internal and external counters and the reservoir.
     *
     * @param data_path Path to the input data file
     */
    void process_data(const std::string &data_path);

    /**
     * @brief Build the network graph
     */
    void build_graph();

    /**
     * @brief Generate embeddings
     *
     * This method generates embeddings for the network graph.
     */
    void generate_embeddings();

    /**
     * @brief Train the classifier
     *
     * @param features The features
     * @param labels The labels
     * @param use_grid_search Whether to use grid search for hyperparameter tuning
     */
    void train_classifier(const auto &features, const auto &labels, bool use_grid_search);

    /**
     * @brief Perform grid search to find the best Random Forest parameters
     *
     * @param features The features
     * @param labels The labels
     * @return The best Random Forest parameters
     */
    RandomForestParams perform_grid_search(const auto &features, const auto &labels);

    /**
     * @brief Generate predictions
     *
     * This method generates predictions using the trained classifier.
     *
     * @param output_path Path to save the predicted dependencies
     * @param ground_truth_path Path to the ground truth file for evaluation (optional)
     */

    void generate_predictions(const std::string &output_path,
                              const std::optional<std::string> &ground_truth_path);

    // Configuration
    config::Config m_config;

    // IP Addressing
    std::unique_ptr<IIPChecker> m_allowed_ip_checker;
    std::unique_ptr<IIPChecker> m_internal_ip_checker;

    // Counters and Storage
    std::unique_ptr<IEvictingCounter<IPAddress>> m_internal_counter;
    std::unique_ptr<IEvictingCounter<IPAddress>> m_external_counter;
    std::unique_ptr<ICapacityLimitedReservoir<IPAddress, IPEdge>> m_reservoir;

    // Graph Management
    std::unique_ptr<NetworkGraphManager> m_graph_manager;

    // Model Components
    std::unique_ptr<SkipGramModel> m_model;

    // Classifier
    std::unique_ptr<BinaryRandomForestClassifier<arma::fmat, arma::Row<size_t>>> m_classifier;

    // Ground Truth
    std::unique_ptr<ground_truth::DependencyAnalyser> m_dependency_analyser;

    // Random seed
    unsigned int m_seed = 42;
};