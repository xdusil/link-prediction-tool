#pragma once

#include "Types.hpp"
#include "classifier/binary/IBinaryRandomForestClassifier.hpp"
#include "config/config.hpp"
#include "constrained_collections/counters/IEvictingCounter.hpp"
#include "constrained_collections/reservoirs/ICapacityLimitedReservoir.hpp"
#include "graph/boost/BoostGraphTraits.hpp"
#include "graph/network/INetworkGraphManager.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "ground_truth/DependencyAnalyzer.hpp"
#include "model/DirectionalEmbeddings.hpp"
#include "model/DirectionalSkipGramModel.hpp"
#include "model/trainer/ITrainer.hpp"
#include "statistics/metrics.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <armadillo>
#include <optional>
#include <string>

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
     * @param verbose Enable verbose output (default: false)
     */
    LinkPredictionApp(const std::optional<std::string>& config_path = std::nullopt,
                      bool verbose = false);

    LinkPredictionApp(const LinkPredictionApp&) = delete;
    LinkPredictionApp& operator=(const LinkPredictionApp&) = delete;
    LinkPredictionApp(LinkPredictionApp&&) = delete;
    LinkPredictionApp& operator=(LinkPredictionApp&&) = delete;

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
     * @param feature_importance Calculate feature importance (default: false)
     */
    void run_training_mode(
        const std::string& classifier_path, const std::string& data_path,
        const std::optional<std::string>& ground_truth_path = std::nullopt,
        const std::optional<std::string>& ground_truth_output_path = std::nullopt,
        const std::optional<std::string>& blocked_ips_path = std::nullopt,
        const std::optional<std::string>& internal_ips_path = std::nullopt,
        bool feature_importance = false);

    /**
     * @brief Run the prediction mode
     *
     * @param classifier_path Path to the classifier model file
     * @param predictions_output_path Path to save the predicted dependencies
     * @param data_path Path to the input data file
     * @param ground_truth_path Path to the ground truth file for evaluation (optional)
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     * @param internal_ips_path Path to the internal IPs file (optional)
     * @param scores_output_path Path to save all evaluated pair scores (optional)
     */
    void run_prediction_mode(
        const std::string& classifier_path, const std::string& predictions_output_path,
        const std::string& data_path,
        const std::optional<std::string>& ground_truth_path = std::nullopt,
        const std::optional<std::string>& blocked_ips_path = std::nullopt,
        const std::optional<std::string>& internal_ips_path = std::nullopt,
        const std::optional<std::string>& scores_output_path = std::nullopt);

    /**
     * @brief Run the ground truth mode
     *
     * This mode calculates the ground truth dependencies only.
     *
     * @param data_path Path to the input data file
     * @param ground_truth_output_path Path to save the calculated ground truth
     * @param blocked_ips_path Path to the blocked IPs file (optional)
     */
    void run_ground_truth_mode(const std::string& data_path,
                               const std::string& ground_truth_output_path,
                               const std::optional<std::string>& blocked_ips_path);

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
    void process_data(const std::string& data_path);

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
     * @param feature_importance Whether to calculate feature importance
     */
    void train_classifier(const auto& features, const auto& labels, bool use_grid_search,
                          bool feature_importance);

    /**
     * @brief Perform grid search to find the best Random Forest parameters
     *
     * @param features The features
     * @param labels The labels
     * @return The best Random Forest parameters
     */
    RandomForestParams perform_grid_search(const auto& features,
                                           const auto& labels) const;

    /**
     * @brief Generate predictions
     *
     * This method generates predictions using the trained classifier.
     *
     * @param output_path Path to save the predicted dependencies
     * @param ground_truth_path Path to the ground truth file for evaluation (optional)
     * @param scores_output_path Path to save all evaluated pair scores (optional)
     * @return Number of directed pairs evaluated by the classifier.
     */

    std::size_t generate_predictions(
        const std::string& output_path,
        const std::optional<std::string>& ground_truth_path,
        const std::optional<std::string>& scores_output_path);

    /**
     * @brief Calculate reference coverage for the currently retained graph.
     *
     * The result is used only for reporting. Labels are still generated from the
     * full loaded/calculated dependency set over the retained graph pairs.
     */
    ground_truth::ProjectionStats calculate_reference_projection_stats() const;

    /**
     * @brief Helper function to log messages when verbose mode is enabled
     *
     * @tparam Args The types of the arguments
     * @param args The arguments to log
     *
     * This method uses fold expressions to log multiple arguments.
     * It only logs if the verbose flag is set to true.
     */
    template <typename... Args>
    void log_verbose(Args&&... args) const {
        if (m_verbose) {
            std::cout << "[VERBOSE] ";
            (std::cout << ... << std::forward<Args>(args));
            std::cout << std::endl;
        }
    }

    // Configuration
    config::Config m_config;
    std::optional<std::string> m_config_path;

    // IP Addressing
    std::unique_ptr<IIPChecker> m_allowed_ip_checker;
    std::unique_ptr<IIPChecker> m_internal_ip_checker;

    // Counters and Storage
    std::unique_ptr<IEvictingCounter<IPAddress>> m_internal_counter;
    std::unique_ptr<IEvictingCounter<IPAddress>> m_external_counter;
    std::unique_ptr<ICapacityLimitedReservoir<IPAddress, IPEdge>> m_reservoir;

    // Graph Management
    std::unique_ptr<INetworkGraphManager<BoostGraphTraits<Graph>>> m_graph_manager;

    // Model Components
    std::unique_ptr<DirectionalSkipGramModel> m_model;

    // Classifier
    std::unique_ptr<IBinaryRandomForestClassifier<arma::fmat, arma::Row<size_t>,
                                                  statistics::Metrics, arma::mat>>
        m_classifier;

    // Ground Truth
    std::unique_ptr<ground_truth::IDependencyAnalyzer<ground_truth::DependencySet>>
        m_dependency_analyzer;

    // Verbose logging
    bool m_verbose = false;
};
