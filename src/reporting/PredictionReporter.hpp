#pragma once

#include "Types.hpp"
#include "config/config.hpp"
#include "graph/boost/BoostGraphTraits.hpp"
#include "graph/network/INetworkGraphManager.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "statistics/metrics.hpp"
#include <armadillo>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace service {
template <typename GraphTraits>
class EdgeServiceClassifier;
}

namespace reporting {

/**
 * @brief Counts emitted positive predictions for one prediction artifact.
 */
struct PredictionWriteSummary {
    std::size_t positive_predictions = 0;
    std::size_t total_predictions = 0;
};

/**
 * @brief Metadata written to a run manifest JSON file.
 */
struct RunManifest {
    std::string path;
    std::string mode;
    std::optional<std::string> config_path;
    std::string data_path;
    std::string classifier_path;
    std::string output_path;
    std::optional<std::string> reference_path;
    const config::Config& config;
    const INetworkGraphManager<BoostGraphTraits<Graph>>& graph_manager;
    std::size_t evaluated_pair_count = 0;
};

/**
 * @brief Writes per-pair classifier scores and optional labels to CSV.
 *
 * @param path Output CSV path.
 * @param pairs Evaluated directed IP pairs, ordered exactly like predictions/scores.
 * @param predictions Predicted binary labels.
 * @param positive_scores Positive-class scores.
 * @param labels Optional reference labels.
 * @throws std::invalid_argument if pair, prediction, score, and label sizes differ.
 */
void write_pair_scores(const std::string& path,
                       const std::vector<std::pair<IPAddress, IPAddress>>& pairs,
                       const arma::Row<std::size_t>& predictions,
                       const arma::rowvec& positive_scores,
                       const std::optional<arma::Row<std::size_t>>& labels);

/**
 * @brief Writes positive predictions to the main prediction CSV.
 *
 * Only pairs predicted as positive are emitted. If a service classifier is provided,
 * service columns are added and filled from the retained graph.
 *
 * @param path Output CSV path.
 * @param pairs Evaluated directed IP pairs, ordered exactly like predictions/scores.
 * @param predictions Predicted binary labels.
 * @param positive_scores Positive-class scores.
 * @param graph_manager Retained graph manager used for optional service classification.
 * @param service_classifier Optional service classifier; nullptr disables service
 * columns.
 * @return Number of positive and total predictions.
 * @throws std::invalid_argument if pair, prediction, and score sizes differ.
 */
PredictionWriteSummary write_positive_predictions(
    const std::string& path,
    const std::vector<std::pair<IPAddress, IPAddress>>& pairs,
    const arma::Row<std::size_t>& predictions, const arma::rowvec& positive_scores,
    const INetworkGraphManager<BoostGraphTraits<Graph>>& graph_manager,
    const service::EdgeServiceClassifier<BoostGraphTraits<Graph>>* service_classifier);

/**
 * @brief Writes run metadata for reproducibility and publication artifacts.
 *
 * @param manifest Run metadata and output path.
 */
void write_run_manifest(const RunManifest& manifest);

/**
 * @brief Writes classification and ranking metrics to JSON.
 *
 * @param path Output JSON path.
 * @param metrics Calculated metrics.
 * @param positive_predictions Number of predicted positive dependencies.
 * @param total_predictions Total number of predictions.
 */
void write_metrics_report(const std::string& path, const statistics::Metrics& metrics,
                          std::size_t positive_predictions,
                          std::size_t total_predictions);

} // namespace reporting
