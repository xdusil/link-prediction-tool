#pragma once

#include "config/config.hpp"
#include "generators/candidate/CandidatePair.hpp"
#include "graph/boost/BoostGraphTraits.hpp"
#include "graph/network/INetworkGraphManager.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "ground_truth/DependencyAnalyzer.hpp"
#include "statistics/metrics.hpp"
#include <armadillo>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace service {
template <typename GraphTraits>
class EdgeServiceClassifier;
}

namespace reporting {
/**
 * @brief Writes positive predictions to the main prediction CSV.
 *
 * Only candidates predicted as positive are emitted. If a service classifier is provided,
 * service columns are added and filled from the retained graph.
 *
 * @param path Output CSV path.
 * @param candidates Candidate pairs, ordered exactly like predictions/scores.
 * @param predictions Predicted binary labels.
 * @param positive_scores Positive-class scores.
 * @param graph_manager Retained graph manager used for optional service classification.
 * @param service_classifier Optional service classifier; nullptr disables service
 * columns.
 * @return Number of positive and total predictions.
 * @throws std::invalid_argument if candidate, prediction, and score sizes differ.
 */
PredictionWriteSummary write_positive_predictions(
    const std::string& path, const std::vector<candidate::CandidatePair>& candidates,
    const arma::Row<std::size_t>& predictions, const arma::rowvec& positive_scores,
    const INetworkGraphManager<BoostGraphTraits<Graph>>& graph_manager,
    const service::EdgeServiceClassifier<BoostGraphTraits<Graph>>* service_classifier);

} // namespace reporting
