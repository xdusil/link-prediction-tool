#pragma once

#include <armadillo>
#include <cstddef>
#include <optional>
#include <vector>

namespace statistics {

struct RankingAtK {
    std::size_t k = 0;
    std::optional<double> precision;
    std::optional<double> recall;
};

struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    std::optional<double> roc_auc;
    std::optional<double> average_precision;
    std::optional<double> mean_reciprocal_rank;
    std::vector<RankingAtK> ranking_at_k;
};

enum class AverageType {
    BINARY, // Only for binary classification
    MICRO,  // Calculate metrics globally by counting total TP, FP, etc.
    MACRO,  // Calculate metrics for each class, then take unweighted mean
};

/**
 * @brief Calculate classification metrics.
 *
 * Metrics like accuracy, precision, recall, and F1 score are calculated.
 * This function can also calculate ranking metrics if positive_scores are provided.
 * Score-based metrics are only applicable for binary classification and interpret
 * label 1 as the positive class. Average precision, MRR, precision@K, and recall@K
 * rank samples by the provided positive-class score in descending order.
 *
 * @param predictions Predicted labels
 * @param labels True labels
 * @param positive_scores Predicted positive class scores
 * @param avg_type Type of averaging to use
 * @param num_classes Number of classes
 * @return Metrics
 */
Metrics calculate_metrics(const arma::Row<size_t> &predictions,
                          const arma::Row<size_t> &labels,
                          const arma::rowvec &positive_scores = arma::rowvec(),
                          AverageType avg_type = AverageType::BINARY,
                          size_t num_classes = 2);

/**
 * @brief Calculate ROC-AUC score.
 *
 * @param scores Predicted scores
 * @param labels True labels
 * @return ROC-AUC score, or std::nullopt when labels contain only one class
 */
std::optional<double> calculate_roc_auc(const arma::rowvec &scores,
                                        const arma::Row<size_t> &labels);

} // namespace statistics
