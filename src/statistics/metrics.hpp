#pragma once

#include <armadillo>

namespace statistics {

struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    std::optional<double> roc_auc;
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
 * This function can also calculate ROC-AUC score if positive_scores are provided. But it
 * is only applicable for binary classification. For multiclass classification, it will
 * ignore the positive_scores and not calculate ROC-AUC.
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
 * @return ROC-AUC score
 */
double calculate_roc_auc(const arma::rowvec &scores, const arma::Row<size_t> &labels);

} // namespace statistics