#pragma once

#include "FeatureBaseline.hpp"
#include <armadillo>
#include <functional>
#include <string>
#include <vector>

namespace explainability {

/**
 * @brief Score changes caused by neutralizing one feature group.
 *
 * contribution[i] is defined as:
 * original_positive_score[i] - score_without_group[i].
 *
 * Positive values mean the group increased the positive-class score for that
 * pair. Negative values mean the group reduced it.
 */
struct GroupAblationResult {
    std::string group;
    std::string explanation;
    arma::rowvec score_without_group;
    arma::rowvec contribution;
};

/**
 * @brief Scores a feature matrix and returns positive-class probabilities.
 *
 * Input uses the project feature layout: rows are features, columns are
 * evaluated directed pairs. The returned row vector must have one score per
 * column.
 */
using ScoreFunction = std::function<arma::rowvec(const arma::fmat&)>;

/**
 * @brief Computes local group ablation explanations for evaluated pairs.
 *
 * For each known feature group, matching feature rows are replaced with their
 * training median baseline, the classifier score is recomputed, and the score
 * difference is reported as that group's local contribution.
 *
 * This is a sensitivity explanation, not a causal guarantee. Correlated feature
 * groups can share evidence, so contributions should be interpreted as score
 * change under this specific median-replacement intervention.
 *
 * @param features Feature matrix for evaluated pairs; rows are features,
 * columns are pairs.
 * @param original_scores Positive-class scores produced from the unmodified
 * feature matrix.
 * @param feature_names Names matching feature rows exactly.
 * @param baseline Training medians matching the active feature configuration.
 * @param score_function Classifier scoring callback.
 * @return One result for each feature group present in the active feature set.
 * @throws std::invalid_argument if dimensions are inconsistent, the baseline
 * does not match, a feature name has no known group, no known groups are found,
 * or the scorer returns wrong size.
 */
std::vector<GroupAblationResult> explain_with_group_ablation(
    const arma::fmat& features,
    const arma::rowvec& original_scores,
    const std::vector<std::string>& feature_names,
    const FeatureBaseline& baseline,
    const ScoreFunction& score_function);

} // namespace explainability
