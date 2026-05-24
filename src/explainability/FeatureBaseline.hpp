#pragma once

#include <armadillo>
#include <string>
#include <vector>

namespace explainability {

/**
 * @brief Per-feature neutral values used by local ablation explanations.
 *
 * The baseline is tied to one trained classifier and one feature configuration.
 * Feature order must match the Armadillo feature matrix layout used by the
 * classifier: rows are features, columns are evaluated pairs.
 */
struct FeatureBaseline {
    std::vector<std::string> feature_names;
    std::vector<double> medians;
};

/**
 * @brief Returns the sidecar path used to store baselines for a classifier.
 */
std::string feature_baseline_path(const std::string& classifier_path);

/**
 * @brief Computes one median value per feature row from training data.
 *
 * @param features Training feature matrix, with rows as features and columns as
 * samples.
 * @param feature_names Names matching feature rows exactly.
 * @return Baseline values suitable for local group ablation.
 * @throws std::invalid_argument if dimensions are inconsistent, no samples are
 * available, or a computed median is not finite.
 */
FeatureBaseline compute_feature_baseline(
    const arma::fmat& features,
    const std::vector<std::string>& feature_names);

/**
 * @brief Writes a feature baseline JSON sidecar.
 *
 * @throws std::invalid_argument if names and medians differ in size or a median
 * is not finite.
 */
void save_feature_baseline(const std::string& path, const FeatureBaseline& baseline);

/**
 * @brief Loads a feature baseline JSON sidecar.
 *
 * @throws boost::system::system_error or boost::json exceptions on malformed
 * input, and std::invalid_argument if a loaded median is not finite.
 */
FeatureBaseline load_feature_baseline(const std::string& path);

/**
 * @brief Verifies that a baseline belongs to the active feature configuration.
 *
 * Exact feature-name order is required because ablation replaces feature rows by
 * index.
 *
 * @throws std::invalid_argument if names or median count do not match.
 */
void validate_feature_baseline(
    const FeatureBaseline& baseline,
    const std::vector<std::string>& feature_names);

} // namespace explainability
