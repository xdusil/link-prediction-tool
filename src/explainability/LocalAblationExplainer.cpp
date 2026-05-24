#include "LocalAblationExplainer.hpp"

#include "FeatureGroups.hpp"
#include <stdexcept>
#include <string>

namespace explainability {

namespace {

void validate_feature_group_coverage(const std::vector<std::string>& feature_names) {
    for (const std::string& feature_name : feature_names) {
        if (!belongs_to_known_group(feature_name)) {
            throw std::invalid_argument(
                "Local ablation has no explainability group for feature: " +
                feature_name);
        }
    }
}

} // namespace

std::vector<GroupAblationResult> explain_with_group_ablation(
    const arma::fmat& features,
    const arma::rowvec& original_scores,
    const std::vector<std::string>& feature_names,
    const FeatureBaseline& baseline,
    const ScoreFunction& score_function) {
    validate_feature_baseline(baseline, feature_names);
    if (features.n_rows != feature_names.size()) {
        throw std::invalid_argument(
            "Local ablation requires one feature name per feature row.");
    }
    if (features.n_cols != original_scores.n_elem) {
        throw std::invalid_argument(
            "Local ablation score count does not match feature columns.");
    }
    validate_feature_group_coverage(feature_names);

    std::vector<GroupAblationResult> results;
    results.reserve(FEATURE_GROUPS.size());

    for (const FeatureGroup& group : FEATURE_GROUPS) {
        arma::fmat ablated_features = features;
        bool group_has_features = false;

        for (std::size_t row = 0; row < feature_names.size(); ++row) {
            if (!has_prefix(feature_names[row], group.prefix)) {
                continue;
            }

            ablated_features.row(row).fill(
                static_cast<arma::fmat::elem_type>(baseline.medians[row]));
            group_has_features = true;
        }

        if (!group_has_features) {
            continue;
        }

        arma::rowvec ablated_scores = score_function(ablated_features);
        if (ablated_scores.n_elem != original_scores.n_elem) {
            throw std::invalid_argument(
                "Local ablation scorer returned an unexpected score count.");
        }

        results.push_back({
            std::string(group.name),
            std::string(group.explanation),
            ablated_scores,
            original_scores - ablated_scores,
        });
    }

    if (results.empty()) {
        throw std::invalid_argument(
            "Local ablation found no known feature groups to explain.");
    }

    return results;
}

} // namespace explainability
