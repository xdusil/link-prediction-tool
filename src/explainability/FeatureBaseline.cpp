#include "FeatureBaseline.hpp"

#include "io/FileReader.hpp"
#include "io/FileWriter.hpp"
#include <algorithm>
#include <boost/json.hpp>
#include <cmath>
#include <stdexcept>

namespace explainability {

namespace json = boost::json;

namespace {

double median_of_row(const arma::fmat& features, std::size_t row_index) {
    std::vector<float> values(features.n_cols);
    for (std::size_t col = 0; col < features.n_cols; ++col) {
        values[col] = features(row_index, col);
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    if (values.size() % 2 == 1) {
        return values[mid];
    }

    const float upper = values[mid];
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return (static_cast<double>(values[mid - 1]) + static_cast<double>(upper)) / 2.0;
}

} // namespace

std::string feature_baseline_path(const std::string& classifier_path) {
    return classifier_path + ".feature_baselines.json";
}

FeatureBaseline compute_feature_baseline(
    const arma::fmat& features,
    const std::vector<std::string>& feature_names) {
    if (features.n_rows != feature_names.size()) {
        throw std::invalid_argument(
            "Feature baseline requires one feature name per feature row.");
    }
    if (features.n_cols == 0) {
        throw std::invalid_argument("Cannot compute feature baseline from empty data.");
    }

    FeatureBaseline baseline;
    baseline.feature_names = feature_names;
    baseline.medians.reserve(features.n_rows);

    for (std::size_t row = 0; row < features.n_rows; ++row) {
        const double median = median_of_row(features, row);
        if (!std::isfinite(median)) {
            throw std::invalid_argument("Feature baseline median is not finite.");
        }
        baseline.medians.push_back(median);
    }

    return baseline;
}

void save_feature_baseline(const std::string& path, const FeatureBaseline& baseline) {
    if (baseline.feature_names.size() != baseline.medians.size()) {
        throw std::invalid_argument(
            "Feature baseline names and medians have different sizes.");
    }

    json::object root;
    json::array features;
    features.reserve(baseline.feature_names.size());

    for (std::size_t i = 0; i < baseline.feature_names.size(); ++i) {
        if (!std::isfinite(baseline.medians[i])) {
            throw std::invalid_argument(
                "Cannot save feature baseline with non-finite median.");
        }

        json::object item;
        item["name"] = baseline.feature_names[i];
        item["median"] = baseline.medians[i];
        features.emplace_back(std::move(item));
    }

    root["features"] = std::move(features);

    FileWriter writer(path);
    writer.write_line(json::serialize(root));
}

FeatureBaseline load_feature_baseline(const std::string& path) {
    FileReader reader(path);
    std::string content;
    reader.read_all(content);

    const json::value parsed = json::parse(content);
    const json::object& root = parsed.as_object();
    const json::array& features = root.at("features").as_array();

    FeatureBaseline baseline;
    baseline.feature_names.reserve(features.size());
    baseline.medians.reserve(features.size());

    for (const json::value& value : features) {
        const json::object& item = value.as_object();
        baseline.feature_names.emplace_back(item.at("name").as_string().c_str());
        const double median = item.at("median").as_double();
        if (!std::isfinite(median)) {
            throw std::invalid_argument(
                "Loaded feature baseline contains non-finite median.");
        }
        baseline.medians.push_back(median);
    }

    return baseline;
}

void validate_feature_baseline(
    const FeatureBaseline& baseline,
    const std::vector<std::string>& feature_names) {
    if (baseline.feature_names != feature_names ||
        baseline.medians.size() != feature_names.size()) {
        throw std::invalid_argument(
            "Feature baseline does not match the active feature configuration.");
    }
}

} // namespace explainability
