#include "reporting/PredictionReporter.hpp"

#include "config/tag_invokes/tag_invokes.hpp"
#include "io/FileWriter.hpp"
#include "service/EdgeServiceClassifier.hpp"
#include "utils/math/TensorMath.hpp"
#include "utils/stream/OstreamFormatGuard.hpp"
#include "utils/utils.hpp"
#include <boost/json.hpp>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace reporting {

namespace {

constexpr int PERCENTAGE_PRECISION = 2;
constexpr int SCORE_PRECISION = 6;

namespace json = boost::json;

std::string csv_escape(std::string_view value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return std::string(value);
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char c : value) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

json::value json_value_or_null(const std::optional<std::string>& value) {
    return value.has_value() ? json::value(*value) : json::value(nullptr);
}

json::value json_value_or_null(const std::optional<double>& value) {
    return value.has_value() ? json::value(*value) : json::value(nullptr);
}

json::array ranking_at_k_json(const std::vector<statistics::RankingAtK>& ranking_at_k) {
    json::array values;
    values.reserve(ranking_at_k.size());

    for (const statistics::RankingAtK& ranking_metric : ranking_at_k) {
        json::object item;
        item["k"] = static_cast<std::uint64_t>(ranking_metric.k);
        item["precision"] = json_value_or_null(ranking_metric.precision);
        item["recall"] = json_value_or_null(ranking_metric.recall);
        values.emplace_back(std::move(item));
    }

    return values;
}
}
void validate_prediction_dimensions(
    const std::vector<candidate::CandidatePair>& candidates,
    const arma::Row<std::size_t>& predictions, const arma::rowvec& positive_scores) {
    if (predictions.n_elem != candidates.size() ||
        positive_scores.n_elem != candidates.size()) {
        throw std::invalid_argument(
            "Prediction report dimensions do not match candidate count.");
    }
}

} // namespace

void print_projection_stats(const ground_truth::ProjectionStats& stats) {
    const utils::OstreamFormatGuard format_guard(std::cout);

    std::cout << "Ground truth projection:\n"
              << "  Total unique dependencies: " << stats.total_dependencies << "\n"
              << "  Retained unique dependencies: " << stats.retained_dependencies;

    if (stats.total_dependencies > 0) {
        const double coverage = 100.0 * static_cast<double>(stats.retained_dependencies) /
                                static_cast<double>(stats.total_dependencies);
        std::cout << " (" << std::fixed << std::setprecision(PERCENTAGE_PRECISION)
                  << coverage << "% coverage)";
    }
    std::cout << "\n  Per-type retention:" << std::endl;

    for (const ground_truth::DependencyType type : ground_truth::all_dependency_types()) {
        const std::size_t index = static_cast<std::size_t>(type);
        std::cout << "    " << ground_truth::to_string(type) << ": "
                  << stats.retained_by_type[index] << " / " << stats.total_by_type[index]
                  << std::endl;
    }
PredictionWriteSummary write_positive_predictions(
    const std::string& path, const std::vector<candidate::CandidatePair>& candidates,
    const arma::Row<std::size_t>& predictions, const arma::rowvec& positive_scores,
    const INetworkGraphManager<BoostGraphTraits<Graph>>& graph_manager,
    const service::EdgeServiceClassifier<BoostGraphTraits<Graph>>* service_classifier) {
    validate_prediction_dimensions(candidates, predictions, positive_scores);

    FileWriter writer(path);
    if (service_classifier) {
        writer.write_line(
            "dependent_ip,dependency_ip,score,candidate_prior,candidate_reasons,"
            "service,service_conf,service_topk");
    } else {
        writer.write_line(
            "dependent_ip,dependency_ip,score,candidate_prior,candidate_reasons");
    }

    using Vertex = BoostGraphTraits<Graph>::Vertex;
    const std::unordered_map<std::string, Vertex>& ip_to_vertex =
        graph_manager.get_ip_to_vertex();

    std::size_t positive_count = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (predictions[i] != 1) {
            continue;
        }

        ++positive_count;
        const candidate::CandidatePair& candidate_pair = candidates[i];

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(SCORE_PRECISION);
        oss << csv_escape(candidate_pair.src) << "," << csv_escape(candidate_pair.dst)
            << "," << positive_scores[i] << ","
            << candidate_pair.prior_score << ","
            << csv_escape(utils::join(candidate_pair.reasons, ";"));

        if (service_classifier) {
            service::ServiceClassificationResult service_result;

            const std::unordered_map<std::string, Vertex>::const_iterator src_it =
                ip_to_vertex.find(candidate_pair.src);
            const std::unordered_map<std::string, Vertex>::const_iterator dst_it =
                ip_to_vertex.find(candidate_pair.dst);

            if (src_it != ip_to_vertex.end() && dst_it != ip_to_vertex.end()) {
                service_result = service_classifier->classify(
                    graph_manager, src_it->second, dst_it->second);
            }

            oss << ","
                << csv_escape(service::ServiceType::to_string(service_result.service))
                << "," << service_result.confidence
                << "," << csv_escape(service_result.top_k_string);
        }

        writer.write_line(oss.str());
    }

    return {positive_count, predictions.n_elem};
}

} // namespace reporting
