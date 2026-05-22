#include "reporting/PredictionReporter.hpp"

#include "config/tag_invokes/tag_invokes.hpp"
#include "io/FileWriter.hpp"
#include "service/EdgeServiceClassifier.hpp"
#include <boost/json.hpp>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace reporting {

namespace {

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

void validate_prediction_dimensions(
    const std::vector<std::pair<IPAddress, IPAddress>>& pairs,
    const arma::Row<std::size_t>& predictions, const arma::rowvec& positive_scores) {
    if (predictions.n_elem != pairs.size() ||
        positive_scores.n_elem != pairs.size()) {
        throw std::invalid_argument(
            "Prediction report dimensions do not match evaluated pair count.");
    }
}

} // namespace

void write_pair_scores(const std::string& path,
                       const std::vector<std::pair<IPAddress, IPAddress>>& pairs,
                       const arma::Row<std::size_t>& predictions,
                       const arma::rowvec& positive_scores,
                       const std::optional<arma::Row<std::size_t>>& labels) {
    validate_prediction_dimensions(pairs, predictions, positive_scores);
    if (labels.has_value() && labels->n_elem != pairs.size()) {
        throw std::invalid_argument(
            "Score report dimensions do not match evaluated pair count.");
    }

    FileWriter writer(path);
    writer.write_line("dependent_ip,dependency_ip,score,predicted_label,label");

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(SCORE_PRECISION);
        oss << csv_escape(pairs[i].first) << "," << csv_escape(pairs[i].second)
            << "," << positive_scores[i] << ","
            << predictions[i] << ",";
        if (labels.has_value()) {
            oss << (*labels)[i];
        }
        writer.write_line(oss.str());
    }
}

PredictionWriteSummary write_positive_predictions(
    const std::string& path,
    const std::vector<std::pair<IPAddress, IPAddress>>& pairs,
    const arma::Row<std::size_t>& predictions, const arma::rowvec& positive_scores,
    const INetworkGraphManager<BoostGraphTraits<Graph>>& graph_manager,
    const service::EdgeServiceClassifier<BoostGraphTraits<Graph>>* service_classifier) {
    validate_prediction_dimensions(pairs, predictions, positive_scores);

    FileWriter writer(path);
    if (service_classifier) {
        writer.write_line(
            "dependent_ip,dependency_ip,score,service,service_conf,service_topk");
    } else {
        writer.write_line("dependent_ip,dependency_ip,score");
    }

    using Vertex = BoostGraphTraits<Graph>::Vertex;
    const std::unordered_map<std::string, Vertex>& ip_to_vertex =
        graph_manager.get_ip_to_vertex();

    std::size_t positive_count = 0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        if (predictions[i] != 1) {
            continue;
        }

        ++positive_count;
        const std::pair<IPAddress, IPAddress>& pair = pairs[i];

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(SCORE_PRECISION);
        oss << csv_escape(pair.first) << "," << csv_escape(pair.second)
            << "," << positive_scores[i];

        if (service_classifier) {
            service::ServiceClassificationResult service_result;

            const std::unordered_map<std::string, Vertex>::const_iterator src_it =
                ip_to_vertex.find(pair.first);
            const std::unordered_map<std::string, Vertex>::const_iterator dst_it =
                ip_to_vertex.find(pair.second);

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

void write_run_manifest(const RunManifest& manifest) {
    FileWriter writer(manifest.path);

    json::object report;
    report["mode"] = manifest.mode;
    report["config_path"] = json_value_or_null(manifest.config_path);
    report["data_path"] = manifest.data_path;
    report["classifier_path"] = manifest.classifier_path;
    report["output_path"] = manifest.output_path;
    report["reference_path"] = json_value_or_null(manifest.reference_path);
    report["seed"] = manifest.config.SEED;
    report["num_threads"] = manifest.config.NUM_THREADS.value_or(0);
    report["retained_vertices"] =
        static_cast<std::uint64_t>(manifest.graph_manager.get_vertex_count());
    report["retained_edges"] =
        static_cast<std::uint64_t>(manifest.graph_manager.get_edge_count());
    report["evaluated_pairs"] =
        static_cast<std::uint64_t>(manifest.evaluated_pair_count);
    report["feature_config"] = json::value_from(manifest.config.FEATURE_CONFIG);
    report["rf_params"] = json::value_from(manifest.config.RF_PARAMS);

    writer.write_line(json::serialize(report));
}

void write_metrics_report(const std::string& path, const statistics::Metrics& metrics,
                          std::size_t positive_predictions,
                          std::size_t total_predictions) {
    FileWriter writer(path);

    json::object report;
    report["positive_predictions"] = static_cast<std::uint64_t>(positive_predictions);
    report["total_predictions"] = static_cast<std::uint64_t>(total_predictions);
    report["accuracy"] = metrics.accuracy;
    report["precision"] = metrics.precision;
    report["recall"] = metrics.recall;
    report["f1"] = metrics.f1_score;
    report["roc_auc"] = json_value_or_null(metrics.roc_auc);
    report["average_precision"] = json_value_or_null(metrics.average_precision);
    report["mrr"] = json_value_or_null(metrics.mean_reciprocal_rank);
    report["ranking_at_k"] = ranking_at_k_json(metrics.ranking_at_k);

    writer.write_line(json::serialize(report));
}

} // namespace reporting
