#pragma once

#include "BidirectionalFeatureExtractor.hpp"
#include "config/config.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

template <typename GraphTraits>
BidirectionalFeatureExtractor::BidirectionalFeatures
BidirectionalFeatureExtractor::extract_all_features(
    const IGraphManager<GraphTraits> &graph_manager,
    const typename GraphTraits::Vertex &v1, const typename GraphTraits::Vertex &v2,
    const FeatureConfig &config) {
    BidirectionalFeatures result;

    // Collect edges once from both directions
    auto edges_v1_to_v2 = collect_edges_directional(graph_manager, v1, v2);
    auto edges_v2_to_v1 = collect_edges_directional(graph_manager, v2, v1);

    if (config.bidirectional_has_flows) {
        result.has_bidirectional_flows =
            !edges_v1_to_v2.empty() && !edges_v2_to_v1.empty();
    }

    if (edges_v1_to_v2.empty() || edges_v2_to_v1.empty()) {
        return result;
    }

    if (config.bidirectional_response_time) {
        auto response_time_forward =
            calculate_avg_response_time(edges_v1_to_v2, edges_v2_to_v1);
        auto response_time_reverse =
            calculate_avg_response_time(edges_v2_to_v1, edges_v1_to_v2);

        // Only set feature if at least one direction has valid response times
        if (response_time_forward.has_value() && response_time_reverse.has_value()) {
            result.avg_response_time =
                (*response_time_forward + *response_time_reverse) / 2.0;
        } else if (response_time_forward.has_value()) {
            result.avg_response_time = *response_time_forward;
        } else if (response_time_reverse.has_value()) {
            result.avg_response_time = *response_time_reverse;
        }
        // else: both directions have no valid matches, leave as nullopt
    }

    if (config.bidirectional_request_ratio) {
        std::size_t forward_count = edges_v1_to_v2.size();
        std::size_t reverse_count = edges_v2_to_v1.size();
        std::size_t max_count = std::max(forward_count, reverse_count);
        std::size_t min_count = std::min(forward_count, reverse_count);

        // min_count is guaranteed > 0 due to early exit check above
        result.request_response_ratio =
            static_cast<double>(max_count) / static_cast<double>(min_count);
    }

    if (config.bidirectional_asymmetry) {
        result.directional_asymmetry =
            calculate_directional_asymmetry(edges_v1_to_v2, edges_v2_to_v1);
    }

    return result;
}

template <typename GraphTraits>
std::vector<typename GraphTraits::EdgeProperties>
BidirectionalFeatureExtractor::collect_edges_directional(
    const IGraphManager<GraphTraits> &graph_manager,
    const typename GraphTraits::Vertex &src, const typename GraphTraits::Vertex &dst) {
    using EdgeProperties = typename GraphTraits::EdgeProperties;
    std::vector<EdgeProperties> edges;

    // Find all edges from src to dst
    auto [edge_it, edge_end] = graph_manager.get_out_edges(src);
    for (; edge_it != edge_end; ++edge_it) {
        if (graph_manager.get_target_vertex(*edge_it) == dst) {
            edges.push_back(graph_manager.get_edge_properties(*edge_it));
        }
    }

    // Sort by start timestamp for temporal analysis
    std::sort(edges.begin(), edges.end(),
              [](const EdgeProperties &a, const EdgeProperties &b) {
                  return a.start_timestamp < b.start_timestamp;
              });

    return edges;
}

template <typename EdgeProperties>
std::optional<double> BidirectionalFeatureExtractor::calculate_avg_response_time(
    const std::vector<EdgeProperties> &forward_edges,
    const std::vector<EdgeProperties> &reverse_edges) {
    if (forward_edges.empty() || reverse_edges.empty()) {
        return std::nullopt;
    }

    std::vector<double> response_times;
    response_times.reserve(forward_edges.size());

    std::size_t reverse_index = 0;

    // For each forward edge, find the nearest subsequent reverse edge
    for (const auto &forward_edge : forward_edges) {
        auto response_time =
            find_response_time(forward_edge, reverse_edges, reverse_index);
        if (response_time.has_value()) {
            response_times.push_back(response_time.value());
        }
    }

    if (response_times.empty()) {
        return std::nullopt; // No valid response times found
    }

    // Calculate mean response time
    double sum = std::accumulate(response_times.begin(), response_times.end(), 0.0);
    return sum / response_times.size();
}

template <typename EdgeProperties>
std::optional<double> BidirectionalFeatureExtractor::find_response_time(
    const EdgeProperties &forward_edge, const std::vector<EdgeProperties> &reverse_edges,
    std::size_t &start_index) {
    // Search for the first reverse edge that starts
    // after the forward edge starts
    while (start_index < reverse_edges.size() &&
           reverse_edges[start_index].start_timestamp <= forward_edge.start_timestamp) {
        ++start_index;
    }

    for (std::size_t i = start_index; i < reverse_edges.size(); ++i) {
        const auto &reverse_edge = reverse_edges[i];

        auto response_duration =
            reverse_edge.start_timestamp - forward_edge.start_timestamp;

        if (response_duration <= MAX_RESPONSE_TIME) {
            start_index = i + 1;
            return response_duration.count();
        }

        break;
    }

    return std::nullopt;
}

template <typename EdgeProperties>
double BidirectionalFeatureExtractor::calculate_directional_asymmetry(
    const std::vector<EdgeProperties> &forward_edges,
    const std::vector<EdgeProperties> &reverse_edges) {
    auto avg_duration = [](const std::vector<EdgeProperties> &edges) -> double {
        if (edges.empty())
            return 0.0;

        double total = 0.0;
        for (const auto &edge : edges) {
            total += (edge.end_timestamp - edge.start_timestamp).count();
        }
        return total / edges.size();
    };

    double forward_avg_duration = avg_duration(forward_edges);
    double reverse_avg_duration = avg_duration(reverse_edges);

    std::size_t forward_count = forward_edges.size();
    std::size_t reverse_count = reverse_edges.size();
    std::size_t total_count = forward_count + reverse_count;
    double count_asymmetry = std::abs(static_cast<double>(forward_count) -
                                      static_cast<double>(reverse_count)) /
                             static_cast<double>(total_count);

    double duration_asymmetry = 0.0;
    if (forward_avg_duration + reverse_avg_duration > 0) {
        duration_asymmetry = std::abs(forward_avg_duration - reverse_avg_duration) /
                             (forward_avg_duration + reverse_avg_duration);
    }

    return COUNT_WEIGHT * count_asymmetry + DURATION_WEIGHT * duration_asymmetry;
}
