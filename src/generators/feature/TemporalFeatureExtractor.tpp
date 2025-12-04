#pragma once

#include "TemporalFeatureExtractor.hpp"
#include "config/config.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

template <typename GraphTraits>
TemporalFeatureExtractor::TemporalFeatures TemporalFeatureExtractor::extract_all_features(
    const IGraphManager<GraphTraits> &graph_manager,
    const typename GraphTraits::Vertex &v1, const typename GraphTraits::Vertex &v2,
    const FeatureConfig &config) {
    TemporalFeatures result;

    // Determine what needs to be collected
    bool need_timestamps = config.temporal_avg_inter_arrival || config.temporal_var_inter_arrival ||
                           config.temporal_regularity || config.temporal_concentration;
    bool need_durations = config.temporal_avg_duration;

    std::vector<std::chrono::milliseconds> timestamps;
    double total_duration = 0.0;
    std::size_t edge_count = 0;

    auto collect = [&](const auto &v_src, const auto &v_dst) {
        auto [it, end] = graph_manager.get_out_edges(v_src);
        for (; it != end; ++it) {
            if (graph_manager.get_target_vertex(*it) == v_dst) {
                const auto &edge = graph_manager.get_edge_properties(*it);
                if (need_timestamps) {
                    timestamps.push_back(edge.start_timestamp);
                }
                if (need_durations) {
                    total_duration += (edge.end_timestamp - edge.start_timestamp).count();
                    ++edge_count;
                }
            }
        }
    };

    collect(v1, v2);
    collect(v2, v1);

    // Avg duration
    if (config.temporal_avg_duration && edge_count > 0) {
        result.avg_duration = total_duration / edge_count;
    }

    if (timestamps.size() < 2) {
        return result;
    }

    std::sort(timestamps.begin(), timestamps.end());

    // Compute inter-arrival times if any feature needs them
    std::vector<double> inter_arrivals;
    if (config.temporal_avg_inter_arrival || config.temporal_var_inter_arrival ||
        config.temporal_regularity) {
        inter_arrivals.reserve(timestamps.size() - 1);
        for (std::size_t i = 1; i < timestamps.size(); ++i) {
            inter_arrivals.push_back((timestamps[i] - timestamps[i - 1]).count());
        }

        double mean = std::accumulate(inter_arrivals.begin(), inter_arrivals.end(), 0.0) /
                      inter_arrivals.size();

        if (config.temporal_avg_inter_arrival) {
            result.avg_inter_arrival = mean;
        }

        if (config.temporal_var_inter_arrival || config.temporal_regularity) {
            double variance = 0.0;
            for (double val : inter_arrivals) {
                variance += (val - mean) * (val - mean);
            }
            variance /= inter_arrivals.size();

            if (config.temporal_var_inter_arrival) {
                result.var_inter_arrival = variance;
            }

            if (config.temporal_regularity) {
                result.regularity = mean >= 1e-9 ? std::sqrt(variance) / mean : 0.0;
            }
        }
    }

    // Temporal concentration
    if (config.temporal_concentration) {
        constexpr auto TIME_WINDOW = CONCENTRATION_TIME_WINDOW;
        std::size_t flows_with_neighbors = 0;

        for (std::size_t i = 0; i < timestamps.size(); ++i) {
            for (std::size_t j = i + 1; j < timestamps.size(); ++j) {
                auto gap = (timestamps[j] - timestamps[i]).count();
                if (gap > TIME_WINDOW.count()) {
                    break; // Sorted: all subsequent timestamps will be farther
                }
                ++flows_with_neighbors;
                break; // Found at least one neighbor within window
            }
        }

        result.temporal_concentration =
            static_cast<double>(flows_with_neighbors) / timestamps.size();
    }

    return result;
}
