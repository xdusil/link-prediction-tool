#pragma once

#include "FeatureConfig.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

/**
 * @brief Extractor for temporal causality features.
 *
 * Analyzes timing patterns between node pairs to detect dependency patterns.
 */
class TemporalFeatureExtractor {
public:
    // Configuration constants
    static constexpr auto kCorrelationTolerance = std::chrono::milliseconds{500};
    static constexpr int kMaxCorrelationLag = 5;

    /**
     * @brief Result structure containing temporal features.
     */
    struct Features {
        std::optional<double> avg_duration;
        std::optional<double> avg_interarrival;
        std::optional<double> regularity;
        std::optional<double> direction_bias;
        std::optional<double> initiation_order;
        std::optional<double> crosscorr_max;
        std::optional<double> crosscorr_lag;
        std::optional<double> spike_score;
    };

    /**
     * @brief Extract temporal features for a vertex pair.
     *
     * @param graph_manager The graph manager instance.
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param config The feature configuration.
     * @return The extracted temporal features. If a feature is not computable,
     * its value will be std::nullopt.
     */
    template <typename GraphTraits>
    static Features extract(const IGraphManager<GraphTraits>& graph_manager,
                            const typename GraphTraits::Vertex& src,
                            const typename GraphTraits::Vertex& dst,
                            const FeatureConfig& config);

private:
    struct EdgeData {
        std::vector<std::chrono::milliseconds> all_timestamps;
        std::vector<std::chrono::milliseconds> forward_timestamps;
        std::vector<std::chrono::milliseconds> reverse_timestamps;
        double total_duration = 0.0;
    };

    /**
     * @brief Collect edge data between two vertices.
     *
     * @param graph_manager The graph manager instance.
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param config The feature configuration.
     * @return The collected edge data, populated based on the enabled features.
     */
    template <typename GraphTraits>
    static EdgeData collect_edge_data(const IGraphManager<GraphTraits>& graph_manager,
                                      const typename GraphTraits::Vertex& src,
                                      const typename GraphTraits::Vertex& dst,
                                      const FeatureConfig& config);

    /**
     * @brief Compute inter-arrival features.
     *
     * @param timestamps The timestamps to analyze.
     * @param config The feature configuration.
     * @param result The result structure to populate.
     */
    static void compute_interarrival_features(
        const std::vector<std::chrono::milliseconds>& timestamps,
        const FeatureConfig& config, Features& result);

    /**
     * @brief Compute cross-correlation features.
     *
     * @param forward The forward direction timestamps.
     * @param reverse The reverse direction timestamps.
     * @param result The result structure to populate.
     */
    static void
    compute_cross_correlation(const std::vector<std::chrono::milliseconds>& forward,
                              const std::vector<std::chrono::milliseconds>& reverse,
                              Features& result);

    /**
     * @brief Compute spike score feature.
     *
     * @param forward The forward direction timestamps.
     * @param reverse The reverse direction timestamps.
     * @param result The result structure to populate.
     */
    static void compute_spike_score(const std::vector<std::chrono::milliseconds>& forward,
                                    const std::vector<std::chrono::milliseconds>& reverse,
                                    Features& result);
};

// ====================================================================================================

template <typename GraphTraits>
TemporalFeatureExtractor::Features
TemporalFeatureExtractor::extract(const IGraphManager<GraphTraits>& graph_manager,
                                  const typename GraphTraits::Vertex& src,
                                  const typename GraphTraits::Vertex& dst,
                                  const FeatureConfig& config) {

    Features result;
    EdgeData data = collect_edge_data(graph_manager, src, dst, config);
    std::size_t edge_count = data.all_timestamps.size();

    // Average duration
    if (config.time_avg_duration && edge_count > 0) {
        result.avg_duration = data.total_duration / static_cast<double>(edge_count);
    }

    if (data.all_timestamps.size() < 2) {
        return result;
    }

    std::sort(data.all_timestamps.begin(), data.all_timestamps.end());
    compute_interarrival_features(data.all_timestamps, config, result);

    // Direction bias
    if (config.time_direction_bias) {
        if (edge_count > 0) {
            result.direction_bias = (static_cast<double>(data.forward_timestamps.size()) -
                                     static_cast<double>(data.reverse_timestamps.size())) /
                                    static_cast<double>(edge_count);
        }
    }

    // Initiation order
    if (config.time_initiation_order) {
        if (!data.forward_timestamps.empty() && !data.reverse_timestamps.empty()) {
            auto first_fwd = *std::min_element(data.forward_timestamps.begin(),
                                               data.forward_timestamps.end());
            auto first_rev = *std::min_element(data.reverse_timestamps.begin(),
                                               data.reverse_timestamps.end());
            if (first_fwd < first_rev) {
                result.initiation_order = 1.0;
            } else if (first_rev < first_fwd) {
                result.initiation_order = -1.0;
            } else {
                result.initiation_order = 0.0;
            }
        } else if (!data.forward_timestamps.empty()) {
            result.initiation_order = 1.0;
        } else if (!data.reverse_timestamps.empty()) {
            result.initiation_order = -1.0;
        }
    }

    // Cross-correlation
    if (config.time_crosscorr_peak && !data.forward_timestamps.empty() &&
        !data.reverse_timestamps.empty()) {
        compute_cross_correlation(data.forward_timestamps, data.reverse_timestamps,
                                  result);
    }

    // Spike score
    if (config.time_spike_score && !data.forward_timestamps.empty() &&
        !data.reverse_timestamps.empty()) {
        compute_spike_score(data.forward_timestamps, data.reverse_timestamps, result);
    }

    return result;
}

template <typename GraphTraits>
TemporalFeatureExtractor::EdgeData TemporalFeatureExtractor::collect_edge_data(
    const IGraphManager<GraphTraits>& graph_manager,
    const typename GraphTraits::Vertex& src, const typename GraphTraits::Vertex& dst,
    const FeatureConfig& config) {

    EdgeData data;

    auto collect_direction = [&](const auto& from, const auto& to, bool is_forward) {
        auto [it, end] = graph_manager.get_out_edges(from);
        for (; it != end; ++it) {
            if (graph_manager.get_target_vertex(*it) != to)
                continue;

            const auto& edge = graph_manager.get_edge_properties(*it);

            data.all_timestamps.push_back(edge.start_timestamp);

            auto& vec =
                is_forward ? data.forward_timestamps : data.reverse_timestamps;
            vec.push_back(edge.start_timestamp);

            data.total_duration += static_cast<double>(
                (edge.end_timestamp - edge.start_timestamp).count());
        }
    };

    collect_direction(src, dst, true);
    collect_direction(dst, src, false);

    return data;
}

inline void TemporalFeatureExtractor::compute_interarrival_features(
    const std::vector<std::chrono::milliseconds>& timestamps, const FeatureConfig& config,
    Features& result) {

    if (timestamps.size() < 2)
        return;
    if (!config.time_avg_interarrival && !config.time_regularity)
        return;

    std::vector<double> inter_arrivals;
    inter_arrivals.reserve(timestamps.size() - 1);

    for (std::size_t i = 1; i < timestamps.size(); ++i) {
        inter_arrivals.push_back(
            static_cast<double>((timestamps[i] - timestamps[i - 1]).count()));
    }

    double mean = std::accumulate(inter_arrivals.begin(), inter_arrivals.end(), 0.0) /
                  static_cast<double>(inter_arrivals.size());

    if (config.time_avg_interarrival) {
        result.avg_interarrival = mean;
    }

    if (config.time_regularity) {
        double variance = 0.0;
        for (double val : inter_arrivals) {
            double diff = val - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(inter_arrivals.size());
        result.regularity = (mean >= 1e-9) ? std::sqrt(variance) / mean : 0.0;
    }
}

inline void TemporalFeatureExtractor::compute_cross_correlation(
    const std::vector<std::chrono::milliseconds>& forward,
    const std::vector<std::chrono::milliseconds>& reverse, Features& result) {

    double best_corr = -1.0;
    int best_lag = 0;

    for (int lag = -kMaxCorrelationLag; lag <= kMaxCorrelationLag; ++lag) {
        int matches = 0;
        for (const auto& fwd_ts : forward) {
            auto target_ts = fwd_ts + std::chrono::milliseconds{lag * 100};

            auto closest = std::min_element(reverse.begin(), reverse.end(),
                                            [&target_ts](const auto& a, const auto& b) {
                                                return std::abs((a - target_ts).count()) <
                                                       std::abs((b - target_ts).count());
                                            });

            if (closest != reverse.end() && std::abs((*closest - target_ts).count()) <
                                                kCorrelationTolerance.count()) {
                ++matches;
            }
        }

        double corr = static_cast<double>(matches) / static_cast<double>(forward.size());
        if (corr > best_corr) {
            best_corr = corr;
            best_lag = lag;
        }
    }

    result.crosscorr_max = best_corr;
    result.crosscorr_lag = static_cast<double>(best_lag);
}

inline void TemporalFeatureExtractor::compute_spike_score(
    const std::vector<std::chrono::milliseconds>& forward,
    const std::vector<std::chrono::milliseconds>& reverse, Features& result) {

    std::vector<double> delays;
    delays.reserve(std::min(forward.size(), reverse.size()));

    auto reverse_sorted = reverse;
    std::sort(reverse_sorted.begin(), reverse_sorted.end());

    for (const auto& fwd : forward) {
        auto it = std::lower_bound(reverse_sorted.begin(), reverse_sorted.end(), fwd);
        if (it != reverse_sorted.end()) {
            delays.push_back(static_cast<double>((*it - fwd).count()));
        }
    }

    if (delays.empty())
        return;

    double mean = std::accumulate(delays.begin(), delays.end(), 0.0) /
                  static_cast<double>(delays.size());

    double variance = 0.0;
    for (double d : delays) {
        double diff = d - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(delays.size());

    result.spike_score = (mean > 1e-6) ? 1.0 / (1.0 + std::sqrt(variance) / mean) : 0.0;
}
