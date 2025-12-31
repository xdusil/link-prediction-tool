#pragma once

#include "FeatureConfig.hpp"
#include "graph/IGraphManager.hpp"
#include "utils/FlowDataCollector.hpp"
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
    /**
     * @brief Compute inter-arrival features.
     *
     * @param timestamps The sorted flow start timestamps.
     * @param config The feature configuration.
     * @param result The result structure to populate.
     */
    static void compute_interarrival_features(
        const std::vector<std::chrono::milliseconds>& timestamps,
        const FeatureConfig& config, Features& result);

    /**
     * @brief Compute cross-correlation features.
     *
     * @param forward The forward direction start timestamps.
     * @param reverse The reverse direction start timestamps.
     * @param result The result structure to populate.
     */
    static void
    compute_cross_correlation(const std::vector<std::chrono::milliseconds>& forward,
                              const std::vector<std::chrono::milliseconds>& reverse,
                              Features& result);

    /**
     * @brief Compute spike score feature.
     *
     * @param forward The forward direction end timestamps.
     * @param reverse The reverse direction start timestamps.
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

    using Fields = FlowDataCollector::Fields;
    Fields needs = Fields::None;

    if (config.time_avg_duration) {
        needs |= Fields::Duration;
    }
    if (config.time_avg_interarrival || config.time_regularity ||
        config.time_direction_bias || config.time_initiation_order ||
        config.time_crosscorr_peak) {
        needs |= Fields::StartTimesOnly;
    }
    if (config.time_spike_score) {
        needs |= Fields::ResponseTiming;  // ForwardEndTimes | ReverseStartTimes
    }

    const FlowDataCollector::FlowData flow_data =
        FlowDataCollector::collect(graph_manager, src, dst, needs);

    // Average duration
    if (config.time_avg_duration && flow_data.has_flows()) {
        result.avg_duration =
            flow_data.total_duration_ms() / static_cast<double>(flow_data.total_count());
    }

    const std::vector<std::chrono::milliseconds> all_timestamps =
        FlowDataCollector::get_sorted_start_times(flow_data);

    compute_interarrival_features(all_timestamps, config, result);

    // Direction bias: (forward - reverse) / total
    if (config.time_direction_bias && flow_data.has_flows()) {
        result.direction_bias = (static_cast<double>(flow_data.forward_count()) -
                                 static_cast<double>(flow_data.reverse_count())) /
                                static_cast<double>(flow_data.total_count());
    }

    // Initiation order: who initiated communication first?
    if (config.time_initiation_order) {
        const auto& fwd_times = flow_data.forward_start_times();
        const auto& rev_times = flow_data.reverse_start_times();
        if (!fwd_times.empty() && !rev_times.empty()) {
            auto first_forward = *std::min_element(fwd_times.begin(), fwd_times.end());
            auto first_reverse = *std::min_element(rev_times.begin(), rev_times.end());

            if (first_forward < first_reverse) {
                result.initiation_order = 1.0; // src initiated
            } else if (first_reverse < first_forward) {
                result.initiation_order = -1.0; // dst initiated
            } else {
                result.initiation_order = 0.0; // simultaneous
            }
        } else if (!fwd_times.empty()) {
            result.initiation_order = 1.0; // only forward flows
        } else if (!rev_times.empty()) {
            result.initiation_order = -1.0; // only reverse flows
        }
    }

    // Cross-correlation: timing correlation between forward and reverse flows
    if (config.time_crosscorr_peak && flow_data.is_bidirectional()) {
        compute_cross_correlation(flow_data.forward_start_times(),
                                  flow_data.reverse_start_times(), result);
    }

    // Spike score: response time consistency
    if (config.time_spike_score && flow_data.is_bidirectional()) {
        compute_spike_score(flow_data.forward_end_times(),
                            flow_data.reverse_start_times(), result);
    }

    return result;
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
