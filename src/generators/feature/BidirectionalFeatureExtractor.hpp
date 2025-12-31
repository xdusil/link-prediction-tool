#pragma once

#include "FeatureConfig.hpp"
#include "graph/IGraphManager.hpp"
#include "utils/FlowDataCollector.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <vector>

/**
 * @brief Extractor for bidirectional flow asymmetry features.
 *
 * Analyzes request-response patterns and flow asymmetry between node pairs.
 */
class BidirectionalFeatureExtractor {
public:
    /**
     * @brief Result structure containing bidirectional features.
     */
    struct Features {
        std::optional<double> response_time;
        std::optional<double> request_ratio;
        std::optional<double> direction_asymmetry;
        std::optional<double> causality_score;
    };

    /**
     * @brief Extract bidirectional features for a vertex pair.
     *
     * @param graph_manager The graph manager instance.
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param config The feature configuration.
     * @return The extracted bidirectional features. If a feature is not computable,
     * its value will be std::nullopt.
     */
    template <typename GraphTraits>
    static Features extract(const IGraphManager<GraphTraits>& graph_manager,
                            const typename GraphTraits::Vertex& src,
                            const typename GraphTraits::Vertex& dst,
                            const FeatureConfig& config);

private:
    /**
     * @brief Compute average response time.
     *
     * It is defined as the average time difference between the end of
     * a request and the start of an response. It does not necessarily
     * require strict pairing of requests and responses (refers to closest subsequent
     * response).
     *
     * @param flow_data The aggregated flow data.
     * @return Average response time in milliseconds.
     */
    template <typename FlowData>
    static double compute_avg_response_time(const FlowData& flow_data);

    /**
     * @brief Compute causality score.
     *
     * It is defined based on counting the number of times a forward flow
     * (request) is followed by a reverse flow (response) versus the opposite.
     *
     * @param flow_data The aggregated flow data.
     * @return Causality score. Positive values indicate forward->reverse dominance,
     * negative values indicate reverse->forward dominance.
     */
    template <typename FlowData>
    static double compute_causality(const FlowData& flow_data);
};

// ====================================================================================================

template <typename GraphTraits>
BidirectionalFeatureExtractor::Features
BidirectionalFeatureExtractor::extract(const IGraphManager<GraphTraits>& graph_manager,
                                       const typename GraphTraits::Vertex& src,
                                       const typename GraphTraits::Vertex& dst,
                                       const FeatureConfig& config) {

    Features result;

    using Fields = FlowDataCollector::Fields;
    Fields needs = Fields::None;

    if (config.flow_response_time || config.flow_causality_score) {
        needs |= Fields::ResponseTiming; // ForwardEndTimes | ReverseStartTimes
    }
    if (config.flow_request_ratio || config.flow_direction_asymmetry) {
        needs |= Fields::StartTimesOnly; // ForwardStartTimes | ReverseStartTimes
    }

    const FlowDataCollector::FlowData flow_data =
        FlowDataCollector::collect(graph_manager, src, dst, needs);

    const std::size_t total_flows = flow_data.total_count();

    // Response time: average time between request end and response start
    if (config.flow_response_time && flow_data.has_flows()) {
        result.response_time = compute_avg_response_time(flow_data);
    }

    // Request ratio: proportion of forward flows vs total
    if (config.flow_request_ratio && flow_data.has_flows()) {
        result.request_ratio = static_cast<double>(flow_data.forward_count()) /
                               static_cast<double>(total_flows);
    }

    // Direction asymmetry: (forward - reverse) / total
    if (config.flow_direction_asymmetry && flow_data.has_flows()) {
        double diff = static_cast<double>(flow_data.forward_count()) -
                      static_cast<double>(flow_data.reverse_count());
        result.direction_asymmetry = diff / static_cast<double>(total_flows);
    }

    // Causality score: temporal ordering pattern strength
    if (config.flow_causality_score && flow_data.has_flows()) {
        result.causality_score = compute_causality(flow_data);
    }

    return result;
}

template <typename FlowData>
double
BidirectionalFeatureExtractor::compute_avg_response_time(const FlowData& flow_data) {

    if (!flow_data.is_bidirectional()) {
        return 0.0;
    }

    auto forward_end_times = flow_data.forward_end_times();
    auto reverse_start_times = flow_data.reverse_start_times();

    std::sort(forward_end_times.begin(), forward_end_times.end());
    std::sort(reverse_start_times.begin(), reverse_start_times.end());

    double total_response_time = 0.0;
    std::size_t matched_pairs = 0;

    // For each forward flow end, find the next reverse flow start
    for (const auto& forward_end : forward_end_times) {
        auto it = std::lower_bound(reverse_start_times.begin(), reverse_start_times.end(),
                                   forward_end);
        if (it != reverse_start_times.end()) {
            total_response_time += static_cast<double>((*it - forward_end).count());
            ++matched_pairs;
        }
    }

    return (matched_pairs > 0) ? total_response_time / static_cast<double>(matched_pairs)
                               : 0.0;
}

template <typename FlowData>
double BidirectionalFeatureExtractor::compute_causality(const FlowData& flow_data) {

    // Create time-ordered events with direction labels
    struct TimedEvent {
        std::chrono::milliseconds timestamp;
        bool is_forward_flow;
    };
    std::vector<TimedEvent> all_events;

    // Mark forward flow completions
    for (const auto& end_time : flow_data.forward_end_times()) {
        all_events.push_back({end_time, true});
    }

    // Mark reverse flow starts
    for (const auto& start_time : flow_data.reverse_start_times()) {
        all_events.push_back({start_time, false});
    }

    if (all_events.size() < 2) {
        return 0.0;
    }

    std::sort(all_events.begin(), all_events.end(),
              [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });

    // Count transition patterns
    std::size_t forward_then_reverse = 0; // Request  -> Response pattern
    std::size_t reverse_then_forward = 0; // Response -> Request pattern

    for (std::size_t i = 1; i < all_events.size(); ++i) {
        bool prev_is_forward = all_events[i - 1].is_forward_flow;
        bool curr_is_forward = all_events[i].is_forward_flow;

        if (prev_is_forward && !curr_is_forward) {
            ++forward_then_reverse; // Forward flow followed by reverse
        } else if (!prev_is_forward && curr_is_forward) {
            ++reverse_then_forward; // Reverse flow followed by forward
        }
    }

    std::size_t total_transitions = forward_then_reverse + reverse_then_forward;
    if (total_transitions == 0) {
        return 0.0;
    }

    // Positive score = forward->reverse dominant (request-response pattern)
    // Negative score = reverse->forward dominant
    return (static_cast<double>(forward_then_reverse) -
            static_cast<double>(reverse_then_forward)) /
           static_cast<double>(total_transitions);
}
