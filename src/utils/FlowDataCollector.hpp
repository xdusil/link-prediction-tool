#pragma once

#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <chrono>
#include <vector>

/**
 * @brief Utility for collecting aggregated flow data between vertex pairs.
 *
 * This class collects and aggregates information from ALL flows (edges) between
 * two vertices in both directions. It provides a unified data structure for
 * temporal and bidirectional flow feature extraction.
 *
 * "Flow" here refers to individual network flows (represented as edges
 * in the graph). A vertex pair typically has multiple flows in each direction.
 */
class FlowDataCollector {
public:
    /**
     * @brief Complete aggregated flow data for a vertex pair.
     *
     * All fields are always populated by collect_all().
     * Forward = src -> dst, Reverse = dst -> src.
     */
    struct FlowData {
        // Forward direction (src -> dst)
        std::vector<std::chrono::milliseconds> forward_start_times;
        std::vector<std::chrono::milliseconds> forward_end_times;

        // Reverse direction (dst -> src)
        std::vector<std::chrono::milliseconds> reverse_start_times;
        std::vector<std::chrono::milliseconds> reverse_end_times;

        // Aggregated metrics
        double total_duration_ms = 0.0;

        /**
         * @brief Get number of forward flows.
         */
        std::size_t forward_count() const { return forward_start_times.size(); }

        /**
         * @brief Get number of reverse flows.
         */
        std::size_t reverse_count() const { return reverse_start_times.size(); }

        /**
         * @brief Get total number of flows (both directions).
         */
        std::size_t total_count() const { return forward_count() + reverse_count(); }

        /**
         * @brief Check if there are any flows.
         */
        bool has_flows() const { return total_count() > 0; }

        /**
         * @brief Check if flows are bidirectional.
         */
        bool is_bidirectional() const {
            return forward_count() > 0 && reverse_count() > 0;
        }
    };

    /**
     * @brief Collect ALL flow data between two vertices.
     *
     * Always populates all fields. Use this when you need complete data.
     *
     * @tparam GraphTraits The graph traits type.
     * @param graph The graph manager providing access to edges.
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @return Complete flow data structure.
     */
    template <typename GraphTraits>
    static FlowData collect_all(const IGraphManager<GraphTraits>& graph,
                                const typename GraphTraits::Vertex& src,
                                const typename GraphTraits::Vertex& dst);

    /**
     * @brief Get sorted combined timestamps from FlowData.
     *
     * @param data The flow data.
     * @return Sorted vector of all start timestamps.
     */
    static std::vector<std::chrono::milliseconds>
    get_sorted_start_times(const FlowData& data);
};

// ============================================================================

template <typename GraphTraits>
FlowDataCollector::FlowData
FlowDataCollector::collect_all(const IGraphManager<GraphTraits>& graph,
                               const typename GraphTraits::Vertex& src,
                               const typename GraphTraits::Vertex& dst) {
    FlowData data;

    // Forward flows (src -> dst)
    {
        auto [it, end] = graph.get_out_edges(src);
        for (; it != end; ++it) {
            if (graph.get_target_vertex(*it) != dst)
                continue;

            const auto& props = graph.get_edge_properties(*it);
            data.forward_start_times.push_back(props.start_timestamp);
            data.forward_end_times.push_back(props.end_timestamp);
            data.total_duration_ms += static_cast<double>(
                (props.end_timestamp - props.start_timestamp).count());
        }
    }

    // Reverse flows (dst -> src)
    {
        auto [it, end] = graph.get_out_edges(dst);
        for (; it != end; ++it) {
            if (graph.get_target_vertex(*it) != src)
                continue;

            const auto& props = graph.get_edge_properties(*it);
            data.reverse_start_times.push_back(props.start_timestamp);
            data.reverse_end_times.push_back(props.end_timestamp);
            data.total_duration_ms += static_cast<double>(
                (props.end_timestamp - props.start_timestamp).count());
        }
    }

    return data;
}

inline std::vector<std::chrono::milliseconds>
FlowDataCollector::get_sorted_start_times(const FlowData& data) {
    std::vector<std::chrono::milliseconds> times;
    times.reserve(data.total_count());
    times.insert(times.end(), data.forward_start_times.begin(),
                 data.forward_start_times.end());
    times.insert(times.end(), data.reverse_start_times.begin(),
                 data.reverse_start_times.end());
    std::sort(times.begin(), times.end());
    return times;
}
