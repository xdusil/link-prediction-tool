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
     * @brief Aggregated flow data for a vertex pair.
     *
     * Contains timing information and counts for all flows between two vertices.
     * Forward = src -> dst, Reverse = dst -> src.
     */
    struct AggregatedFlowData {
        // Forward direction (src -> dst)
        std::vector<std::chrono::milliseconds> forward_start_times;
        std::vector<std::chrono::milliseconds> forward_end_times;

        // Reverse direction (dst -> src)
        std::vector<std::chrono::milliseconds> reverse_start_times;
        std::vector<std::chrono::milliseconds> reverse_end_times;

        // Aggregated metrics
        double total_duration_ms = 0.0; // Sum of all flow durations

        /**
         * @brief Get number of forward flows.
         */
        std::size_t forward_flow_count() const { return forward_start_times.size(); }

        /**
         * @brief Get number of reverse flows.
         */
        std::size_t reverse_flow_count() const { return reverse_start_times.size(); }

        /**
         * @brief Get total number of flows in both directions.
         */
        std::size_t total_flow_count() const {
            return forward_flow_count() + reverse_flow_count();
        }

        /**
         * @brief Check if there are any flows.
         */
        bool has_flows() const { return total_flow_count() > 0; }

        /**
         * @brief Check if there are flows in both directions.
         */
        bool is_bidirectional() const {
            return forward_flow_count() > 0 && reverse_flow_count() > 0;
        }
    };

    /**
     * @brief Collect all flow data between two vertices.
     *
     * Iterates through all edges (flows) from src to dst and from dst to src,
     * collecting timing information and aggregating statistics.
     *
     * @param graph_manager The graph manager providing access to edges.
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @return Aggregated flow data structure with all collected information.
     */
    template <typename GraphTraits>
    static AggregatedFlowData collect(const IGraphManager<GraphTraits>& graph_manager,
                                      const typename GraphTraits::Vertex& src,
                                      const typename GraphTraits::Vertex& dst);

    /**
     * @brief Count flows from one vertex to another.
     *
     * Simple utility to count edges without collecting timing data.
     *
     * @param graph_manager The graph manager.
     * @param from The source vertex.
     * @param to The destination vertex.
     * @return Number of flows (edges) from 'from' to 'to'.
     */
    template <typename GraphTraits>
    static std::size_t count_flows(const IGraphManager<GraphTraits>& graph_manager,
                                   const typename GraphTraits::Vertex& from,
                                   const typename GraphTraits::Vertex& to);

    /**
     * @brief Get combined and sorted timestamps from all flows.
     *
     * Merges forward and reverse flow start times into a single sorted vector.
     * Useful for temporal pattern analysis across both directions.
     *
     * @param data The aggregated flow data.
     * @return Sorted vector of all start timestamps.
     */
    static std::vector<std::chrono::milliseconds>
    get_all_start_times_sorted(const AggregatedFlowData& data);
};

// ============================================================================

template <typename GraphTraits>
FlowDataCollector::AggregatedFlowData
FlowDataCollector::collect(const IGraphManager<GraphTraits>& graph_manager,
                           const typename GraphTraits::Vertex& src,
                           const typename GraphTraits::Vertex& dst) {

    AggregatedFlowData data;

    // Collect forward flows (src -> dst)
    {
        auto [edge_it, edge_end] = graph_manager.get_out_edges(src);
        for (; edge_it != edge_end; ++edge_it) {
            // Skip edges not pointing to dst
            if (graph_manager.get_target_vertex(*edge_it) != dst) {
                continue;
            }

            const auto& flow_properties = graph_manager.get_edge_properties(*edge_it);

            // Store timing information
            data.forward_start_times.push_back(flow_properties.start_timestamp);
            data.forward_end_times.push_back(flow_properties.end_timestamp);

            // Accumulate duration
            auto duration_ms =
                (flow_properties.end_timestamp - flow_properties.start_timestamp).count();
            data.total_duration_ms += static_cast<double>(duration_ms);
        }
    }

    // Collect reverse flows (dst -> src)
    {
        auto [edge_it, edge_end] = graph_manager.get_out_edges(dst);
        for (; edge_it != edge_end; ++edge_it) {
            // Skip edges not pointing back to src
            if (graph_manager.get_target_vertex(*edge_it) != src) {
                continue;
            }

            const auto& flow_properties = graph_manager.get_edge_properties(*edge_it);

            // Store timing information
            data.reverse_start_times.push_back(flow_properties.start_timestamp);
            data.reverse_end_times.push_back(flow_properties.end_timestamp);

            // Accumulate duration
            auto duration_ms =
                (flow_properties.end_timestamp - flow_properties.start_timestamp).count();
            data.total_duration_ms += static_cast<double>(duration_ms);
        }
    }

    return data;
}

template <typename GraphTraits>
std::size_t
FlowDataCollector::count_flows(const IGraphManager<GraphTraits>& graph_manager,
                               const typename GraphTraits::Vertex& from,
                               const typename GraphTraits::Vertex& to) {

    std::size_t flow_count = 0;

    auto [edge_it, edge_end] = graph_manager.get_out_edges(from);
    for (; edge_it != edge_end; ++edge_it) {
        if (graph_manager.get_target_vertex(*edge_it) == to) {
            ++flow_count;
        }
    }

    return flow_count;
}

inline std::vector<std::chrono::milliseconds>
FlowDataCollector::get_all_start_times_sorted(const AggregatedFlowData& data) {

    std::vector<std::chrono::milliseconds> all_times;
    all_times.reserve(data.total_flow_count());

    // Merge forward and reverse start times
    all_times.insert(all_times.end(), data.forward_start_times.begin(),
                     data.forward_start_times.end());
    all_times.insert(all_times.end(), data.reverse_start_times.begin(),
                     data.reverse_start_times.end());

    // Sort chronologically
    std::sort(all_times.begin(), all_times.end());

    return all_times;
}
