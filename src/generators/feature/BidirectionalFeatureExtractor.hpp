#pragma once

#include "FeatureConfig.hpp"
#include "graph/IGraphManager.hpp"
#include <chrono>
#include <optional>
#include <vector>

/**
 * @brief Extractor for bidirectional flow features.
 *
 * Analyzes request-response patterns between node pairs.
 *
 * @note All bidirectional features are commutative: f(v1, v2) == f(v2, v1).
 */
class BidirectionalFeatureExtractor {
public:
    // Configuration constants
    static constexpr auto MAX_RESPONSE_TIME =
        std::chrono::milliseconds(10000); // 10 seconds

    static constexpr double COUNT_WEIGHT = 0.6;    // Weight for count asymmetry
    static constexpr double DURATION_WEIGHT = 0.4; // Weight for duration asymmetry
    static_assert(COUNT_WEIGHT + DURATION_WEIGHT > 0.999 &&
                      COUNT_WEIGHT + DURATION_WEIGHT < 1.001,
                  "Asymmetry weights must sum to 1.0");

    struct BidirectionalFeatures {
        std::optional<bool> has_bidirectional_flows;
        std::optional<double> avg_response_time;
        std::optional<double> request_response_ratio;
        std::optional<double> directional_asymmetry;
    };

    /**
     * @brief Batch extraction - collects edges once and computes only enabled features.
     *
     * @complexity O(degree(v1) + degree(v2) + E_pair*log(E_pair)) where E_pair is edges
     * between v1 and v2
     * @param graph_manager The graph manager interface
     * @param v1 First vertex
     * @param v2 Second vertex
     * @param config Feature configuration specifying which features to compute
     * @return BidirectionalFeatures containing entries for only the enabled features;
     *         disabled features are left unset, and enabled features may be
     *         `std::nullopt` if computation is not possible.
     */
    template <typename GraphTraits>
    static BidirectionalFeatures
    extract_all_features(const IGraphManager<GraphTraits> &graph_manager,
                         const typename GraphTraits::Vertex &v1,
                         const typename GraphTraits::Vertex &v2,
                         const FeatureConfig &config);

private:
    /**
     * @brief Collect all edges in a specific direction sorted by timestamp.
     *
     * @param graph_manager The graph manager interface
     * @param src Source vertex
     * @param dst Destination vertex
     * @return Vector of edge properties sorted by start timestamp
     */
    template <typename GraphTraits>
    static std::vector<typename GraphTraits::EdgeProperties>
    collect_edges_directional(const IGraphManager<GraphTraits> &graph_manager,
                              const typename GraphTraits::Vertex &src,
                              const typename GraphTraits::Vertex &dst);

    /**
     * @brief Calculate average response time for request-response pairs.
     *
     * Matches forward flows with subsequent reverse flows and calculates
     * the time delay.
     *
     * @param forward_edges Edges in forward direction (sorted by timestamp)
     * @param reverse_edges Edges in reverse direction (sorted by timestamp)
     * @return Optional mean response time in milliseconds (nullopt if no valid matches)
     */
    template <typename EdgeProperties>
    static std::optional<double>
    calculate_avg_response_time(const std::vector<EdgeProperties> &forward_edges,
                                const std::vector<EdgeProperties> &reverse_edges);

    /**
     * @brief Calculate directional asymmetry in edge properties.
     *
     * Measures difference in flow characteristics between directions.
     *
     * @param forward_edges Edges in forward direction
     * @param reverse_edges Edges in reverse direction
     * @return Normalized asymmetry score (0 = symmetric, 1 = highly asymmetric)
     */
    template <typename EdgeProperties>
    static double
    calculate_directional_asymmetry(const std::vector<EdgeProperties> &forward_edges,
                                    const std::vector<EdgeProperties> &reverse_edges);

    /**
     * @brief Find the closest reverse flow after a forward flow.
     *
     * Searches for the first reverse edge that starts after the forward edge.
     *
     * @param forward_edge The forward flow edge
     * @param reverse_edges All reverse flows (sorted by timestamp)
     * @param start_index Index to start searching from; updated to exclude processed
     * edges
     * @return Optional response time in milliseconds if matching reverse flow found
     */
    template <typename EdgeProperties>
    static std::optional<double>
    find_response_time(const EdgeProperties &forward_edge,
                       const std::vector<EdgeProperties> &reverse_edges,
                       std::size_t &start_index);
};

#include "BidirectionalFeatureExtractor.tpp"
