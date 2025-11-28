#pragma once

#include "FeatureConfig.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

/**
 * @brief Extractor for temporal features from edge properties.
 *
 * Analyzes timing patterns between node pairs to detect dependency patterns.
 */
class TemporalFeatureExtractor {
public:
    static constexpr auto CONCENTRATION_TIME_WINDOW = std::chrono::milliseconds(1000);

    struct TemporalFeatures {
        std::optional<double> avg_duration;
        std::optional<double> avg_inter_arrival;
        std::optional<double> burstiness;
        std::optional<double> regularity;
        std::optional<double> temporal_concentration;
    };

    /**
     * @brief Batch extraction - collects data once and computes only enabled features.
     *
     * @complexity O(E_pair + E_pair*log(E_pair)) where E_pair is the number of edges
     * between v1 and v2
     * @param graph_manager The graph manager interface
     * @param v1 First vertex
     * @param v2 Second vertex
     * @param config Feature configuration specifying which features to compute
     * @return TemporalFeatures struct with only enabled features populated
     */
    template <typename GraphTraits>
    static TemporalFeatures
    extract_all_features(const IGraphManager<GraphTraits> &graph_manager,
                         const typename GraphTraits::Vertex &v1,
                         const typename GraphTraits::Vertex &v2,
                         const FeatureConfig &config);
};

#include "TemporalFeatureExtractor.tpp"
