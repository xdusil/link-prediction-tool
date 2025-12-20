#pragma once

#include "BoostGraphAnalytics.hpp"
#include <algorithm>
#include <boost/property_map/property_map.hpp>
#include <cmath>
#include <unordered_map>
#include <limits>

template <typename Graph>
double BoostGraphAnalytics<Graph>::adamic_adar_index(const Vertex &u, const Vertex &v) const {
    double score = 0.0;
    auto common_neighbors = get_common_neighbors(u, v);
    
    for (const auto &w : common_neighbors) {
        std::size_t out_deg_w = m_graph_manager.get_out_degree(w);
        if (out_deg_w > 1) { // Avoid log(1) = 0 and division by zero
            score += 1.0 / std::log(static_cast<double>(out_deg_w));
        }
    }
    
    return score;
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::jaccard_coefficient(const Vertex &u, const Vertex &v) const {
    std::size_t common = common_neighbors_count(u, v);
    std::size_t out_deg_u = m_graph_manager.get_out_degree(u);
    std::size_t in_deg_v = m_graph_manager.get_in_degree(v);
    
    // |A ∪ B| = |A| + |B| - |A ∩ B|
    std::size_t union_size = out_deg_u + in_deg_v - common;
    
    return union_size > 0 ? static_cast<double>(common) / union_size : 0.0;
}

template <typename Graph>
std::size_t BoostGraphAnalytics<Graph>::common_neighbors_count(const Vertex &u, const Vertex &v) const {
    return get_common_neighbors(u, v).size();
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::normalized_common_neighbors_count(const Vertex &u, const Vertex &v) const {
    std::size_t cn = common_neighbors_count(u, v);
    std::size_t out_deg_u = m_graph_manager.get_out_degree(u);
    std::size_t in_deg_v = m_graph_manager.get_in_degree(v);
    
    // Maximum possible directional common neighbors
    std::size_t max_possible = std::min(out_deg_u, in_deg_v);
    
    return max_possible > 0 ? static_cast<double>(cn) / max_possible : 0.0;
}

template <typename Graph>
std::vector<typename BoostGraphAnalytics<Graph>::Vertex>
BoostGraphAnalytics<Graph>::get_common_neighbors(const Vertex &u, const Vertex &v) const {
    std::vector<Vertex> common;
    auto u_out_neighbors = m_graph_manager.get_out_neighbors(u);
    auto v_in_neighbors = m_graph_manager.get_in_neighbors(v);
    
    std::sort(u_out_neighbors.begin(), u_out_neighbors.end());
    std::sort(v_in_neighbors.begin(), v_in_neighbors.end());
    
    std::set_intersection(
        u_out_neighbors.begin(), u_out_neighbors.end(),
        v_in_neighbors.begin(), v_in_neighbors.end(),
        std::back_inserter(common)
    );
    
    return common;
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::preferential_attachment(const Vertex &u, const Vertex &v) const {
    std::size_t out_deg_u = m_graph_manager.get_out_degree(u);
    std::size_t in_deg_v = m_graph_manager.get_in_degree(v);
    
    return static_cast<double>(out_deg_u * in_deg_v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::resource_allocation_index(const Vertex &u, const Vertex &v) const {
    double score = 0.0;
    auto common_neighbors = get_common_neighbors(u, v);
    
    for (const auto &w : common_neighbors) {
        std::size_t out_deg_w = m_graph_manager.get_out_degree(w);
        if (out_deg_w > 0) { // Avoid division by zero
            score += 1.0 / out_deg_w;
        }
    }
    
    return score;
}

template <typename Graph>
std::size_t BoostGraphAnalytics<Graph>::in_degree(const Vertex &v) const {
    return m_graph_manager.get_in_degree(v);
}

template <typename Graph>
std::size_t BoostGraphAnalytics<Graph>::out_degree(const Vertex &v) const {
    return m_graph_manager.get_out_degree(v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::clustering_coefficient(const Vertex &v) const {
    auto neighbors = m_graph_manager.get_out_neighbors(v);
    std::size_t k = neighbors.size();
    
    // If out-degree is less than 2, clustering coefficient is 0
    if (k < 2) return 0.0;
    
    // Count connections between out-neighbors (in either direction)
    std::size_t connections = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
            // Check both directions
            if (m_graph_manager.has_edge(neighbors[i], neighbors[j]) ||
                m_graph_manager.has_edge(neighbors[j], neighbors[i])) {
                connections++;
            }
        }
    }
    
    // Calculate clustering coefficient: 2 * connections / (k * (k-1))
    return static_cast<double>(2 * connections) / (k * (k - 1));
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::clustering_coefficient_difference(
    const Vertex &u, const Vertex &v) const {
    double cc_u = clustering_coefficient(u);
    double cc_v = clustering_coefficient(v);
    
    return std::abs(cc_u - cc_v);
}

template <typename Graph>
std::vector<typename BoostGraphAnalytics<Graph>::Vertex>
BoostGraphAnalytics<Graph>::get_out_neighbors(const Vertex &v) const {
    return m_graph_manager.get_out_neighbors(v);
}

template <typename Graph>
std::vector<typename BoostGraphAnalytics<Graph>::Vertex>
BoostGraphAnalytics<Graph>::get_in_neighbors(const Vertex &v) const {
    return m_graph_manager.get_in_neighbors(v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::avg_out_degree() const {
    double sum = 0.0;
    std::size_t num_vertices = 0;
    auto vertices = m_graph_manager.get_vertices();
    for (auto it = vertices.first; it != vertices.second; ++it) {
        sum += m_graph_manager.get_out_degree(*it);
        num_vertices++;
    }
    return num_vertices > 0 ? sum / num_vertices : 0.0;
}

template <typename Graph>
const IGraphManager<BoostGraphTraits<Graph>>& BoostGraphAnalytics<Graph>::get_graph_manager() const {
    return m_graph_manager;
}