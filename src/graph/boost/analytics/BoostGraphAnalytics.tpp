#pragma once

#include "BoostGraphAnalytics.hpp"
#include <algorithm>
#include <boost/property_map/property_map.hpp>
#include <cmath>
#include <unordered_map>
#include <limits>

template <typename Graph>
double BoostGraphAnalytics<Graph>::adamic_adar(const Vertex &u, const Vertex &v) const {
    double score = 0.0;
    auto common_neighbors = m_graph_manager.get_common_neighbors(u, v);
    
    for (const auto &w : common_neighbors) {
        std::size_t deg_w = m_graph_manager.get_degree(w);
        if (deg_w > 1) { // Avoid log(1) = 0 and division by zero
            score += 1.0 / std::log(static_cast<double>(deg_w));
        }
    }
    
    return score;
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::jaccard_coefficient(const Vertex &u, const Vertex &v) const {
    std::size_t common = m_graph_manager.get_common_neighbors_count(u, v);
    std::size_t deg_u = m_graph_manager.get_degree(u);
    std::size_t deg_v = m_graph_manager.get_degree(v);
    
    // Total number of unique neighbors (union of neighbor sets)
    std::size_t total = deg_u + deg_v - common;
    
    return total > 0 ? static_cast<double>(common) / total : 0.0;
}

template <typename Graph>
std::size_t BoostGraphAnalytics<Graph>::common_neighbors_count(const Vertex &u, const Vertex &v) const {
    return m_graph_manager.get_common_neighbors_count(u, v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::normalized_common_neighbors_count(const Vertex &u, const Vertex &v) const {
    std::size_t cn = common_neighbors_count(u, v);
    std::size_t deg_u = m_graph_manager.get_degree(u);
    std::size_t deg_v = m_graph_manager.get_degree(v);
    
    // Maximum possible common neighbors is min(deg_u, deg_v)
    std::size_t max_possible = std::min(deg_u, deg_v);
    
    return max_possible > 0 ? static_cast<double>(cn) / max_possible : 0.0;
}

template <typename Graph>
std::vector<typename BoostGraphAnalytics<Graph>::Vertex>
BoostGraphAnalytics<Graph>::get_common_neighbors(const Vertex &u, const Vertex &v) const {
    return m_graph_manager.get_common_neighbors(u, v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::preferential_attachment(const Vertex &u, const Vertex &v) const {
    std::size_t deg_u = m_graph_manager.get_degree(u);
    std::size_t deg_v = m_graph_manager.get_degree(v);
    
    return static_cast<double>(deg_u * deg_v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::resource_allocation(const Vertex &u, const Vertex &v) const {
    double score = 0.0;
    auto common_neighbors = m_graph_manager.get_common_neighbors(u, v);
    
    for (const auto &w : common_neighbors) {
        std::size_t deg_w = m_graph_manager.get_degree(w);
        if (deg_w > 0) { // Avoid division by zero
            score += 1.0 / deg_w;
        }
    }
    
    return score;
}

template <typename Graph>
std::size_t BoostGraphAnalytics<Graph>::degree(const Vertex &v) const {
    return m_graph_manager.get_degree(v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::clustering_coefficient(const Vertex &v) const {
    std::size_t k = m_graph_manager.get_degree(v);
    
    // If degree is less than 2, clustering coefficient is 0
    if (k < 2) return 0.0;
    
    // Get neighbors
    auto neighbors = m_graph_manager.get_neighbors(v);
    
    // Count connections between neighbors
    std::size_t connections = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (size_t j = i + 1; j < neighbors.size(); ++j) {
            if (m_graph_manager.are_connected(neighbors[i], neighbors[j])) {
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
BoostGraphAnalytics<Graph>::get_neighbors(const Vertex &v) const {
    return m_graph_manager.get_neighbors(v);
}

template <typename Graph>
double BoostGraphAnalytics<Graph>::avg_degree() const {
    double avg = 0.0;
    std::size_t num_vertices = 0;
    auto vertices = m_graph_manager.get_vertices();
    for (auto it = vertices.first; it != vertices.second; ++it) {
        avg += m_graph_manager.get_degree(*it);
        num_vertices++;
    }
    return avg / num_vertices;
}