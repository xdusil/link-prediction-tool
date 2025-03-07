#pragma once
#include "BoostGraphManager.hpp"
#include <cstddef>

template <typename Graph>
typename BoostGraphManager<Graph>::Vertex
BoostGraphManager<Graph>::add_vertex(const VertexProperties &properties) {
    return boost::add_vertex(properties, m_graph);
}

template <typename Graph>
bool BoostGraphManager<Graph>::add_edge(const Vertex &src, const Vertex &dst,
                                        const EdgeProperties &properties) {
    auto [edge, inserted] = boost::add_edge(src, dst, properties, m_graph);
    return inserted;
}

template <typename Graph>
std::size_t BoostGraphManager<Graph>::get_vertex_count() const {
    return boost::num_vertices(m_graph);
}

template <typename Graph>
std::size_t BoostGraphManager<Graph>::get_edge_count() const {
    return boost::num_edges(m_graph);
}

template <typename Graph>
std::pair<typename BoostGraphManager<Graph>::vertex_iterator,
          typename BoostGraphManager<Graph>::vertex_iterator>
BoostGraphManager<Graph>::get_vertices() const {
    return boost::vertices(m_graph);
}

template <typename Graph>
std::pair<typename BoostGraphManager<Graph>::edge_iterator,
          typename BoostGraphManager<Graph>::edge_iterator>
BoostGraphManager<Graph>::get_edges() const {
    return boost::edges(m_graph);
}

template <typename Graph>
std::pair<typename BoostGraphManager<Graph>::out_edge_iterator,
          typename BoostGraphManager<Graph>::out_edge_iterator>
BoostGraphManager<Graph>::get_out_edges(const Vertex &vertex) const {
    return boost::out_edges(vertex, m_graph);
}

template <typename Graph>
typename BoostGraphManager<Graph>::Vertex
BoostGraphManager<Graph>::get_source_vertex(const Edge &edge) const {
    return boost::source(edge, m_graph);
}

template <typename Graph>
typename BoostGraphManager<Graph>::Vertex
BoostGraphManager<Graph>::get_target_vertex(const Edge &edge) const {
    return boost::target(edge, m_graph);
}

template <typename Graph>
const typename BoostGraphManager<Graph>::VertexProperties &
BoostGraphManager<Graph>::get_vertex_properties(const Vertex &vertex) const {
    return m_graph[vertex];
}

template <typename Graph>
const typename BoostGraphManager<Graph>::EdgeProperties &
BoostGraphManager<Graph>::get_edge_properties(const Edge &edge) const {
    return m_graph[edge];
}

template <typename Graph>
std::size_t BoostGraphManager<Graph>::get_degree(const Vertex &vertex) const {
    return boost::out_degree(vertex, m_graph);
}

template <typename Graph>
std::vector<typename BoostGraphManager<Graph>::Vertex> 
BoostGraphManager<Graph>::get_neighbors(const Vertex &vertex) const {
    std::vector<Vertex> neighbors;
    auto range = boost::adjacent_vertices(vertex, m_graph);
    for (auto it = range.first; it != range.second; ++it) {
        neighbors.push_back(*it);
    }
    return neighbors;
}

template <typename Graph>
bool BoostGraphManager<Graph>::are_connected(const Vertex &u, const Vertex &v) const {
    return boost::edge(u, v, m_graph).second;
}

template <typename Graph>
std::vector<typename BoostGraphManager<Graph>::Vertex>
BoostGraphManager<Graph>::get_common_neighbors(const Vertex &u, const Vertex &v) const {
    std::vector<Vertex> common;
    auto u_neighbors = get_neighbors(u);
    auto v_neighbors = get_neighbors(v);
    
    std::sort(u_neighbors.begin(), u_neighbors.end());
    std::sort(v_neighbors.begin(), v_neighbors.end());
    
    std::set_intersection(
        u_neighbors.begin(), u_neighbors.end(),
        v_neighbors.begin(), v_neighbors.end(),
        std::back_inserter(common)
    );
    
    return common;
}

template <typename Graph>
std::size_t
BoostGraphManager<Graph>::get_common_neighbors_count(const Vertex &u, const Vertex &v) const {
    return get_common_neighbors(u, v).size();
}