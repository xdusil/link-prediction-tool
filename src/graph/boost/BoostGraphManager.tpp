#pragma once
#include "BoostGraphManager.hpp"

// Add a vertex with associated properties.
template <typename Graph>
typename BoostGraphManager<Graph>::Vertex
BoostGraphManager<Graph>::add_vertex(const VertexProperties &properties) {
    return boost::add_vertex(properties, m_graph);
}

// Add an edge between two vertices with associated properties.
template <typename Graph>
bool BoostGraphManager<Graph>::add_edge(const Vertex &src, const Vertex &dst,
                                        const EdgeProperties &properties) {
    auto [edge, inserted] = boost::add_edge(src, dst, properties, m_graph);
    return inserted;
}

// Get count of vertices in the graph.
template <typename Graph>
std::size_t BoostGraphManager<Graph>::get_vertex_count() const {
    return boost::num_vertices(m_graph);
}

// Get count of edges in the graph.
template <typename Graph>
std::size_t BoostGraphManager<Graph>::get_edge_count() const {
    return boost::num_edges(m_graph);
}

// Get the vertices in the graph.
template <typename Graph>
std::pair<typename BoostGraphManager<Graph>::vertex_iterator,
          typename BoostGraphManager<Graph>::vertex_iterator>
BoostGraphManager<Graph>::get_vertices() const {
    return boost::vertices(m_graph);
}

// Get the edges in the graph.
template <typename Graph>
std::pair<typename BoostGraphManager<Graph>::edge_iterator,
          typename BoostGraphManager<Graph>::edge_iterator>
BoostGraphManager<Graph>::get_edges() const {
    return boost::edges(m_graph);
}

// Get the out-edges of a vertex.
template <typename Graph>
std::pair<typename BoostGraphManager<Graph>::out_edge_iterator,
          typename BoostGraphManager<Graph>::out_edge_iterator>
BoostGraphManager<Graph>::get_out_edges(const Vertex &vertex) const {
    return boost::out_edges(vertex, m_graph);
}

// Get the source vertex of an edge.
template <typename Graph>
typename BoostGraphManager<Graph>::Vertex
BoostGraphManager<Graph>::get_source_vertex(const Edge &edge) const {
    return boost::source(edge, m_graph);
}

// Get the target vertex of an edge.
template <typename Graph>
typename BoostGraphManager<Graph>::Vertex
BoostGraphManager<Graph>::get_target_vertex(const Edge &edge) const {
    return boost::target(edge, m_graph);
}

// Get the vertex properties
template <typename Graph>
const typename BoostGraphManager<Graph>::VertexProperties &
BoostGraphManager<Graph>::get_vertex_properties(const Vertex &vertex) const {
    return m_graph[vertex];
}

// Get the properties of an edge
template <typename Graph>
const typename BoostGraphManager<Graph>::EdgeProperties &
BoostGraphManager<Graph>::get_edge_properties(const Edge &edge) const {
    return m_graph[edge];
}