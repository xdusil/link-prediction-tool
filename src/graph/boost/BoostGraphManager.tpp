#pragma once
#include "BoostGraphManager.hpp"

// Add a vertex with associated properties.
template <
    typename VertexProperties,
    typename EdgeProperties,
    typename Graph
>
typename BoostGraphManager<VertexProperties, EdgeProperties, Graph>::Vertex
BoostGraphManager<VertexProperties, EdgeProperties, Graph>::add_vertex(
    const VertexProperties& properties) {
    return boost::add_vertex(properties, m_graph);
}

// Add an edge between two vertices with associated properties.
template <
    typename VertexProperties,
    typename EdgeProperties,
    typename Graph
>
bool BoostGraphManager<VertexProperties, EdgeProperties, Graph>::add_edge(
    const Vertex& src, const Vertex& dst, const EdgeProperties& properties) {
    auto [edge, inserted] = boost::add_edge(src, dst, properties, m_graph);
    return inserted;
}

// Get count of vertices in the graph.
template <
    typename VertexProperties,
    typename EdgeProperties,
    typename Graph
>
std::size_t BoostGraphManager<VertexProperties, EdgeProperties, Graph>::get_vertex_count() const {
    return boost::num_vertices(m_graph);
}

// Get count of edges in the graph.
template <
    typename VertexProperties,
    typename EdgeProperties,
    typename Graph
>
std::size_t BoostGraphManager<VertexProperties, EdgeProperties, Graph>::get_edge_count() const {
    return boost::num_edges(m_graph);
}