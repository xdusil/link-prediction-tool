#pragma once

#include <cstddef>

/**
 * @brief Generic interface for managing graphs.
 *
 * This interface defines methods for adding vertices and edges to a graph,
 * as well as getting the count of vertices and edges in the graph.
 *
 * @tparam Vertex The vertex descriptor type used in the graph.
 * @tparam VertexProperties The type used to represent properties of vertices.
 * @tparam EdgeProperties The type used to represent properties of edges.
 */
template <typename Vertex, typename VertexProperties, typename EdgeProperties>
class IGraphManager {
public:
    virtual ~IGraphManager() = default;

    /**
     * @brief Add a vertex with associated properties.
     *
     * @param properties Properties of the vertex to add.
     * @return The added vertex descriptor.
     */
    virtual Vertex add_vertex(const VertexProperties& properties) = 0;

    /**
     * @brief Add an edge between two vertices with associated properties.
     *
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    virtual bool add_edge(const Vertex& src, const Vertex& dst, const EdgeProperties& properties) = 0;

    /**
     * @brief Get count of vertices in the graph.
     *
     * @return The number of vertices in the graph.
     */
    virtual std::size_t get_vertex_count() const = 0;

    /**
     * @brief Get count of edges in the graph.
     *
     * @return The number of edges in the graph.
     */
    virtual std::size_t get_edge_count() const = 0;
};
