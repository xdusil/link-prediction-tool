#pragma once

#include <cstddef>
#include <utility>
#include <vector>

/**
 * @brief Generic interface for managing directed graphs.
 *
 * This interface provides methods for adding vertices and edges to a directed graph,
 * as well as accessing the vertices and edges in the graph.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 */
template <typename GraphTraits>
class IGraphManager {
public:
    // Define the graph element types
    using GraphType = typename GraphTraits::GraphType;
    using Vertex = typename GraphTraits::Vertex;
    using Edge = typename GraphTraits::Edge;
    using VertexProperties = typename GraphTraits::VertexProperties;
    using EdgeProperties = typename GraphTraits::EdgeProperties;
    using vertex_iterator = typename GraphTraits::vertex_iterator;
    using edge_iterator = typename GraphTraits::edge_iterator;
    using out_edge_iterator = typename GraphTraits::out_edge_iterator;

    virtual ~IGraphManager() = default;

    /**
     * @brief Add a vertex with associated properties.
     *
     * @param properties Properties of the vertex to add.
     * @return The added vertex descriptor.
     */
    virtual Vertex add_vertex(const VertexProperties &properties) = 0;

    /**
     * @brief Add a directed edge from src to dst with associated properties.
     *
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    virtual bool add_edge(const Vertex &src, const Vertex &dst,
                          const EdgeProperties &properties) = 0;

    /**
     * @brief Add a directed edge between two vertices with associated properties.
     * If the vertices do not exist, they are added to the graph.
     *
     * @param src_properties Properties of the source vertex.
     * @param dst_properties Properties of the destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    virtual bool add_edge_and_vertex_if_not_exists(const VertexProperties &src_properties,
                                                   const VertexProperties &dst_properties,
                                                   const EdgeProperties &properties) = 0;

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

    /**
     * @brief Get the vertices in the graph.
     *
     * @return A pair of iterators representing the range of vertices in the graph.
     */
    virtual std::pair<vertex_iterator, vertex_iterator> get_vertices() const = 0;

    /**
     * @brief Get the edges in the graph.
     *
     * @return A pair of iterators representing the range of edges in the graph.
     */
    virtual std::pair<edge_iterator, edge_iterator> get_edges() const = 0;

    /**
     * @brief Get the out-edges of a vertex in the graph.
     *
     * @param vertex The vertex to get the out-edges for.
     * @return A pair of iterators representing the range of out-edges for the vertex.
     */
    virtual std::pair<out_edge_iterator, out_edge_iterator>
    get_out_edges(const Vertex &vertex) const = 0;

    /**
     * @brief Get the source vertex of an edge in the graph.
     *
     * @param edge The edge to get the source vertex for.
     * @return The source vertex of the edge.
     */
    virtual Vertex get_source_vertex(const Edge &edge) const = 0;

    /**
     * @brief Get the target vertex of an edge in the graph.
     *
     * @param edge The edge to get the target vertex for.
     * @return The target vertex of the edge.
     */
    virtual Vertex get_target_vertex(const Edge &edge) const = 0;

    /**
     * @brief Get the properties of a vertex in the graph.
     *
     * @param vertex The vertex to get the properties for.
     * @return The properties of the vertex.
     */
    virtual const VertexProperties &get_vertex_properties(const Vertex &vertex) const = 0;

    /**
     * @brief Get the properties of an edge in the graph.
     *
     * @param edge The edge to get the properties for.
     * @return The properties of the edge.
     */
    virtual const EdgeProperties &get_edge_properties(const Edge &edge) const = 0;

    /**
     * @brief Get the in-degree of a vertex (number of incoming edges).
     *
     * @param vertex The vertex to get the in-degree for.
     * @return The in-degree of the vertex.
     */
    virtual std::size_t get_in_degree(const Vertex &vertex) const = 0;

    /**
     * @brief Get the out-degree of a vertex (number of outgoing edges).
     *
     * @param vertex The vertex to get the out-degree for.
     * @return The out-degree of the vertex.
     */
    virtual std::size_t get_out_degree(const Vertex &vertex) const = 0;

    /**
     * @brief Get out-neighbors of a vertex (vertices this vertex points to).
     *
     * Returns all vertices v such that there exists an edge (vertex -> v).
     *
     * @param vertex The vertex to get out-neighbors for.
     * @return Vector of out-neighbor vertices.
     */
    virtual std::vector<Vertex> get_out_neighbors(const Vertex &vertex) const = 0;

    /**
     * @brief Get in-neighbors of a vertex (vertices that point to this vertex).
     *
     * Returns all vertices u such that there exists an edge (u -> vertex).
     *
     * @param vertex The vertex to get in-neighbors for.
     * @return Vector of in-neighbor vertices.
     */
    virtual std::vector<Vertex> get_in_neighbors(const Vertex &vertex) const = 0;

    /**
     * @brief Check if a directed edge exists from src to dst.
     *
     * @param src Source vertex
     * @param dst Destination vertex
     * @return True if edge (src -> dst) exists, false otherwise.
     */
    virtual bool has_edge(const Vertex &src, const Vertex &dst) const = 0;

    /**
     * @brief Get common out-neighbors between two vertices.
     *
     * Returns vertices that both u and v point to:
     * { w : (u -> w) AND (v -> w) }
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Vector of common out-neighbor vertices
     */
    virtual std::vector<Vertex> get_common_out_neighbors(const Vertex &u,
                                                         const Vertex &v) const = 0;

    /**
     * @brief Get common in-neighbors between two vertices.
     *
     * Returns vertices that point to both u and v:
     * { w : (w -> u) AND (w -> v) }
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Vector of common in-neighbor vertices
     */
    virtual std::vector<Vertex> get_common_in_neighbors(const Vertex &u,
                                                        const Vertex &v) const = 0;

    /**
     * @brief Get the underlying graph object.
     *
     * @return The underlying graph.
     */
    virtual const GraphType &get_graph() const = 0;
};
