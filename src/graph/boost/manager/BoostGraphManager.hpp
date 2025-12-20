#pragma once

#include "graph/IGraphManager.hpp"
#include "graph/boost/BoostGraphTraits.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

/**
 * @brief Graph manager implementation using Boost Graph Library.
 *
 * This class provides methods for managing a directed graph using the Boost Graph
 * Library.
 *
 * @tparam Graph The type of the graph to manage.
 */
template <typename Graph>
class BoostGraphManager : public virtual IGraphManager<BoostGraphTraits<Graph>> {
public:
    // Define the base class
    using Base = BoostGraphTraits<Graph>;

    // Define the graph element types
    using Vertex = Base::Vertex;
    using Edge = Base::Edge;
    using VertexProperties = Base::VertexProperties;
    using EdgeProperties = Base::EdgeProperties;
    using vertex_iterator = Base::vertex_iterator;
    using edge_iterator = Base::edge_iterator;
    using out_edge_iterator = Base::out_edge_iterator;

    virtual ~BoostGraphManager() = default;

    /**
     * @brief Add a vertex with associated properties.
     *
     * @param properties Properties of the vertex to add.
     * @return The added vertex descriptor.
     */
    virtual Vertex add_vertex(const VertexProperties &properties) override;

    /**
     * @brief Add an edge between two vertices with associated properties.
     *
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    virtual bool add_edge(const Vertex &src, const Vertex &dst,
                          const EdgeProperties &properties) override;

    /**
     * @brief Get count of vertices in the graph.
     *
     * @return The number of vertices in the graph.
     */
    virtual std::size_t get_vertex_count() const override;

    /**
     * @brief Get count of edges in the graph.
     *
     * @return The number of edges in the graph.
     */
    virtual std::size_t get_edge_count() const override;

    /**
     * @brief Get the vertices in the graph.
     *
     * @return A pair of iterators representing the range of vertices in the graph.
     */
    virtual std::pair<vertex_iterator, vertex_iterator> get_vertices() const override;

    /**
     * @brief Get the edges in the graph.
     *
     * @return A pair of iterators representing the range of edges in the graph.
     */
    virtual std::pair<edge_iterator, edge_iterator> get_edges() const override;

    /**
     * @brief Get the out-edges of a vertex.
     *
     * @param vertex The vertex to get the out-edges for.
     * @return A pair of iterators representing the range of out-edges for the vertex.
     */
    virtual std::pair<out_edge_iterator, out_edge_iterator>
    get_out_edges(const Vertex &vertex) const override;

    /**
     * @brief Get the source vertex of an edge.
     *
     * @param edge The edge to get the source vertex for.
     * @return The source vertex of the edge.
     */
    virtual Vertex get_source_vertex(const Edge &edge) const override;

    /**
     * @brief Get the target vertex of an edge.
     *
     * @param edge The edge to get the target vertex for.
     * @return The target vertex of the edge.
     */
    virtual Vertex get_target_vertex(const Edge &edge) const override;

    /**
     * @brief Get the properties of a vertex.
     *
     * @param vertex The vertex to get the properties for.
     * @return The properties of the vertex.
     */
    virtual const VertexProperties &
    get_vertex_properties(const Vertex &vertex) const override;

    /**
     * @brief Get the properties of a edge.
     *
     * @param edge The edge to get the properties for.
     * @return The properties of the edge.
     */
    virtual const EdgeProperties &get_edge_properties(const Edge &edge) const override;

    /**
     * @brief Get the in-degree of a vertex (number of incoming edges).
     *
     * @param vertex The vertex to get the in-degree for.
     * @return The in-degree of the vertex.
     */
    virtual std::size_t get_in_degree(const Vertex &vertex) const override;

    /**
     * @brief Get the out-degree of a vertex (number of outgoing edges).
     *
     * @param vertex The vertex to get the out-degree for.
     * @return The out-degree of the vertex.
     */
    virtual std::size_t get_out_degree(const Vertex &vertex) const override;

    /**
     * @brief Get out-neighbors of a vertex (vertices this vertex points to).
     *
     * Returns all vertices v such that there exists an edge (vertex -> v).
     *
     * @param vertex The vertex to get out-neighbors for.
     * @return Vector of out-neighbor vertices.
     */
    virtual std::vector<Vertex> get_out_neighbors(const Vertex &vertex) const override;

    /**
     * @brief Get in-neighbors of a vertex (vertices that point to this vertex).
     *
     * Returns all vertices u such that there exists an edge (u -> vertex).
     *
     * @param vertex The vertex to get in-neighbors for.
     * @return Vector of in-neighbor vertices.
     */
    virtual std::vector<Vertex> get_in_neighbors(const Vertex &vertex) const override;

    /**
     * @brief Check if a directed edge exists from src to dst.
     *
     * @param src Source vertex
     * @param dst Destination vertex
     * @return True if edge (src -> dst) exists, false otherwise.
     */
    virtual bool has_edge(const Vertex &src, const Vertex &dst) const override;

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
                                                         const Vertex &v) const override;

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
                                                        const Vertex &v) const override;

    /**
     * @brief Get the underlying graph object.
     *
     * @return The underlying graph.
     */
    virtual const Graph &get_graph() const override;

protected:
    Graph m_graph; // The graph managed by this class
};

#include "BoostGraphManager.tpp"