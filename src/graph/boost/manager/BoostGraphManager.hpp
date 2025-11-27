#pragma once

#include "graph/IGraphManager.hpp"
#include "graph/boost/BoostGraphTraits.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

/**
 * @brief Graph manager implementation using Boost Graph Library.
 *
 * This class provides methods for managing a graph using the Boost Graph Library.
 * The behavior of the graph is defined by the template parameter.
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
    virtual const VertexProperties &get_vertex_properties(const Vertex &vertex) const override;

    /**
     * @brief Get the properties of a edge.
     *
     * @param edge The edge to get the properties for.
     * @return The properties of the edge.
     */
    virtual const EdgeProperties &get_edge_properties(const Edge &edge) const override;

    /**
     * @brief Get the degree of a vertex.
     *
     * @param vertex The vertex to get the degree for.
     * @return The degree of the vertex.
     */
     virtual std::size_t get_degree(const Vertex &vertex) const override;

     /**
      * @brief Get neighbors of a vertex.
      *
      * @param vertex The vertex to get neighbors for.
      * @return Vector of neighbor vertices.
      */
     virtual std::vector<Vertex> get_neighbors(const Vertex &vertex) const override;
 
     /**
      * @brief Check if two vertices are connected.
      *
      * @param u First vertex
      * @param v Second vertex
      * @return True if vertices are connected, false otherwise.
      */
     virtual bool are_connected(const Vertex &u, const Vertex &v) const override;
 
     /**
      * @brief Get common neighbors between two vertices.
      *
      * @param u First vertex
      * @param v Second vertex
      * @return Vector of common neighbor vertices
      */
     virtual std::vector<Vertex> get_common_neighbors(const Vertex &u, const Vertex &v) const override;
 
     /**
      * @brief Get count of common neighbors between two vertices.
      *
      * @param u First vertex
      * @param v Second vertex
      * @return Number of common neighbors
      */
     virtual std::size_t get_common_neighbors_count(const Vertex &u, const Vertex &v) const override;

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