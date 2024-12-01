#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include "../IGraphManager.hpp"

/**
 * @brief Graph manager implementation using Boost Graph Library.
 *
 * This class provides methods for managing a graph using the Boost Graph Library.
 * The behavior of the graph is defined by the template parameters.
 *
 * @tparam VertexProperties The type used to represent properties of vertices.
 * @tparam EdgeProperties The type used to represent properties of edges.
 * @tparam Graph The type of the graph to manage.
 */
template <
    typename VertexProperties,
    typename EdgeProperties,
    typename Graph
>
class BoostGraphManager : public IGraphManager<
    typename boost::graph_traits<Graph>::vertex_descriptor,
    VertexProperties,
    EdgeProperties
> {
public:

    // Define the graph element types
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = typename boost::graph_traits<Graph>::edge_descriptor;

    virtual ~BoostGraphManager() = default;

    /**
     * @brief Add a vertex with associated properties.
     *
     * @param properties Properties of the vertex to add.
     * @return The added vertex descriptor.
     */
    virtual Vertex add_vertex(const VertexProperties& properties) override;

    /**
     * @brief Add an edge between two vertices with associated properties.
     *
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    virtual bool add_edge(const Vertex& src, const Vertex& dst,
                          const EdgeProperties& properties) override;

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

//private:

    Graph m_graph;      // The graph managed by this class
};

#include "BoostGraphManager.tpp"