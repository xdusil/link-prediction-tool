#pragma once

#include "graph/IGraphAnalytics.hpp"
#include "graph/IGraphManager.hpp"
#include "graph/boost/BoostGraphTraits.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/clustering_coefficient.hpp>

/**
 * @brief Implementation of graph analytics using Boost Graph Library.
 *
 * @tparam Graph The type of the graph to analyze.
 */
template <typename Graph>
class BoostGraphAnalytics : public IGraphAnalytics<BoostGraphTraits<Graph>> {
public:
    // Define the types
    using GraphTraits = BoostGraphTraits<Graph>;
    using Base = IGraphAnalytics<GraphTraits>;
    using Vertex = typename GraphTraits::Vertex;
    using Edge = typename GraphTraits::Edge;

    /**
     * @brief Constructor taking a graph manager reference.
     *
     * @param graph_manager Reference to the graph manager
     */
    BoostGraphAnalytics(const IGraphManager<GraphTraits> &graph_manager)
        : m_graph_manager(graph_manager) {}

    /**
     * @brief Calculate Adamic-Adar index between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The Adamic-Adar score
     */
    double adamic_adar_index(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate Jaccard coefficient between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The Jaccard coefficient
     */
    double jaccard_coefficient(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate common neighbors count between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Number of common neighbors
     */
    std::size_t common_neighbors_count(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate normalized common neighbors count between two vertices.
     *
     * This metric is the common neighbors count divided by the maximum possible
     * number of common neighbors for these two vertices, which is min(deg_u, deg_v).
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Normalized common neighbors count in range [0,1]
     */
    double normalized_common_neighbors_count(const Vertex &u,
                                             const Vertex &v) const override;

    /**
     * @brief Get common neighbors between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Set of common neighbor vertices
     */
    std::vector<Vertex> get_common_neighbors(const Vertex &u,
                                             const Vertex &v) const override;

    /**
     * @brief Calculate preferential attachment score between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The preferential attachment score
     */
    double preferential_attachment(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate resource allocation index between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The resource allocation score
     */
    double resource_allocation_index(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Get the degree of a vertex.
     *
     * @param v The vertex to get the degree for.
     * @return The degree of the vertex.
     */
    std::size_t degree(const Vertex &v) const override;

    /**
     * @brief Calculate the clustering coefficient of a vertex.
     *
     * @param v The vertex to calculate the clustering coefficient for.
     * @return The clustering coefficient of the vertex.
     */
    double clustering_coefficient(const Vertex &v) const override;

    /**
     * @brief Calculate the difference in clustering coefficient between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The difference in clustering coefficient
     */
    double clustering_coefficient_difference(const Vertex &u,
                                             const Vertex &v) const override;

    /**
     * @brief Get the neighbors of a vertex.
     *
     * @param v The vertex to get the neighbors for.
     * @return The neighbors of the vertex.
     */
    std::vector<Vertex> get_neighbors(const Vertex &v) const override;

    /**
     * @brief Get the average degree of the graph.
     *
     * @return The average degree of the graph
     */

    double avg_degree() const override;

    /**
     * @brief Get access to the graph manager interface.
     *
     * @return Reference to the graph manager
     */
    const IGraphManager<GraphTraits>& get_graph_manager() const override;

private:
    const IGraphManager<GraphTraits> &m_graph_manager; // Reference to the graph manager
};

#include "BoostGraphAnalytics.tpp"