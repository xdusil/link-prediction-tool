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
     * @brief Calculate directional Adamic-Adar index.
     *
     * Uses directional common neighbors: vertices w where (u -> w) AND (w -> v).
     * For each such w, contributes 1/log(out_degree(w)).
     * out_degree(w) is used because w acts as an intermediary forwarding connections.
     *
     * @param u Source vertex
     * @param v Destination vertex
     * @return The directional Adamic-Adar score
     */
    double adamic_adar_index(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate directional Jaccard coefficient.
     *
     * Uses directional common neighbors: vertices w where (u -> w) AND (w -> v).
     * Jaccard = |out_neighbors(u) ∩ in_neighbors(v)| / |out_neighbors(u) ∪ in_neighbors(v)|
     * 
     * @param u Source vertex
     * @param v Destination vertex
     * @return The directional Jaccard coefficient
     */
    double jaccard_coefficient(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Get directional common neighbors count.
     *
     * Counts vertices w where (u -> w) AND (w -> v), forming 2-hop paths from u to v.
     * 
     * @param u Source vertex
     * @param v Destination vertex
     * @return Number of directional common neighbors
     */
    std::size_t common_neighbors_count(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate normalized directional common neighbors count.
     *
     * Normalized by min(out_degree(u), in_degree(v)) - the maximum possible
     * number of directional common neighbors for these two vertices.
     * 
     * @param u Source vertex
     * @param v Destination vertex
     * @return Normalized directional common neighbors count in range [0,1]
     */
    double normalized_common_neighbors_count(const Vertex &u,
                                             const Vertex &v) const override;

    /**
     * @brief Get directional common neighbors for predicting edge (u -> v).
     *
     * Returns vertices w where (u -> w) AND (w -> v), i.e., out_neighbors(u) ∩ in_neighbors(v).
     *
     * @param u Source vertex
     * @param v Destination vertex
     * @return Vector of directional common neighbor vertices
     */
    std::vector<Vertex> get_common_neighbors(const Vertex &u,
                                             const Vertex &v) const override;

    /**
     * @brief Calculate directional preferential attachment for edge (u -> v).
     *
     * Uses out_degree(u) × in_degree(v), reflecting that:
     * - u with high out-degree is more likely to create new outgoing edges
     * - v with high in-degree is more likely to attract new incoming edges
     *
     * @param u Source vertex
     * @param v Destination vertex
     * @return The directional preferential attachment score
     */
    double preferential_attachment(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Calculate directional resource allocation index for edge (u -> v).
     *
     * Uses directional common neighbors: for each w where (u -> w) AND (w -> v),
     * contributes 1/out_degree(w) representing w's capacity to forward resources.
     * 
     * @param u Source vertex
     * @param v Destination vertex
     * @return The directional resource allocation score
     */
    double resource_allocation_index(const Vertex &u, const Vertex &v) const override;

    /**
     * @brief Get the in-degree of a vertex.
     *
     * @param v The vertex to get the in-degree for.
     * @return The in-degree of the vertex.
     */
    std::size_t in_degree(const Vertex &v) const override;

    /**
     * @brief Get the out-degree of a vertex.
     *
     * @param v The vertex to get the out-degree for.
     * @return The out-degree of the vertex.
     */
    std::size_t out_degree(const Vertex &v) const override;

    /**
     * @brief Calculate the clustering coefficient of a vertex.
     *
     * Uses out-neighbors and checks for edges between them (in either direction).
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
     * @brief Get the out-neighbors of a vertex.
     *
     * @param v The vertex to get the out-neighbors for.
     * @return The out-neighbors of the vertex.
     */
    std::vector<Vertex> get_out_neighbors(const Vertex &v) const override;

    /**
     * @brief Get the in-neighbors of a vertex.
     *
     * @param v The vertex to get the in-neighbors for.
     * @return The in-neighbors of the vertex.
     */
    std::vector<Vertex> get_in_neighbors(const Vertex &v) const override;

    /**
     * @brief Get the average out-degree of the graph.
     *
     * For a directed graph, avg_out_degree == avg_in_degree == |E|/|V|.
     * 
     * @return The average out-degree of the graph
     */
    double avg_out_degree() const override;

    /**
     * @brief Get access to the graph manager interface.
     *
     * @return Reference to the graph manager
     */
    const IGraphManager<GraphTraits> &get_graph_manager() const override;

private:
    const IGraphManager<GraphTraits> &m_graph_manager; // Reference to the graph manager
};

#include "BoostGraphAnalytics.tpp"