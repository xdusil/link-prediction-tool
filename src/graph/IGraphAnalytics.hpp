#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <utility>
#include <vector>

// Forward declaration
template <typename GraphTraits>
class IGraphManager;

/**
 * @brief Interface providing graph analytics and link prediction metrics for directed
 * graphs.
 *
 * This interface offers methods for computing various graph statistics,
 * node-centric metrics, and link prediction scores with explicit directional semantics.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 */
template <typename GraphTraits>
class IGraphAnalytics {
public:
    // Define the graph element types
    using Vertex = typename GraphTraits::Vertex;
    using Edge = typename GraphTraits::Edge;

    virtual ~IGraphAnalytics() = default;

    // ========== Directional Link Prediction Metrics ==========

    /**
     * @brief Calculate directional Adamic-Adar index.
     *
     * Uses directional common neighbors: vertices w where (u -> w) AND (w -> v).
     * For each such w, contributes 1/log(out_degree(w)) to account for w's
     * role as an intermediary that forwards connections.
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return The directional Adamic-Adar score
     */
    virtual double adamic_adar_index(const Vertex &u, const Vertex &v) const = 0;

    /**
     * @brief Calculate directional Jaccard coefficient.
     *
     * Uses directional common neighbors: vertices w where (u -> w) AND (w -> v).
     * Jaccard = |out_neighbors(u) ∩ in_neighbors(v)| / |out_neighbors(u) ∪
     * in_neighbors(v)|
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return The directional Jaccard coefficient
     */
    virtual double jaccard_coefficient(const Vertex &u, const Vertex &v) const = 0;

    /**
     * @brief Get directional common neighbors count.
     *
     * Counts vertices w where (u -> w) AND (w -> v), forming 2-hop paths from u to v.
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return Number of directional common neighbors
     */
    virtual std::size_t common_neighbors_count(const Vertex &u,
                                               const Vertex &v) const = 0;

    /**
     * @brief Calculate normalized directional common neighbors count.
     *
     * Normalized by min(out_degree(u), in_degree(v)) - the maximum possible
     * number of directional common neighbors for these two vertices.
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return Normalized directional common neighbors count in range [0,1]
     */
    virtual double normalized_common_neighbors_count(const Vertex &u,
                                                     const Vertex &v) const = 0;

    /**
     * @brief Get directional common neighbors for predicting edge (u -> v).
     *
     * Returns vertices w where (u -> w) AND (w -> v), i.e., out_neighbors(u) ∩ in_neighbors(v).
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return Vector of directional common neighbor vertices
     */
    virtual std::vector<Vertex> get_common_neighbors(const Vertex &u,
                                                     const Vertex &v) const = 0;

    /**
     * @brief Calculate directional preferential attachment score for edge (u -> v).
     *
     * Uses out_degree(u) × in_degree(v), reflecting that:
     * - u with high out-degree is more likely to create new outgoing edges
     * - v with high in-degree is more likely to attract new incoming edges
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return The directional preferential attachment score
     */
    virtual double preferential_attachment(const Vertex &u, const Vertex &v) const = 0;

    /**
     * @brief Calculate directional resource allocation index for edge (u -> v).
     *
     * Uses directional common neighbors: for each w where (u -> w) AND (w -> v),
     * contributes 1/out_degree(w) representing w's capacity to forward resources.
     *
     * @param u Source vertex (potential edge origin)
     * @param v Destination vertex (potential edge target)
     * @return The directional resource allocation score
     */
    virtual double resource_allocation_index(const Vertex &u, const Vertex &v) const = 0;

    // ========== Node-centric Metrics ==========

    /**
     * @brief Get the in-degree of a vertex (number of incoming edges).
     *
     * @param v The vertex
     * @return In-degree of the vertex
     */
    virtual std::size_t in_degree(const Vertex &v) const = 0;

    /**
     * @brief Get the out-degree of a vertex (number of outgoing edges).
     *
     * @param v The vertex
     * @return Out-degree of the vertex
     */
    virtual std::size_t out_degree(const Vertex &v) const = 0;

    /**
     * @brief Calculate local clustering coefficient of a vertex.
     *
     * For directed graphs, measures how often out-neighbors of v are connected
     * to each other (in either direction).
     *
     * @param v The vertex
     * @return Clustering coefficient value
     */
    virtual double clustering_coefficient(const Vertex &v) const = 0;

    /**
     * @brief Calculate absolute difference in clustering coefficients.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return |clustering_coefficient(u) - clustering_coefficient(v)|
     */
    virtual double clustering_coefficient_difference(const Vertex &u,
                                                     const Vertex &v) const = 0;

    /**
     * @brief Get out-neighbors of a vertex (vertices this vertex points to).
     *
     * @param v The vertex
     * @return Vector of out-neighbor vertices
     */
    virtual std::vector<Vertex> get_out_neighbors(const Vertex &v) const = 0;

    /**
     * @brief Get in-neighbors of a vertex (vertices that point to this vertex).
     *
     * @param v The vertex
     * @return Vector of in-neighbor vertices
     */
    virtual std::vector<Vertex> get_in_neighbors(const Vertex &v) const = 0;

    // ========== Graph-level Analytics ==========

    /**
     * @brief Get the average out-degree of the graph.
     *
     * For a directed graph, avg_out_degree == avg_in_degree == |E|/|V|.
     *
     * @return The average out-degree
     */
    virtual double avg_out_degree() const = 0;

    /**
     * @brief Get access to the graph manager interface.
     *
     * @return Reference to the graph manager
     */
    virtual const IGraphManager<GraphTraits> &get_graph_manager() const = 0;
};