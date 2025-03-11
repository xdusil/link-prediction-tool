#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <utility>
#include <vector>

/**
 * @brief Interface providing graph analytics and link prediction metrics.
 *
 * This interface offers methods for computing various graph statistics,
 * node-centric metrics, and link prediction scores.
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

    // ---------- Link Prediction Metrics ----------

    /**
     * @brief Calculate Adamic-Adar index between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The Adamic-Adar score
     */
    virtual double adamic_adar(const Vertex &u, const Vertex &v) const = 0;

    /**
     * @brief Calculate Jaccard coefficient between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The Jaccard coefficient
     */
    virtual double jaccard_coefficient(const Vertex &u, const Vertex &v) const = 0;

    /**
     * @brief Calculate common neighbors count between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Number of common neighbors
     */
    virtual std::size_t common_neighbors_count(const Vertex &u,
                                               const Vertex &v) const = 0;

    /**
     * @brief Get common neighbors between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return Set of common neighbor vertices
     */
    virtual std::vector<Vertex> get_common_neighbors(const Vertex &u,
                                                     const Vertex &v) const = 0;

    /**
     * @brief Calculate preferential attachment score between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The preferential attachment score
     */
    virtual double preferential_attachment(const Vertex &u, const Vertex &v) const = 0;

    /**
     * @brief Calculate resource allocation index between two vertices.
     *
     * @param u First vertex
     * @param v Second vertex
     * @return The resource allocation score
     */
    virtual double resource_allocation(const Vertex &u, const Vertex &v) const = 0;

    // ---------- Node-centric Metrics ----------

    /**
     * @brief Calculate degree of a vertex.
     *
     * @param v The vertex
     * @return Degree of the vertex
     */
    virtual std::size_t degree(const Vertex &v) const = 0;

    /**
     * @brief Calculate local clustering coefficient of a vertex.
     *
     * @param v The vertex
     * @return Clustering coefficient value
     */
    virtual double clustering_coefficient(const Vertex &v) const = 0;

    /**
     * @brief Calculate difference in clustering coefficients of two vertices.
     *
     * @param u First vertex
        * @param v Second vertex
        * @return Difference in clustering coefficients

    */
    virtual double clustering_coefficient_difference(const Vertex &u,
                                                     const Vertex &v) const = 0;

    /**
     * @brief Get neighbors of a vertex.
     *
     * @param v The vertex
     * @return Vector of neighbor vertices
     */
    virtual std::vector<Vertex> get_neighbors(const Vertex &v) const = 0;

    // ---------- Graph Analytics ----------

    /**
     * @brief Get the average degree of the graph.
     *
     * @return The average degree
     */
    virtual double avg_degree() const = 0;
};