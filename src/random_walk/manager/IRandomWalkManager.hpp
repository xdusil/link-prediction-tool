#pragma once

#include <vector>

/**
 * @brief Interface for managing random walks on a graph.
 *
 * @tparam Vertex The vertex type of the graph.
 */
template <typename Vertex> class IRandomWalkManager {
  public:
    virtual ~IRandomWalkManager() = default;

    /**
     * @brief Generate random walks for the given vertices.
     *
     * @param start_vertices A set of starting vertices for the random walks
     * @return A vector of random walks, where each walk is a vector of vertices.
     */
    virtual std::vector<std::vector<Vertex>>
    generate_random_walks(const std::vector<Vertex> &start_vertices) const = 0;

    /**
     * @brief Generate random walks for all vertices in the graph.
     *
     * @return A vector of random walks, where each walk is a vector of vertices.
     */
    virtual std::vector<std::vector<Vertex>> generate_random_walks() const = 0;
};