#pragma once

#include "../logic/IRandomWalkLogic.hpp"
#include "IRandomWalkManager.hpp"
#include <vector>

/**
 * @brief Manager for generating random walks on a graph.
 *
 * @tparam Graph The graph type.
 * @tparam Vertex The vertex type.
 */
template <typename Graph, typename Vertex>
class RandomWalkManager : public IRandomWalkManager<Vertex> {
  public:
    /**
     * @brief Construct a new Random Walk Manager object.
     *
     * @param graph The graph to perform random walks on.
     * @param num_threads The number of threads to use for parallel execution.
     * @param walk_logic Logic object implementing single random walk generation.
     * @param walk_length The fixed length of each random walk.
     */
    RandomWalkManager(const Graph &graph, int num_threads,
                      const IRandomWalkLogic<Graph, Vertex> &walk_logic, int walk_length);

    /**
     * @brief Generate random walks for a set of vertices.
     *
     * @param start_vertices A set of starting vertices for the random walks.
     * @return A vector of random walks, where each walk is a vector of vertices.
     */
    std::vector<std::vector<Vertex>>
    generate_random_walks(const std::vector<Vertex> &start_vertices) const override;

    /**
     * @brief Generate random walks for all vertices in the graph.
     *
     * @return A vector of random walks, where each walk is a vector of vertices.
     */
    std::vector<std::vector<Vertex>> generate_random_walks() const override;

    /**
     * @brief Generate random walks for a set of vertices.
     *
     * @tparam InputIt Iterator type for the input range.
     * @param begin Iterator pointing to the start of the input range.
     * @param end Iterator pointing to the end of the input range.
     * @return A vector of random walks, where each walk is a vector of vertices.
     */
    template <typename InputIt>
    std::vector<std::vector<Vertex>> generate_random_walks(InputIt begin,
                                                           InputIt end) const;

  private:
    const Graph &m_graph; // The graph to perform random walks on
    int m_num_threads;    // The number of threads to use for parallel execution
    int m_walk_length;    // The fixed length of each random walk
    const IRandomWalkLogic<Graph, Vertex>
        &m_walk_logic; // Logic object for generating random walks
};

#include "RandomWalkManager.tpp"