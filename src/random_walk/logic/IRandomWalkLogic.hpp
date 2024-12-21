#pragma once

#include "graph/IGraphManager.hpp"
#include <vector>

/**
 * @brief Interface for generating random walks on a graph.
 *
 * This interface provides a method for generating a single random walk starting from a
 * given vertex.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 */
template <typename GraphTraits>
class IRandomWalkLogic {
public:
    virtual ~IRandomWalkLogic() = default;

    /**
     * @brief Generate a single random walk starting from a given vertex.
     *
     * @param graph The graph on which to perform the random walk.
     * @param start_vertex The starting vertex of the random walk.
     * @param walk_length The length of the random walk.
     * @return A vector representing the vertices in the random walk.
     */
    virtual std::vector<typename GraphTraits::Vertex>
    generate_single_walk(const IGraphManager<GraphTraits> &graph,
                         GraphTraits::Vertex start_vertex, int walk_length) const = 0;
};