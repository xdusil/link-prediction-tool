#pragma once

#include <ctime>
#include <optional>
#include <random>
#include <vector>

#include "../IRandomWalkLogic.hpp"
#include "graph/IGraphManager.hpp"

/**
 * @brief Custom random walk logic implementation.
 *
 * This class provides a custom implementation of the random walk logic.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 **/
template <typename GraphTraits>
class CustomRandomWalkLogic : public IRandomWalkLogic<GraphTraits> {
public:
    using Vertex = GraphTraits::Vertex;
    using Edge = GraphTraits::Edge;

    /**
     * @brief Constructor for CustomRandomWalkLogic.
     *
     * @param n_appearances Minimum appearances threshold.
     * @param epsilon Time gap threshold.
     * @param epsilon_rev Reverse flow time gap threshold.
     */
    CustomRandomWalkLogic(int n_appearances, int epsilon, int epsilon_rev)
        : n_appearances(n_appearances), epsilon(epsilon), epsilon_rev(epsilon_rev) {}

    /**
     * @brief Generate a single random walk starting from a given vertex.
     *
     * @param graph The graph on which to perform the random walk.
     * @param start_vertex The starting vertex of the random walk.
     * @param walk_length The length of the random walk.
     * @return A vector representing the vertices in the random walk.
     */
    std::vector<Vertex> generate_single_walk(const IGraphManager<GraphTraits> &graph,
                                             Vertex start_vertex,
                                             int walk_length) const override;

private:
    /**
     * @brief Determine the possible next vertices in the random walk.
     *
     * @param graph The graph on which to perform the random walk.
     * @param previous_edge The previous edge in the random walk.
     * @param walk_sequence The current walk sequence.
     * @param vertices The possible next vertices in the random walk.
     */
    void determine_random_walk_possibilities(
        const IGraphManager<GraphTraits> &graph, const Edge &previous_edge,
        const std::vector<Vertex> &walk_sequence,
        std::vector<std::pair<Vertex, Edge>> &vertices) const;

    /**
     * @brief Choose the second vertex in the random walk.
     *
     * @param graph The graph on which to perform the random walk.
     * @param fst_vertex The first vertex in the random walk.
     * @param rng The random number generator.
     * @return An optional pair containing the second vertex and the edge connecting the
     * two vertices.
     */
    std::optional<std::pair<Vertex, Edge>>
    choose_snd_vertex(const IGraphManager<GraphTraits> &graph, Vertex fst_vertex, auto &rng) const;

    /**
     * @brief Get the count of target vertices in a range of edges.
     *
     * @param graph The graph on which to perform the random walk.
     * @param it_begin The beginning of the range of edges.
     * @param it_end The end of the range of edges.
     * @return A map of target vertices to their appearance count.
     */
    auto get_target_count(const IGraphManager<GraphTraits> &graph, auto it_begin, auto it_end) const;
    
    int n_appearances; // Minimum appearances threshold
    int epsilon;       // Time gap threshold
    int epsilon_rev;   // Reverse flow time gap threshold
};

#include "CustomRandomWalkLogic.tpp"
