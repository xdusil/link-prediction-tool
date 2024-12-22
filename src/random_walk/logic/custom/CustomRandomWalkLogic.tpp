#pragma once

#include "CustomRandomWalkLogic.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/detail/adjacency_list.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "conditions.hpp"
#include "graph/IGraphManager.hpp"
#include "graph/network/NetworkGraphDefinition.hpp"
#include "utils/utils.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>

using namespace custom_conditions;

// Get the count of appearances of each target vertex
template <typename GraphTraits>
auto CustomRandomWalkLogic<GraphTraits>::get_target_count(const IGraphManager<GraphTraits> &graph, auto it_begin,
                      auto it_end) const {
    std::unordered_map<Vertex, std::size_t> target_count;
    for (auto it = it_begin; it != it_end; ++it) {
        Vertex target = graph.get_target_vertex(*it);
        target_count[target]++;
    }
    return target_count;
}

// Choose a second vertex in the random walk
template <typename GraphTraits>
std::optional<std::pair<typename GraphTraits::Vertex, typename GraphTraits::Edge>>
CustomRandomWalkLogic<GraphTraits>::choose_snd_vertex(
    const IGraphManager<GraphTraits> &graph, Vertex fst_vertex, auto &rng) const {
    auto edges = graph.get_out_edges(fst_vertex);
    if (edges.first == edges.second) {
        return std::nullopt; // No outgoing edges from the first vertex
    }

    std::vector<std::pair<Vertex, Edge>> vertices;
    auto distance = std::distance(edges.first, edges.second);
    vertices.reserve(distance);

    // Get the count of appearances of each target vertex
    auto target_count = get_target_count(graph, edges.first, edges.second);

    // Filter edges that meet the appearance threshold imposed on the target vertex
    for (auto it = edges.first; it != edges.second; ++it) {
        Vertex target = graph.get_target_vertex(*it);
        if (target_count[target] < n_appearances) {
            continue;
        }
        vertices.push_back({target, *it});
    }

    // If no vertices meet the appearance threshold, choose a random edge
    if (vertices.empty()) {
        auto maybe_edge = utils::choice_random_item(edges.first, edges.second, rng, distance);
        if (!maybe_edge.has_value()) {
            return std::nullopt;
        }
        Edge edge = maybe_edge.value();
        return std::make_pair(graph.get_target_vertex(edge), edge);
    }

    return utils::choice_random_item(vertices, rng);
}

// Generate a single random walk starting from a given vertex
template <typename GraphTraits>
std::vector<typename GraphTraits::Vertex>
CustomRandomWalkLogic<GraphTraits>::generate_single_walk(
    const IGraphManager<GraphTraits> &graph, Vertex start_vertex, int walk_length) const {
    std::vector<Vertex> walk;
    walk.reserve(walk_length);
    walk.push_back(start_vertex);

    boost::random::mt19937 rng(
    static_cast<unsigned int>(std::time(0))); // Initialize random number generator
    
    // Choose the second vertex in the random walk
    std::optional<std::pair<Vertex, Edge>> next_pair =
        choose_snd_vertex(graph, start_vertex, rng);
    if (!next_pair.has_value()) {
        return walk; // No outgoing edges from the first vertex
    }

    walk.push_back(next_pair->first);
    Edge previous_edge = next_pair->second;

    // Continue the walk until the desired length is reached
    while (walk.size() < walk_length) {

        std::vector<std::pair<Vertex, Edge>> vertices;
        determine_random_walk_possibilities(graph, previous_edge, walk, vertices);

        auto next_pair = utils::choice_random_item(vertices, rng);
        if (!next_pair.has_value()) {
            break;
        }

        walk.push_back(next_pair->first);
        previous_edge = next_pair->second;
    }

    return walk;
}

// Determine the possible next vertices in the random walk
template <typename GraphTraits>
void CustomRandomWalkLogic<GraphTraits>::determine_random_walk_possibilities(
    const IGraphManager<GraphTraits> &graph, const Edge &previous_edge,
    const std::vector<Vertex> &walk_sequence,
    std::vector<std::pair<Vertex, Edge>> &vertices) const {

    if (walk_sequence.empty()) {
        throw std::runtime_error(
            "Walk sequence is empty, cannot access the last vertex.");
    }

    if (walk_sequence.size() < 2) {
        throw std::runtime_error(
            "Walk sequence is too short, cannot access the second to last vertex.");
        return;
    }

    Vertex previous_vertex = walk_sequence.back();
    auto edges = graph.get_out_edges(previous_vertex);

    if (edges.first == edges.second) {
        // No outgoing edges from the previous vertex
        return;
    }

    // Count the appearances of each target
    auto target_count = get_target_count(graph, edges.first, edges.second);

    // Check each edge from the previous vertex
    for (auto it = edges.first; it != edges.second; ++it) {
        Vertex target = graph.get_target_vertex(*it);

        // Check if the target meets the appearance threshold
        if (target_count[target] < n_appearances) {
            continue;
        }

        const EdgeProperties &current_props = graph.get_edge_properties(*it);
        const EdgeProperties &previous_props = graph.get_edge_properties(previous_edge);

        bool was_two_steps_ago = (walk_sequence[walk_sequence.size() - 2] == target);

        // Check if the edge satisfies any of the conditions
        if (satisfies_LR_opening_conditions(current_props, previous_props,
                                            was_two_steps_ago) ||
            satisfies_return_from_LR_conditions(current_props, previous_props,
                                                walk_sequence, target,
                                                was_two_steps_ago) ||
            satisfies_return_over_reverse_edge_conditions(
                current_props, previous_props, walk_sequence, target, was_two_steps_ago,
                epsilon_rev)) {
            vertices.push_back({target, *it});
        }
    }

    auto prev_prev_vertex = walk_sequence[walk_sequence.size() - 2];
    auto prev_prev_edges = graph.get_out_edges(prev_prev_vertex);
    auto prev_prev_target_count = get_target_count(graph, prev_prev_edges.first, prev_prev_edges.second);


    for (auto it = prev_prev_edges.first; it != prev_prev_edges.second; ++it) {
        Vertex target = graph.get_target_vertex(*it);

        if (prev_prev_target_count[target] < n_appearances) {
            continue;
        }

        const EdgeProperties &current_props = graph.get_edge_properties(*it);
        const EdgeProperties &previous_props = graph.get_edge_properties(previous_edge);

        if (satisfies_RR_opening_conditions(current_props, previous_props, epsilon,
                                            previous_vertex == target)) {
            vertices.push_back({target, *it});
        }
    }
}
