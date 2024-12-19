#pragma once
#include "RandomWalkManager.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <thread>

// Construct a new Random Walk Manager object.
template <typename GraphTraits>
RandomWalkManager<GraphTraits>::RandomWalkManager(
    const IGraphManager<GraphTraits> &graph, int num_threads,
    const IRandomWalkLogic<GraphTraits> &walk_logic, int walk_length)
    : m_graph(graph), m_num_threads(num_threads), m_walk_logic(walk_logic),
      m_walk_length(walk_length) {
    if (m_num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be positive.");
    }

    if (m_walk_length <= 0) {
        throw std::invalid_argument("Walk length must be positive.");
    }
}

// Generate random walks for a set of vertices.
template <typename GraphTraits>
std::vector<std::vector<typename GraphTraits::Vertex>>
RandomWalkManager<GraphTraits>::generate_random_walks(
    const std::vector<Vertex> &start_vertices) const {
    return generate_random_walks(start_vertices.begin(), start_vertices.end());
}

// Generate random walks for all vertices in the graph.
template <typename GraphTraits>
std::vector<std::vector<typename GraphTraits::Vertex>>
RandomWalkManager<GraphTraits>::generate_random_walks() const {
    auto vertices = m_graph.get_vertices();
    return generate_random_walks(vertices.first, vertices.second);
}

// Generate random walks for a set of vertices.
template <typename GraphTraits>
template <typename InputIt>
std::vector<std::vector<typename GraphTraits::Vertex>>
RandomWalkManager<GraphTraits>::generate_random_walks(InputIt begin, InputIt end) const {
    auto vertex_count = std::distance(begin, end);
    std::vector<std::vector<Vertex>> all_walks(vertex_count);
    std::vector<std::thread> threads;

    auto generate_walks_segment = [this, &begin, &all_walks, vertex_count](
                                      InputIt start, InputIt end, size_t offset) {
        for (auto it = start; it != end; ++it, ++offset) {
            all_walks[offset] =
                std::move(m_walk_logic.generate_single_walk(m_graph, *it, m_walk_length));
        }
    };

    size_t num_threads =
        std::min(static_cast<size_t>(m_num_threads), static_cast<size_t>(vertex_count));
    size_t segment_size = vertex_count / num_threads;

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * segment_size;
        size_t end = (i == num_threads - 1) ? vertex_count : start + segment_size;
        threads.emplace_back(generate_walks_segment, begin + start, begin + end, start);
    }

    // Join threads
    for (auto &thread : threads) {
        thread.join();
    }

    return all_walks;
}