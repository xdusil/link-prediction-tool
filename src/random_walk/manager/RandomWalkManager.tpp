#pragma once
#include "RandomWalkManager.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <random>
#include <thread>

// Construct a new Random Walk Manager object.
template <typename GraphTraits, typename RNG>
RandomWalkManager<GraphTraits, RNG>::RandomWalkManager(
    const IGraphManager<GraphTraits> &graph, int num_threads,
    const IRandomWalkLogic<GraphTraits, RNG> &walk_logic, int walk_length,
    int seed /*= 0*/)
    : m_graph(graph), m_num_threads(num_threads), m_walk_logic(walk_logic),
      m_walk_length(walk_length) {

    if (m_num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be positive.");
    }

    if (m_walk_length <= 0) {
        throw std::invalid_argument("Walk length must be positive.");
    }

    initialize_rngs(seed);
}

// Initialize the random number generators.
template <typename GraphTraits, typename RNG>
void RandomWalkManager<GraphTraits, RNG>::initialize_rngs(int seed) const {
    m_rngs.clear();
    m_rngs.reserve(m_num_threads);
    RNG rng(seed);
    for (int i = 0; i < m_num_threads; ++i) {
        m_rngs.emplace_back(rng());
    }
}

// Generate random walks for a set of vertices.
template <typename GraphTraits, typename RNG>
std::vector<std::vector<typename GraphTraits::Vertex>>
RandomWalkManager<GraphTraits, RNG>::generate_random_walks(
    const std::vector<Vertex> &start_vertices) const {
    return generate_random_walks(start_vertices.begin(), start_vertices.end());
}

// Generate random walks for all vertices in the graph.
template <typename GraphTraits, typename RNG>
std::vector<std::vector<typename GraphTraits::Vertex>>
RandomWalkManager<GraphTraits, RNG>::generate_random_walks() const {
    auto vertices = m_graph.get_vertices();
    return generate_random_walks(vertices.first, vertices.second);
}

// Generate random walks for a set of vertices.
template <typename GraphTraits, typename RNG>
template <typename InputIt>
std::vector<std::vector<typename GraphTraits::Vertex>>
RandomWalkManager<GraphTraits, RNG>::generate_random_walks(InputIt begin,
                                                           InputIt end) const {
    auto vertex_count = std::distance(begin, end);
    std::vector<std::vector<Vertex>> all_walks(vertex_count);
    std::vector<std::thread> threads;

    auto generate_walks_segment = [this, &all_walks](InputIt start, InputIt end,
                                                     size_t offset, int rng_idx) {
        for (auto it = start; it != end; ++it, ++offset) {
            all_walks[offset] = std::move(m_walk_logic.generate_single_walk(
                m_graph, *it, m_walk_length, m_rngs[rng_idx]));
        }
    };

    size_t num_threads =
        std::min(static_cast<size_t>(m_num_threads), static_cast<size_t>(vertex_count));
    size_t segment_size = vertex_count / num_threads;

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * segment_size;
        size_t end = (i == num_threads - 1) ? vertex_count : start + segment_size;
        threads.emplace_back(generate_walks_segment, begin + start, begin + end, start,
                             i);
    }

    // Join threads
    for (auto &thread : threads) {
        thread.join();
    }

    return all_walks;
}