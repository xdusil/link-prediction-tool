#pragma once
#include "RandomWalkManager.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <random>
#include <thread>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Construct a new Random Walk Manager object.
template <typename GraphTraits, typename RNG>
RandomWalkManager<GraphTraits, RNG>::RandomWalkManager(
    const IGraphManager<GraphTraits> &graph, int num_threads,
    const IRandomWalkLogic<GraphTraits, RNG> &walk_logic, int walk_length,
    int seed /*= 0*/)
    : m_graph(graph), m_num_threads(num_threads), m_walk_logic(walk_logic),
      m_walk_length(walk_length), m_rng(seed) {

    if (m_num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be positive.");
    }

    if (m_walk_length <= 0) {
        throw std::invalid_argument("Walk length must be positive.");
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
    const std::size_t vertex_count = std::distance(begin, end);
    if (vertex_count == 0) return {};

    std::vector<std::vector<Vertex>> all_walks(vertex_count);
    const std::size_t num_threads =
        std::min(static_cast<std::size_t>(m_num_threads), vertex_count);
    int seed = m_rng();

#ifdef USE_OPENMP
    // OpenMP implementation
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        RNG rng(seed + tid);

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < vertex_count; ++i) {
            all_walks[i] = m_walk_logic.generate_single_walk(m_graph, *(begin + i),
                                                             m_walk_length, rng);
        }
    }

#else
    //  Non-OpenMP implementation
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto generate_walks_segment = [this, &all_walks, begin, seed](std::size_t start, std::size_t end,
                                                            std::size_t tid) {
        RNG rng(seed + tid);

        for (std::size_t i = start; i < end; ++i) {
            all_walks[i] = m_walk_logic.generate_single_walk(m_graph, *(begin + i),
                                                             m_walk_length, rng);
        }
    };

    const std::size_t segment_size = vertex_count / num_threads;

    for (std::size_t tid = 0; tid < num_threads; ++tid) {
        std::size_t start = tid * segment_size;
        std::size_t end = (tid == num_threads - 1) ? vertex_count : start + segment_size;

        threads.emplace_back(generate_walks_segment, start, end, tid);
    }

    // Join threads
    for (auto &thread : threads) {
        thread.join();
    }
#endif

    return all_walks;
}