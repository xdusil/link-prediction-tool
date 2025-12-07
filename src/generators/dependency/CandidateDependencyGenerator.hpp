#pragma once
#include "IDependencyGenerator.hpp"
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

/**
 * @brief A dependency generator that generates candidate dependencies from contexts.
 *
 * @tparam T The type of elements in the contexts.
 */
template <typename T>
class CandidateDependencyGenerator : public IDependencyGenerator<T> {
public:
    /**
     * @brief Constructs a new CandidateDependencyGenerator object.
     *
     * @param total_vertices The total number of vertices in the graph.
     * @param to_long A function that converts an element of type T to a long.
     * @param num_negative_samples The number of negative samples to generate per positive
     * sample.
     */
    CandidateDependencyGenerator(std::size_t total_vertices,
                                 std::function<long(const T&)> to_long,
                                 int num_negative_samples, int seed = 42);

    /**
     * @brief Generate dependencies from a list of contexts.
     *
     * For each context, the first element is treated as the source, and all subsequent
     * elements are treated as targets.
     * Negative samples are drawn uniformly from all vertices.
     *
     * @param context_list A vector of contexts, where each context is a vector of
     * elements of type T.
     * @return A tuple containing three tensors of equal length `num_pairs`:
     *   - sources_tensor    [num_pairs]
     *   - targets_tensor    [num_pairs]
     *   - negatives_tensor  [num_pairs, num_negative_samples]
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    generate_dependencies(const std::vector<std::vector<T>>& contexts) override;

private:
    std::size_t m_total_vertices;          // total number of vertices in the graph
    std::function<long(const T&)> to_long; // convert an element of type T to a long
    int m_num_negative_samples; // number of negative samples to generate per positive
                                // sample
    std::mt19937 m_rng;         // random number generator
};

#include "CandidateDependencyGenerator.tpp"