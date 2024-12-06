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
     * @param initial_selector A function that selects the "initial" element from a
     * context.
     * @param total_vertices The total number of vertices in the graph.
     * @param to_long A function that converts an element of type T to a long.
     * @param num_negative_samples The number of negative samples to generate per positive
     * sample.
     */
    CandidateDependencyGenerator(
        std::function<T(const std::vector<T> &)> initial_selector,
        std::size_t total_vertices, std::function<long(T)> to_long,
        int num_negative_samples);

    /**
     * @brief Generate dependencies from a list of contexts.
     *
     * @param context_list A vector of contexts, where each context is a vector of
     * elements of type T.
     * @return A tuple containing the context, positive target, and negative target
     * tensors.
     */
    std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
    generate_dependencies(const std::vector<std::vector<T>> &contexts) override;

private:
    std::function<T(const std::vector<T> &)>
        initial_selector;           // select the "initial" element
    std::size_t total_vertices;     // total number of vertices in the graph
    std::function<long(T)> to_long; // convert an element of type T to a long
    int num_negative_samples;       // number of negative samples to generate per positive
                                    // sample
    std::mt19937 rng;               // random number generator
};

#include "CandidateDependencyGenerator.tpp"