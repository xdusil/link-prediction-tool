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
     */
    CandidateDependencyGenerator(
        std::function<T(const std::vector<T> &)> initial_selector,
        std::size_t total_vertices, std::function<long(T)> to_long);

    /**
     * @brief Generate dependencies from a list of contexts.
     *
     * For each context, the "initial" element is selected using the provided initial
     * selector. For each element in the context that is not the "initial" element, a
     * candidate dependency is generated with the "initial" element as the target and the
     * current element as the context.
     *
     * @param contexts A vector of contexts, where each context is a vector of elements of
     * type T.
     * @return A pair of tensors representing the context and target dependencies.
     */
    std::pair<torch::Tensor, torch::Tensor>
    generate_dependencies(const std::vector<std::vector<T>> &contexts) override;

private:
    std::function<T(const std::vector<T> &)>
        initial_selector;           // Function to select the "initial" element
    std::size_t total_vertices;     // Total number of vertices in the graph
    std::function<long(T)> to_long; // Function to convert an element of type T to a long
};

#include "CandidateDependencyGenerator.tpp"
