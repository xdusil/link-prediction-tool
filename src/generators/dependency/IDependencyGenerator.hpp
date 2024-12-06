#pragma once
#include <torch/torch.h>
#include <tuple>
#include <vector>

/**
 * @brief Interface for a dependency generator.
 *
 * A dependency generator is responsible for generating dependencies from contexts.
 *
 * @tparam T The type of elements in the contexts.
 */
template <typename T>
class IDependencyGenerator {
public:
    virtual ~IDependencyGenerator() = default;

    /**
     * @brief Generate dependencies from a list of contexts.
     *
     * @param context_list A vector of contexts, where each context is a vector of elements
     * of type T.
     * @return A tuple containing the context, positive target, and negative target tensors.
     */
    virtual std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
    generate_dependencies(const std::vector<std::vector<T>> &contexts) = 0;
};