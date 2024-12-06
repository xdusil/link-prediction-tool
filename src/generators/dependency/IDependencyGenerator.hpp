#pragma once
#include <torch/torch.h>
#include <utility>
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
     * @param contexts A vector of contexts, where each context is a vector of elements of
     * type T.
     * @return A pair of tensors representing the context and target dependencies.
     */
    virtual std::pair<torch::Tensor, torch::Tensor>
    generate_dependencies(const std::vector<std::vector<T>> &contexts) = 0;
};
