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
     * @param context_list A vector of contexts, where each context is a vector of
     * elements of type T.
     * @return A tuple containing three tensors of equal length `num_pairs`:
     *   - sources_tensor    [num_pairs]
     *   - targets_tensor    [num_pairs]
     *   - negatives_tensor  [num_pairs, num_negative_samples]
     */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    generate_dependencies(const std::vector<std::vector<T>> &contexts) = 0;
};