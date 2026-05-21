#pragma once

#include <torch/torch.h>
#include <tuple>
#include <vector>

/**
 * @brief Interface for generating source/target/negative samples for embedding training.
 *
 * This generator belongs to the embedding pipeline. It consumes random-walk contexts and
 * converts them to tensors used by the embedding model.
 *
 * @tparam T The vertex type stored in generated contexts.
 */
template <typename T>
class IEmbeddingTrainingPairGenerator {
public:
    virtual ~IEmbeddingTrainingPairGenerator() = default;

    /**
     * @brief Generate embedding training tensors from a batch of contexts.
     *
     * @param contexts Contexts generated from random walks.
     * @return A tuple containing three tensors of equal length `num_pairs`:
     *   - sources_tensor    [num_pairs]
     *   - targets_tensor    [num_pairs]
     *   - negatives_tensor  [num_pairs, num_negative_samples]
     */
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    generate_training_pairs(const std::vector<std::vector<T>>& contexts) = 0;
};
