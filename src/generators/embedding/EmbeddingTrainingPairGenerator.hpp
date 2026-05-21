#pragma once

#include "IEmbeddingTrainingPairGenerator.hpp"
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

/**
 * @brief Generates skip-gram style training pairs for the embedding model.
 *
 * For each context, the first vertex is treated as the source and the remaining vertices
 * are positive targets. Negative targets are sampled uniformly from graph vertex ids.
 *
 * @tparam T The vertex type stored in generated contexts.
 */
template <typename T>
class EmbeddingTrainingPairGenerator : public IEmbeddingTrainingPairGenerator<T> {
public:
    /**
     * @brief Construct a training-pair generator.
     *
     * @param total_vertices Number of vertices available for negative sampling.
     * @param to_long Converts a vertex value from a context to the embedding row id.
     * @param num_negative_samples Number of negative targets generated for each
     * source/positive-target pair.
     * @param seed Seed for deterministic negative sampling.
     * @throws std::invalid_argument if total_vertices is zero or num_negative_samples is
     * negative.
     */
    EmbeddingTrainingPairGenerator(std::size_t total_vertices,
                                   std::function<long(const T&)> to_long,
                                   int num_negative_samples, int seed = 42);

    /**
     * @brief Convert contexts to tensors consumed by the embedding trainer.
     *
     * For each context `[src, dst1, dst2, ...]`, this method emits pairs
     * `(src, dst1)`, `(src, dst2)`, ... and samples `num_negative_samples` negative
     * targets per pair. Contexts with fewer than two vertices are ignored.
     *
     * @param contexts Random-walk contexts produced by the context generator.
     * @return Tuple containing:
     * - source ids tensor with shape `[num_pairs]`
     * - positive target ids tensor with shape `[num_pairs]`
     * - negative target ids tensor with shape `[num_pairs, num_negative_samples]`
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    generate_training_pairs(const std::vector<std::vector<T>>& contexts) override;

private:
    std::size_t m_total_vertices;
    std::function<long(const T&)> m_to_long;
    int m_num_negative_samples;
    std::mt19937 m_rng;
};

#include "EmbeddingTrainingPairGenerator.tpp"
