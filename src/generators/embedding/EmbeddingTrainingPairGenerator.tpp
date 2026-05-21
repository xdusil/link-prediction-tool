#pragma once

#include "EmbeddingTrainingPairGenerator.hpp"
#include <cstdint>
#include <stdexcept>
#include <utility>

template <typename T>
EmbeddingTrainingPairGenerator<T>::EmbeddingTrainingPairGenerator(
    std::size_t total_vertices, std::function<long(const T&)> to_long,
    int num_negative_samples, int seed /*= 42*/)
    : m_total_vertices(total_vertices), m_to_long(std::move(to_long)),
      m_num_negative_samples(num_negative_samples), m_rng(seed) {
    if (m_total_vertices == 0) {
        throw std::invalid_argument("EmbeddingTrainingPairGenerator requires vertices.");
    }
    if (m_num_negative_samples < 0) {
        throw std::invalid_argument("Negative sample count cannot be negative.");
    }
}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EmbeddingTrainingPairGenerator<T>::generate_training_pairs(
    const std::vector<std::vector<T>>& context_list) {
    // Count pairs: each context of size N has (N-1) source->target pairs
    int64_t num_pairs = 0;
    for (const auto& context : context_list) {
        if (context.size() > 1) {
            num_pairs += static_cast<int64_t>(context.size() - 1);
        }
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kInt64)
                       .device(torch::kCPU)
                       .memory_format(torch::MemoryFormat::Contiguous);

    torch::Tensor sources_tensor = torch::empty({num_pairs}, options);
    torch::Tensor targets_tensor = torch::empty({num_pairs}, options);
    torch::Tensor negatives_tensor =
        torch::empty({num_pairs, m_num_negative_samples}, options);

    auto src_acc = sources_tensor.accessor<int64_t, 1>();
    auto tgt_acc = targets_tensor.accessor<int64_t, 1>();
    auto neg_acc = negatives_tensor.accessor<int64_t, 2>();

    std::uniform_int_distribution<int64_t> dist(
        0, static_cast<int64_t>(m_total_vertices - 1));

    std::size_t idx = 0;
    for (const auto& context : context_list) {
        if (context.size() < 2) {
            continue;
        }

        // The FIRST element is the source
        const long source_id = m_to_long(context[0]);

        // The REMAINING elements are targets
        for (std::size_t i = 1; i < context.size(); ++i) {
            const long target_id = m_to_long(context[i]);

            src_acc[idx] = source_id;
            tgt_acc[idx] = target_id;

            for (int j = 0; j < m_num_negative_samples; ++j) {
                long negative;
                int attempts = 0;
                constexpr int max_attempts = 10;

                do {
                    negative = dist(m_rng);
                    ++attempts;
                    if (attempts >= max_attempts) {
                        negative = (target_id + static_cast<long>(j) + 1) %
                                   static_cast<long>(m_total_vertices);
                        break;
                    }
                } while (negative == target_id || negative == source_id);

                neg_acc[idx][j] = negative;
            }

            ++idx;
        }
    }

    return {sources_tensor, targets_tensor, negatives_tensor};
}
