#pragma once

#include "CandidateDependencyGenerator.hpp"
#include <iostream>

template <typename T>
CandidateDependencyGenerator<T>::CandidateDependencyGenerator(
    std::size_t total_vertices, std::function<long(const T&)> to_long, int num_negative_samples,
    int seed /*= 42*/)
    : m_total_vertices(total_vertices), to_long(std::move(to_long)),
      m_num_negative_samples(num_negative_samples), m_rng(seed) {}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CandidateDependencyGenerator<T>::generate_dependencies(
    const std::vector<std::vector<T>>& context_list) {
    // Count pairs: each context of size N has (N-1) source->target pairs
    int64_t num_pairs = 0;
    for (const auto& context : context_list) {
        if (context.size() > 1) {
            num_pairs += context.size() - 1;
        }
    }

    // Create tensors directly with the right size
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt64)
                       .device(torch::kCPU)
                       .memory_format(torch::MemoryFormat::Contiguous);

    torch::Tensor sources_tensor = torch::empty({num_pairs}, options);
    torch::Tensor targets_tensor = torch::empty({num_pairs}, options);
    torch::Tensor negatives_tensor =
        torch::empty({num_pairs, m_num_negative_samples}, options);

    // Create accessors for direct indexing
    auto src_acc = sources_tensor.accessor<int64_t, 1>();
    auto tgt_acc = targets_tensor.accessor<int64_t, 1>();
    auto neg_acc = negatives_tensor.accessor<int64_t, 2>();

    std::uniform_int_distribution<int64_t> dist(0, m_total_vertices - 1);

    // Second pass to fill the tensors
    std::size_t idx = 0;
    for (const auto& context : context_list) {
        if (context.size() < 2)
            continue;

        // The FIRST element is the source
        T source_vertex = context[0];
        long source_id = to_long(source_vertex);

        // All remaining elements are targets
        for (std::size_t i = 1; i < context.size(); ++i) {
            long target_id = to_long(context[i]);

            src_acc[idx] = source_id;
            tgt_acc[idx] = target_id;

            // Negative sampling
            for (int j = 0; j < m_num_negative_samples; ++j) {
                long negative;
                int attempts = 0;
                const int max_attempts = 10;

                do {
                    negative = dist(m_rng);
                    ++attempts;
                    if (attempts >= max_attempts) {
                        // Fallback
                        negative = (target_id + j + 1) % m_total_vertices;
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