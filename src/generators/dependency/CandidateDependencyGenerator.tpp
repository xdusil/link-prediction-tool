#pragma once

#include "CandidateDependencyGenerator.hpp"
#include <c10/core/TensorImpl.h>
#include <iostream>

// Constructor
template <typename T>
CandidateDependencyGenerator<T>::CandidateDependencyGenerator(
    std::function<T(const std::vector<T> &)> initial_selector, std::size_t total_vertices,
    std::function<long(T)> to_long, int num_negative_samples)
    : initial_selector(std::move(initial_selector)), total_vertices(total_vertices),
      to_long(std::move(to_long)), num_negative_samples(num_negative_samples),
      rng(std::random_device{}()) {}

// Generate dependencies from a list of contexts.
template <typename T>
std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
CandidateDependencyGenerator<T>::generate_dependencies(
    const std::vector<std::vector<T>> &context_list) {
    std::vector<long> contexts;
    std::vector<long> positives;
    std::vector<std::vector<long>> negatives;

    std::uniform_int_distribution<long> dist(0, static_cast<long>(total_vertices) - 1);

    for (const auto &context : context_list) {
        if (context.empty()) {
            continue;
        }

        // Use the provided initial selector to determine the "target" element
        T target = initial_selector(context);

        for (size_t i = 0; i < context.size(); ++i) {
            // Avoid self-dependency
            if (context[i] != target) {
                contexts.push_back(to_long(context[i]));
                positives.push_back(to_long(target));

                negatives.push_back({});
                negatives.back().reserve(num_negative_samples);

                // Generate negative samples - naive approach
                for (int j = 0; j < num_negative_samples; ++j) {
                    long negative = dist(rng);
                    if (negative != to_long(target) && negative != to_long(context[i])) {
                        negatives.back().push_back(negative);
                    }
                }
            }
        }
    }

    std::vector<torch::Tensor> negatives_tensor;
    negatives_tensor.reserve(negatives.size());
    for (const auto &neg : negatives) {
        negatives_tensor.push_back(torch::tensor(neg));
    }

    return {torch::tensor(contexts), torch::tensor(positives), negatives_tensor};
}
