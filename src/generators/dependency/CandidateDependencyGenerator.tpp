#pragma once

#include "CandidateDependencyGenerator.hpp"
#include <c10/core/TensorImpl.h>
#include <iostream>

// Constructor
template <typename T>
CandidateDependencyGenerator<T>::CandidateDependencyGenerator(
    std::function<T(const std::vector<T> &)> initial_selector, std::size_t total_vertices,
    std::function<long(T)> to_long)
    : initial_selector(std::move(initial_selector)), total_vertices(total_vertices),
      to_long(std::move(to_long)) {}

// Generate dependencies from a list of contexts.
template <typename T>
std::pair<torch::Tensor, torch::Tensor>
CandidateDependencyGenerator<T>::generate_dependencies(
    const std::vector<std::vector<T>> &context_list) {
    std::vector<long> contexts;
    std::vector<long> targets;

    for (const auto &context : context_list) {
        if (context.empty()) {
            continue;
        }

        // Use the provided initial selector to determine the "initial" element
        T initial = initial_selector(context);

        for (size_t i = 0; i < context.size(); ++i) {
            // Avoid self-dependency
            if (context[i] != initial) {
                contexts.push_back(to_long(context[i]));
                targets.push_back(to_long(initial));
            }
        }
    }

    return {torch::tensor(contexts), torch::tensor(targets)};
}
