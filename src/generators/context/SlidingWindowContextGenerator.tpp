#pragma once
#include "SlidingWindowContextGenerator.hpp"
#include <iostream>

template <typename T>
SlidingWindowContextGenerator<T>::SlidingWindowContextGenerator(int context_size)
    : context_size(context_size) {}

// Generate contexts from a list of sequences.
template <typename T>
std::vector<std::vector<T>> SlidingWindowContextGenerator<T>::generate_contexts(
    const std::vector<std::vector<T>> &sequences) {

    std::vector<std::vector<T>> contexts;
    contexts.reserve(sequences.size()); // TODO calculate the size of the contexts vector

    for (const auto &sequence : sequences) {
        if (sequence.size() < context_size) {
            continue;
        }

        // Slide the window over the sequence
        for (size_t i = 0; i <= sequence.size() - context_size; ++i) {
            contexts.emplace_back(sequence.begin() + i,
                                  sequence.begin() + i + context_size);
        }
    }

    return contexts;
}
