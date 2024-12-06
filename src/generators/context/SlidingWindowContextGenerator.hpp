#pragma once
#include "IContextGenerator.hpp"

/**
 * @brief Sliding window context generator.
 *
 * This context generator creates contexts by sliding a window over the input sequences.
 * The window size is specified by the context_size parameter.
 *
 * @tparam T The type of elements in the sequences.
 */
template <typename T>
class SlidingWindowContextGenerator : public IContextGenerator<T> {
public:
    /**
     * @brief Constructs a new SlidingWindowContextGenerator object.
     *
     * @param context_size The size of the context window.
     */
    explicit SlidingWindowContextGenerator(int context_size);

    /**
     * @brief Generate contexts from a list of sequences.
     *
     * @param sequences A vector of sequences, where each sequence is a vector of elements
     * of type T.
     * @return A vector of contexts, where each context is a vector of elements of type T.
     */
    std::vector<std::vector<T>>
    generate_contexts(const std::vector<std::vector<T>> &sequences) override;

private:
    int context_size; // The size of the context window
};

#include "SlidingWindowContextGenerator.tpp"