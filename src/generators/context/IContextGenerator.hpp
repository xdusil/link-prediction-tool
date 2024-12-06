#pragma once
#include <vector>

/**
 * @brief Interface for a context generator.
 *
 * A context generator is responsible for generating contexts from a list of sequences.
 * The sequences can be random walks, sentences, or any other type of sequence.
 *
 * @tparam T The type of elements in the sequences.
 */
template <typename T>
class IContextGenerator {
public:
    virtual ~IContextGenerator() = default;

    /**
     * @brief Generate contexts from a list of sequences (e.g., random walks).
     *
     * @param sequences A vector of sequences, where each sequence is a vector of elements
     * of type T.
     * @return A vector of contexts, where each context is a vector of elements of type T.
     */
    virtual std::vector<std::vector<T>>
    generate_contexts(const std::vector<std::vector<T>> &sequences) = 0;
};
