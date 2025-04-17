#pragma once

#include <armadillo>
#include <torch/torch.h>
#ifdef USE_OPENMP
    #include <omp.h>
#endif
#include <optional>
#include <random>
#include <tuple>
#include <vector>

/**
 * @brief Utility functions.
 */
namespace utils {

/**
 * @brief Choose a random item from a std::vector.
 *
 * @tparam T   The type of the items in the vector.
 * @tparam RNG The random number generator type.
 * @param items The vector of items.
 * @param rng   The random number generator reference.
 * @return An optional item from the vector, or std::nullopt if empty.
 */
template <typename T, typename RNG>
std::optional<T> choice_random_item(std::vector<T> &items, RNG &rng);

/**
 * @brief Choose a random item from an iterator range (random-access only).
 *
 * @tparam RandomIt The random-access iterator type.
 * @tparam RNG      The random number generator type.
 * @param it_begin  The beginning of the range.
 * @param it_end    The end of the range.
 * @param rng       The random number generator reference.
 * @param distance  Optional: if provided, uses that as the range length;
 *                  otherwise calculates via std::distance.
 * @return An optional item from the range, or std::nullopt if empty.
 */
template <typename RandomIt, typename RNG>
std::optional<typename std::iterator_traits<RandomIt>::value_type>
choice_random_item(RandomIt it_begin, RandomIt it_end, RNG &rng,
                   std::optional<std::size_t> distance = std::nullopt);

/**
 * @brief Splits the dataset into training and testing subsets.
 *
 * @tparam T The type of the labels.
 * @param features The feature matrix (features x samples).
 * @param labels The label row vector (1 x samples).
 * @param train_fraction The fraction of data to be used for training (e.g., 0.7 for 70%).
 * @return A tuple containing:
 *         - Training features
 *         - Training labels
 *         - Testing features
 *         - Testing labels
 */
template <typename T>
std::tuple<arma::mat, arma::Row<T>, arma::mat, arma::Row<T>>
split_train_test(const arma::mat &features, const arma::Row<T> &labels,
                 double train_fraction = 0.7);

/**
 * @brief Check if a vertex pair is in a sequence.
 *
 * @tparam Vertex The vertex type.
 * @param sequence The sequence of vertices.
 * @param src The source vertex.
 * @param dst The destination vertex.
 * @return True if the vertex pair is in the sequence, false otherwise.
 */
template <typename Vertex>
bool is_vertex_pair_in_sequence(const std::vector<Vertex> &sequence, Vertex src,
                                Vertex dst);

/**
 * @brief Check if a vertex pair is in a sequence in the opposite order.
 *
 * @tparam Vertex The vertex type.
 * @param sequence The sequence of vertices.
 * @param src The source vertex.
 * @param dst The destination vertex.
 * @return True if the vertex pair is in the sequence in the opposite order, false
 * otherwise.
 */
template <typename Vertex>
bool is_vertex_pair_in_sequence_opposite(const std::vector<Vertex> &sequence, Vertex src,
                                         Vertex dst);

// Structure to hold both the Armadillo matrix and the tensor that owns the data
template <typename T>
struct TensorMatrixView {
    arma::Mat<T> matrix;
    torch::Tensor tensor_owner; // Only populated when copy_mem is false
    
    // Allow easy conversion to just the matrix
    explicit operator arma::Mat<T>&() { return matrix; }
    explicit operator const arma::Mat<T>&() const { return matrix; }
};

/**
 * @brief Convert a 2D tensor to an Armadillo matrix.
 *
 * This function will detach the tensor (cpu, contiguous) and then convert it to an
 * Armadillo matrix.
 * @tparam T The type of the data.
 * @param tensor The tensor to convert.
 * @param copy_mem Whether to copy the memory.
 * @param transpose Whether to transpose the matrix.
 * @return The Armadillo matrix.
 */
template <typename T>
TensorMatrixView<T> conv_2d_tensor_to_arma(const torch::Tensor &tensor, bool copy_mem,
                                    bool transpose);

/**
 * @brief Set the number of threads for global operations.
 *
 * This function sets the number of threads for both Torch and OpenMP.
 *
 * @param threads The number of threads to set.
 */
inline void set_global_threads_count(int threads) {
    
    // Torch
    torch::set_num_threads(threads);
    torch::set_num_interop_threads(threads);

    // OpenMP
    #ifdef USE_OPENMP
        omp_set_num_threads(threads);
    #endif
}

/**
 * @brief Print an exception and its nested exceptions with better formatting.
 *
 * @param e The exception to print.
 * @param level The nesting level (used for indentation).
 * @param prefix Optional custom prefix for the message.
 */
 inline void print_exception(const std::exception& e, int level = 0, const std::string& prefix = "") {
    // Prepare symbols for better visualization
    const std::string indent(level * 2, ' ');
    const std::string arrow = level > 0 ? "└─ " : "";
    
    // Print the exception with better formatting
    std::cerr << indent << arrow;
    if (!prefix.empty()) {
        std::cerr << prefix << ": ";
    }
    
    // Just print the exception message without the complicated type info
    std::cerr << e.what() << std::endl;

    try {
        // Re-throw to check for nested exceptions
        std::rethrow_if_nested(e);
    } catch (const std::exception& nested) {
        // Recursively print nested exceptions
        print_exception(nested, level + 1, "Caused by");
    } catch (...) {
        // Handle non-standard exceptions
        std::cerr << indent << "  └─ Unknown exception" << std::endl;
    }
}

} // namespace utils

#include "utils.tpp"