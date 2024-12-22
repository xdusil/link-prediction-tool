#pragma once

#include <armadillo>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

/**
 * @brief Utility functions.
 */
namespace utils {

/**
 * @brief Choose a random item from a vector.
 *
 * @tparam T The type of the items in the vector.
 * @param items The vector of items.
 * @param rng The random number generator.
 * @return The chosen item.
 */
template <typename T>
std::optional<T> choice_random_item(std::vector<T> &items, auto &rng);

/**
 * @brief Choose a random item from a range.
 *
 * @tparam T The type of the items in the range.
 * @param it_begin The beginning of the range.
 * @param it_end The end of the range.
 * @param rng The random number generator.
 * @param distance The distance between it_begin and it_end.
 * @return The chosen item.
 */
template <typename T>
std::optional<T> choice_random_item(auto it_begin, auto it_end, auto &rng, std::optional<std::size_t> distance = {});

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
} // namespace utils

#include "utils.tpp"