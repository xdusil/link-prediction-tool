#pragma once

#include <optional>
#include <vector>
#include <armadillo>
#include <random>
#include <tuple>

/**
 * @brief Utility functions.
 */
namespace utils {

/**
 * @brief Choose a random item from a vector and remove it.
 *
 * @tparam T The type of the items in the vector.
 * @param items The vector of items.
 * @param rng The random number generator.
 * @return The chosen item.
 */
template <typename T>
std::optional<T> choice_random_item(std::vector<T> &items, auto &rng);

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

} // namespace utils

#include "utils.tpp"