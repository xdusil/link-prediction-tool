#pragma once

#include "utils.hpp"

namespace utils {

template <typename T>
std::optional<T> choice_random_item(std::vector<T> &items, auto &rng) {
    if (items.empty()) {
        return std::nullopt;
    }
    std::uniform_int_distribution<size_t> dist(0, items.size() - 1);
    size_t index = dist(rng);
    return items[index];
}

template <typename T>
std::tuple<arma::mat, arma::Row<T>, arma::mat, arma::Row<T>>
split_train_test(const arma::mat &features, const arma::Row<T> &labels,
                 double train_fraction)  {
    if (train_fraction < 0.0 || train_fraction > 1.0) {
        throw std::invalid_argument("train_fraction must be between 0.0 and 1.0");
    }

    arma::uword num_samples = features.n_cols;
    if (num_samples != labels.n_elem) {
        throw std::invalid_argument("Number of samples in features and labels must be the same");
    }
    arma::Row<arma::uword> indices(num_samples);

    // Initialize indices
    for (arma::uword i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    arma::Row<arma::uword> shuffled_indices = arma::shuffle(indices);

    // Determine the number of training samples
    arma::uword num_train = static_cast<arma::uword>(train_fraction * num_samples);

    // Split the indices
    arma::Row<arma::uword> train_indices = shuffled_indices.subvec(0, num_train - 1);
    arma::Row<arma::uword> test_indices =
        shuffled_indices.subvec(num_train, num_samples - 1);

    // Extract training and testing features and labels
    arma::mat train_features = features.cols(train_indices);
    arma::Row<arma::uword> train_labels = labels.cols(train_indices);

    arma::mat test_features = features.cols(test_indices);
    arma::Row<arma::uword> test_labels = labels.cols(test_indices);

    return std::make_tuple(train_features, train_labels, test_features, test_labels);
}


} // namespace utils