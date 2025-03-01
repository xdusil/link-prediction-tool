#pragma once

#include "utils.hpp"

namespace utils {

template <typename RandomIt, typename RNG>
std::optional<typename std::iterator_traits<RandomIt>::value_type>
choice_random_item(RandomIt it_begin, RandomIt it_end, RNG &rng,
                   std::optional<std::size_t> distance /*= std::nullopt*/) {
    if (it_begin == it_end) {
        return std::nullopt;
    }

    std::size_t dist_val = distance.value_or(std::distance(it_begin, it_end));
    if (dist_val == 0) {
        return std::nullopt;
    }

    std::uniform_int_distribution<std::size_t> dist(0, dist_val - 1);
    std::advance(it_begin, dist(rng));
    return *it_begin;
}

template <typename T, typename RNG>
std::optional<T> choice_random_item(std::vector<T> &items, RNG &rng) {
    return choice_random_item(items.begin(), items.end(), rng, items.size());
}

template <typename FeatureType, typename LabelType>
std::tuple<arma::Mat<LabelType>, arma::Row<FeatureType>, arma::Mat<LabelType>,
           arma::Row<FeatureType>>
split_train_test(const arma::Mat<LabelType> &features,
                 const arma::Row<FeatureType> &labels, double train_fraction) {
    if (train_fraction < 0.0 || train_fraction > 1.0) {
        throw std::invalid_argument("train_fraction must be between 0.0 and 1.0");
    }

    arma::uword num_samples = features.n_cols;
    if (num_samples != labels.n_elem) {
        throw std::invalid_argument(
            "Number of samples in features and labels must be the same");
    }

    // Shuffle the indices
    arma::uvec shuffled_indices =
        arma::shuffle(arma::linspace<arma::uvec>(0, num_samples - 1, num_samples));

    // Determine the number of training samples
    arma::uword num_train = static_cast<arma::uword>(train_fraction * num_samples);

    // Split the indices
    auto train_indices = shuffled_indices.subvec(0, num_train - 1);
    auto test_indices = shuffled_indices.subvec(num_train, num_samples - 1);

    // Extract training and testing features
    arma::Mat<LabelType> train_features = features.cols(train_indices);
    arma::Mat<LabelType> test_features = features.cols(test_indices);

    // Convert labels to row vectors
    arma::Row<FeatureType> train_labels =
        arma::conv_to<arma::Row<FeatureType>>::from(labels.cols(train_indices));
    arma::Row<FeatureType> test_labels =
        arma::conv_to<arma::Row<FeatureType>>::from(labels.cols(test_indices));

    return std::make_tuple(train_features, train_labels, test_features, test_labels);
}

template <typename Vertex>
bool is_vertex_pair_in_sequence(const std::vector<Vertex> &sequence, Vertex src,
                                Vertex dst) {
    for (int i = 0; i < sequence.size() - 1; ++i) {
        if (sequence[i] == src && sequence[i + 1] == dst) {
            return true;
        }
    }

    return false;
}

template <typename Vertex>
bool is_vertex_pair_in_sequence_opposite(const std::vector<Vertex> &sequence, Vertex src,
                                         Vertex dst) {
    return is_vertex_pair_in_sequence(sequence, dst, src);
}

template <typename T>
TensorMatrixView<T> conv_2d_tensor_to_arma(const torch::Tensor &tensor, bool copy_mem, bool transpose) {
    // First ensure tensor is detached, on CPU, and contiguous
    auto safe_tensor = tensor.detach().cpu().contiguous();

    // Type checking
    if constexpr (std::is_same_v<T, float>) {
        if (safe_tensor.scalar_type() != at::ScalarType::Float) {
            throw std::invalid_argument("Tensor must contain float data");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (safe_tensor.scalar_type() != at::ScalarType::Double) {
            throw std::invalid_argument("Tensor must contain double data");
        }
    } else {
        throw std::invalid_argument("Invalid type");
    }

    T *data_ptr = safe_tensor.template data_ptr<T>();

    // Create Armadillo matrix
    long dim_0 = static_cast<long>(safe_tensor.size(0));
    long dim_1 = static_cast<long>(safe_tensor.size(1));
    if (transpose) {
        std::swap(dim_0, dim_1);
    }

    arma::Mat<T> result_matrix(data_ptr, dim_0, dim_1, copy_mem);
    
    // Return both the matrix and the tensor (if not copying)
    return {
        result_matrix,
        copy_mem ? torch::Tensor() : safe_tensor // Only keep tensor alive if not copying
    };
}
} // namespace utils