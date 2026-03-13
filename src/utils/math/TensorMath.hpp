#pragma once

#include <cassert>
#include <cmath>
#include <torch/torch.h>

/**
 * @brief Namespace containing mathematical utility functions for tensor operations.
 *
 * These functions are stateless and perform common mathematical operations on tensors.
 */
namespace math {

/**
 * @brief Compute safe division ratio (returns default value if denominator is too small).
 *
 * @tparam T The scalar type for the result and parameters (must be floating-point)
 * @param numerator Numerator value
 * @param denominator Denominator value
 * @param epsilon Threshold for considering denominator as zero (default: 1e-8)
 * @param default_value Value to return if denominator is too small (default: 0.0)
 * @return numerator / denominator if denominator > epsilon, else default_value
 */
template <typename T>
    requires std::is_floating_point_v<T>
inline T safe_ratio(T numerator, T denominator, T epsilon = static_cast<T>(1e-8),
                    T default_value = static_cast<T>(0.0)) {
    return (denominator > epsilon) ? numerator / denominator : default_value;
}

/**
 * @brief Compute dot product between two tensors.
 *
 * @tparam T The scalar type for the result
 * @param a First tensor (must be 1D, on CPU)
 * @param b Second tensor (must be 1D, on CPU, same shape as a)
 * @return Dot product as scalar of type T
 */
template <typename T>
inline T dot_product(const torch::Tensor& a, const torch::Tensor& b) {
    assert(a.device().is_cpu() && "Tensor 'a' must be on CPU");
    assert(b.device().is_cpu() && "Tensor 'b' must be on CPU");
    assert(a.dim() == 1 && "Tensor 'a' must be 1-dimensional");
    assert(b.dim() == 1 && "Tensor 'b' must be 1-dimensional");
    assert(a.size(0) == b.size(0) && "Tensors must have the same size");
    return torch::dot(a, b).template item<T>();
}

/**
 * @brief Compute cosine similarity between two tensors.
 *
 * Cosine similarity = (a · b) / (||a|| * ||b||)
 * Returns 0 if either norm is too small (< 1e-8).
 *
 * @tparam T The scalar type for the result
 * @param a First tensor (must be 1D, on CPU)
 * @param b Second tensor (must be 1D, on CPU, same shape as a)
 * @return Cosine similarity in range [-1, 1], or 0 if norms are too small
 */
template <typename T>
inline T cosine_similarity(const torch::Tensor& a, const torch::Tensor& b) {
    assert(a.device().is_cpu() && "Tensor 'a' must be on CPU");
    assert(b.device().is_cpu() && "Tensor 'b' must be on CPU");
    assert(a.dim() == 1 && "Tensor 'a' must be 1-dimensional");
    assert(b.dim() == 1 && "Tensor 'b' must be 1-dimensional");
    assert(a.size(0) == b.size(0) && "Tensors must have the same size");

    T norm_a = a.norm().template item<T>();
    T norm_b = b.norm().template item<T>();
    T dot = torch::dot(a, b).template item<T>();

    return safe_ratio(dot, norm_a * norm_b);
}

/**
 * @brief Compute L1 (Manhattan) distance between two tensors.
 *
 * L1 distance = sum(|a - b|)
 *
 * @tparam T The scalar type for the result
 * @param a First tensor (must be on CPU)
 * @param b Second tensor (must be on CPU, same shape as a)
 * @return L1 distance (always non-negative)
 */
template <typename T>
inline T l1_distance(const torch::Tensor& a, const torch::Tensor& b) {
    assert(a.device().is_cpu() && "Tensor 'a' must be on CPU");
    assert(b.device().is_cpu() && "Tensor 'b' must be on CPU");
    assert(a.sizes() == b.sizes() && "Tensors must have the same shape");
    return (a - b).abs().sum().template item<T>();
}

/**
 * @brief Compute L2 (Euclidean) distance between two tensors.
 *
 * L2 distance = sqrt(sum((a - b)^2))
 *
 * @tparam T The scalar type for the result
 * @param a First tensor (must be on CPU)
 * @param b Second tensor (must be on CPU, same shape as a)
 * @return L2 distance (always non-negative)
 */
template <typename T>
inline T l2_distance(const torch::Tensor& a, const torch::Tensor& b) {
    assert(a.device().is_cpu() && "Tensor 'a' must be on CPU");
    assert(b.device().is_cpu() && "Tensor 'b' must be on CPU");
    assert(a.sizes() == b.sizes() && "Tensors must have the same shape");
    return torch::norm(a - b, 2).template item<T>();
}

/**
 * @brief Compute element-wise Hadamard product and return its sum.
 *
 * @tparam T The scalar type for the result
 * @param a First tensor (must be on CPU)
 * @param b Second tensor (must be on CPU, same shape as a)
 * @return Sum of element-wise products
 */
template <typename T>
inline T hadamard_sum(const torch::Tensor& a, const torch::Tensor& b) {
    assert(a.device().is_cpu() && "Tensor 'a' must be on CPU");
    assert(b.device().is_cpu() && "Tensor 'b' must be on CPU");
    assert(a.sizes() == b.sizes() && "Tensors must have the same shape");
    return (a * b).sum().template item<T>();
}

/**
 * @brief Compute element-wise Hadamard product and return its mean.
 *
 * @tparam T The scalar type for the result
 * @param a First tensor (must be on CPU)
 * @param b Second tensor (must be on CPU, same shape as a)
 * @return Mean of element-wise products
 */
template <typename T>
inline T hadamard_mean(const torch::Tensor& a, const torch::Tensor& b) {
    assert(a.device().is_cpu() && "Tensor 'a' must be on CPU");
    assert(b.device().is_cpu() && "Tensor 'b' must be on CPU");
    assert(a.sizes() == b.sizes() && "Tensors must have the same shape");
    return (a * b).mean().template item<T>();
}

/**
 * @brief Compute tensor norm (L2 norm).
 *
 * @tparam T The scalar type for the result
 * @param tensor Input tensor (must be on CPU)
 * @return L2 norm of the tensor
 */
template <typename T>
inline T tensor_norm(const torch::Tensor& tensor) {
    assert(tensor.device().is_cpu() && "Tensor must be on CPU");
    return tensor.norm().template item<T>();
}

/**
 * @brief Apply z-score normalization to a value.
 *
 * Z-score = (value - mean) / std
 *
 * @tparam T The scalar type for the result and parameters (must be floating-point)
 * @param value Value to normalize
 * @param mean Distribution mean
 * @param std Distribution standard deviation
 * @return Z-score normalized value
 */
template <typename T>
    requires std::is_floating_point_v<T>
inline T apply_z_score(T value, T mean, T std) {
    return safe_ratio(value - mean, std);
}

/**
 * @brief Get percentile rank of a value within a sorted distribution.
 *
 * @tparam T The scalar type for the result (must be floating-point)
 * @tparam U The type of values in the distribution
 * @param value Value to rank
 * @param sorted_values Pre-sorted distribution (ascending order)
 * @return Percentile rank in range [0, 1]
 */
template <typename T, typename U = double>
    requires std::is_floating_point_v<T> && std::totally_ordered<U>
inline T get_percentile_rank(U value, const std::vector<U>& sorted_values) {
    if (sorted_values.empty()) {
        return static_cast<T>(0);
    }

    auto it = std::lower_bound(sorted_values.begin(), sorted_values.end(), value);
    std::size_t pos = it - sorted_values.begin();
    return static_cast<T>(pos) / static_cast<T>(sorted_values.size());
}

} // namespace math
