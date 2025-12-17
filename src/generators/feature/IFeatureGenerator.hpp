#pragma once

#include "Types.hpp"
#include <armadillo>
#include <concepts>
#include <torch/torch.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <type_traits>

/**
 * @brief Concept to ensure Container has contains() method.
 *
 * @tparam Container The container type to check.
 * @tparam KeyType The key type for the container.
 */
template <typename Container>
concept HasContains = requires(const std::remove_cvref_t<Container> &c,
                               typename std::remove_cvref_t<Container>::value_type v) {
    { c.contains(v) } -> std::convertible_to<bool>;
};

/**
 * @brief Interface for generating features/labels for graph vertex pairs.
 *
 * @tparam Vertex The vertex descriptor type.
 * @tparam GroundTruthDependencies The Container type for ground truth.
 */
template <typename Vertex, typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies>
class IFeatureGenerator {
public:
    virtual ~IFeatureGenerator() = default;

    /**
     * @brief Generate feature tensors with corresponding labels.
     *
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @param ground_truth_dependencies Ground truth dependency pairs.
     * @return Tuple of (features, labels):
     *         - features: tensor [num_pairs, feature_dim]
     *         - labels: row vector [num_pairs] (1=dependency, 0=no dependency)
     */
    virtual std::tuple<torch::Tensor, arma::Row<std::size_t>> generate_labeled_features(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const GroundTruthDependencies& ground_truth_dependencies) = 0;

    /**
     * @brief Generate feature tensors with labels and vertex pairs.
     *
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @param ground_truth_dependencies Ground truth dependency pairs.
     * @return Tuple of (features, labels, vertex_pairs):
     *         - features: tensor [num_pairs, feature_dim]
     *         - labels: row vector [num_pairs] (1=dependency, 0=no dependency)
     *         - vertex_pairs: vector [num_pairs] of (src_ip, dst_ip) pairs
     */
    virtual std::tuple<torch::Tensor, arma::Row<std::size_t>,
                       std::vector<std::pair<IPAddress, IPAddress>>>
    generate_labeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const GroundTruthDependencies& ground_truth_dependencies) = 0;

    /**
     * @brief Generate feature tensors for prediction (no labels).
     *
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @return Tuple of (features, vertex_pairs):
     *         - features: tensor [num_pairs, feature_dim]
     *         - vertex_pairs: vector [num_pairs] of (src_ip, dst_ip) pairs
     */
    virtual std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index) = 0;
};
