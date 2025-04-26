#pragma once

#include "Types.hpp"
#include <armadillo>
#include <torch/torch.h>
#include <tuple>

/**
 * @brief Interface for generating features/labels for graph vertices.
 *
 * @tparam Vertex The type of the vertex.
 * @tparam GroundTruthDependencies The type of the ground truth dependencies.
*/
template <typename Vertex, typename, typename GroundTruthDependencies>
class IFeatureGenerator {
public:
    virtual ~IFeatureGenerator() = default;

    /**
     * @brief Generate feature tensors with corresponding labels.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of the features and labels.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *
     * @note Since features are symmetric, each pair is processed once (avoiding duplicates).
     */
    virtual std::tuple<torch::Tensor, arma::Row<size_t>> generate_labeled_features(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) = 0;

    /**
     * @brief Generate feature tensors with corresponding labels and vertex pairs.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of features, labels, and vertex pairs.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since features are symmetric, each pair is processed once (avoiding duplicates).
     */
    virtual std::tuple<torch::Tensor, arma::Row<size_t>, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_labeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) = 0;

    /**
     * @brief Generate feature tensors for prediction with corresponding vertex pairs.
     *
     * @param vertex_to_index The map of vertex to index.
     * @return The tuple of features and vertex pairs.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since features are symmetric, each pair is processed once (avoiding duplicates).
     */
     virtual std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>> 
     generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index) = 0;
};