#pragma once

#include "Types.hpp"

#include <armadillo>
#include <torch/torch.h>
#include <tuple>
#include <vector>

/**
 * @brief Interface for generating embeddings.
 *
 * @tparam Vertex The type of the vertex.
 * @tparam EmbeddingModule The type of the embedding module.
 * @tparam GroundTruthDependencies The type of the ground truth dependencies.
 */
template <typename Vertex, typename EmbeddingModule, typename GroundTruthDependencies>
class IEmbeddingGenerator {
public:
    virtual ~IEmbeddingGenerator() = default;

    /**
     * @brief Generate dependency embeddings.
     *
     * @param vertex_to_index The map of vertex to index.
     * @return The tensor of dependency embeddings.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     */
    virtual torch::Tensor generate_dependency_embeddings(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        EmbeddingModule &embedding_module) = 0;

    /**
     * @brief Generate dependency embeddings and labels.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth The ground truth dependencies.
     * @return The tuple of the dependency embeddings and labels.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - arma_labels: row vector of size num_pairs
     */
    virtual std::tuple<torch::Tensor, arma::Row<size_t>>
    generate_dependency_embeddings_and_labels(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies,
        EmbeddingModule &embedding_module) = 0;

    /**
     * @brief Generate dependency embeddings and vertex pairs.
     *
     * The pairs in the vertex_pairs correspond to the rows in the combined tensor.
     *
     * @param vertex_to_index The map of vertex to index.
     * @return The tuple of the dependency embeddings and vertex pairs.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - vertex_pairs: vector of IP address pairs
     */
    virtual std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_dependency_embeddings_and_vertex_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        EmbeddingModule &embedding_module) = 0;
    
    /**
     * @brief Generate dependency embeddings and labels.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth The ground truth dependencies.
     * @param embedding_module The embedding module.
     * @return The tuple of the dependency embeddings and labels.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - arma_labels: row vector of size num_pairs
     *        - vertex_pairs: vector of IP address pairs
     */
    virtual std::tuple<torch::Tensor, arma::Row<size_t>, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_dependency_embeddings_labels_and_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies,
        EmbeddingModule &embedding_module) = 0;
};