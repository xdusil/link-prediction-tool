#pragma once

#include "IEmbeddingGenerator.hpp"

/**
 * @brief Class for generating embeddings.
 *
 * @tparam Vertex The type of the vertex.
 * @tparam Dependencies The type of the dependencies.
 * @tparam EmbeddingModule The type of the embedding module.
*/
template <typename Vertex, typename Dependencies, typename EmbeddingModule>
class EmbeddingGenerator
    : public IEmbeddingGenerator<Vertex, Dependencies, EmbeddingModule> {
public:
    /**
     * @brief Generate dependency embeddings.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param dependencies The dependencies.
     * @return The tensor of dependency embeddings.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     */
    torch::Tensor generate_dependency_embeddings(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies,
        EmbeddingModule &embedding_module) override;

    /**
     * @brief Generate dependency embeddings and labels.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param dependencies The dependencies.
     * @return The tuple of the dependency embeddings and labels.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - arma_labels: row vector of size num_pairs
     */
    std::tuple<torch::Tensor, arma::Row<size_t>>
    generate_dependency_embeddings_and_labels(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies,
        EmbeddingModule &embedding_module) override;

private:
    /**
     * @brief Generate dependency embeddings and optional labels.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param dependencies The dependencies.
     * @return The tuple of the dependency embeddings and labels.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - arma_labels: row vector of size num_pairs if CreateLabels is true, else a
     *                       single value
     */
    template <bool CreateLabels>
    std::tuple<torch::Tensor, arma::Row<size_t>>
    generate_dependency_embeddings_and_labels(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies, EmbeddingModule &embedding_module);

    /**
     * @brief Create vertex pairs and labels.
     *
     * @tparam CreateLabels Whether to create labels.
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of the vertex pairs and labels:
     *         - all_v1: tensor of shape [num_pairs] of int64
     *         - all_v2: tensor of shape [num_pairs] of int64
     *         - arma_labels: row vector of size num_pairs if CreateLabels is true, else a
     *                        single value
     */
    template <bool CreateLabels = true>
    std::tuple<torch::Tensor, torch::Tensor, arma::Row<size_t>>
    create_vertex_pairs_and_labels(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &ground_truth_dependencies = {});
};

#include "EmbeddingGenerator.tpp"