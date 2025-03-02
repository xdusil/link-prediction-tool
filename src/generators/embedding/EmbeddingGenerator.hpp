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

    /**
    * @brief Generate dependency embeddings and vertex pairs.
    *
    * @param vertex_to_index The map of vertex to index.
    * @param dependencies The dependencies.
    * @return The tuple of the dependency embeddings and vertex pairs.
    *        - combined: tensor of shape [num_pairs, embedding_dim]
    *        - arma_vertex_pairs: row vector of IP address pairs
    */
    std::tuple<torch::Tensor, arma::Row<std::pair<IPAddress, IPAddress>>>
    generate_dependency_embeddings_and_vertex_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies,
        EmbeddingModule &embedding_module) override;

private:
    /**
     * @brief Generate dependency embeddings, and optionally labels and vertex pairs.
     *
     * @tparam WithLabels Whether to include labels in the result.
     * @tparam WithVertexPairs Whether to include vertex pairs in the result.
     * @param vertex_to_index The map of vertex to index.
     * @param dependencies The dependencies.
     * @return The tuple of the dependency embeddings and labels.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - arma_labels: row vector of size num_pairs if WithLabels is true, else a
     *                       single value
     *        - arma_vertex_pairs: row vector of IP address pairs if WithVertexPairs is true,
     *                             else a single value
     */
    template <bool WithLabels, bool WithVertexPairs>
    std::tuple<torch::Tensor, arma::Row<size_t>, arma::Row<std::pair<IPAddress, IPAddress>>>
    generate_dependency_embeddings_and_labels_impl(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies, EmbeddingModule &embedding_module);

/**
 * @brief Create vertex pairs and labels.
 *
 * This function creates all possible vertex pairs and labels for the given vertex map.
 * Returned tensors are in the same order as the vertex pairs.
 * - tensor1[i] and tensor2[i] are the vertex pairs.
 *
 * @tparam WithLabels Whether to include labels in the result.
 * @tparam WithVertexPairs Whether to include vertex pairs in the result.
 * @param vertex_to_index The map of vertex to index.
 * @param ground_truth_dependencies The ground truth dependencies.
 * @return The tuple of the vertex pairs and labels:
 *         - all_v1: tensor of shape [num_pairs] of int64
 *         - all_v2: tensor of shape [num_pairs] of int64
 *         - arma_labels: row vector of size num_pairs if WithLabels is true, else a
 *                        single value
 *         - arma_vertex_pairs: row vector of IP address pairs if WithVertexPairs is true,
 *                        else a single value
 */
 template <bool WithLabels = true, bool WithVertexPairs = false>
 std::tuple<torch::Tensor, torch::Tensor, arma::Row<size_t>, arma::Row<std::pair<IPAddress, IPAddress>>>
 create_vertex_pairs_and_labels(
     const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
     const Dependencies &ground_truth_dependencies = {});
};

#include "EmbeddingGenerator.tpp"