#pragma once

#include "Types.hpp"

#include <armadillo>
#include <torch/torch.h>
#include <tuple>

/**
 * @brief Interface for generating embeddings.
 *
 * @tparam Vertex The type of the vertex.
 * @tparam Dependencies The type of the dependencies.
 * @tparam EmbeddingModule The type of the embedding module.
 */
template <typename Vertex, typename Dependencies, typename EmbeddingModule>
class IEmbeddingGenerator {
public:
    virtual ~IEmbeddingGenerator() = default;

    /**
     * @brief Generate dependency embeddings.
     *
     * @tparam Vertex The type of the vertex.
     * @tparam Dependencies The type of the dependencies.
     * @param vertex_to_index The map of vertex to index.
     * @param dependencies The dependencies.
     * @return The tensor of dependency embeddings.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     */
    virtual torch::Tensor generate_dependency_embeddings(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies, EmbeddingModule &embedding_module) = 0;

    /**
     * @brief Generate dependency embeddings and labels.
     *
     * @tparam Vertex The type of the vertex.
     * @tparam Dependencies The type of the dependencies.
     * @param vertex_to_index The map of vertex to index.
     * @param dependencies The dependencies.
     * @return The tuple of the dependency embeddings and labels.
     *        - combined: tensor of shape [num_pairs, embedding_dim]
     *        - arma_labels: row vector of size num_pairs
     */
    virtual std::tuple<torch::Tensor, arma::Row<size_t>>
    generate_dependency_embeddings_and_labels(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies, EmbeddingModule &embedding_module) = 0;
    
    /**
    * @brief Generate dependency embeddings and vertex pairs.
    *
    * @param vertex_to_index The map of vertex to index.
    * @param dependencies The dependencies.
    * @return The tuple of the dependency embeddings and vertex pairs.
    *        - combined: tensor of shape [num_pairs, embedding_dim]
    *        - arma_vertex_pairs: row vector of IP address pairs
    */
    virtual std::tuple<torch::Tensor, arma::Row<std::pair<IPAddress, IPAddress>>>
    generate_dependency_embeddings_and_vertex_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const Dependencies &dependencies,
        EmbeddingModule &embedding_module) = 0;
};