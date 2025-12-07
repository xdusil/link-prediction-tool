#pragma once

#include <torch/torch.h>

/**
 * @brief Wrapper providing directional embeddings for feature generation.
 *
 * For a directed pair (source -> destination):
 * - source_embedding[source_idx]: embedding for the source/caller/client role
 * - destination_embedding[dest_idx]: embedding for the destination/callee/server role
 *
 */
class DirectionalEmbeddings {
public:
    /**
     * @brief Construct from source and destination embedding modules.
     *
     * @param source_emb Source embeddings (for nodes acting as callers/clients).
     * @param dest_emb Destination embeddings (for nodes acting as callees/servers).
     */
    DirectionalEmbeddings(torch::nn::Embedding source_emb, torch::nn::Embedding dest_emb)
        : m_source_embeddings(source_emb), m_destination_embeddings(dest_emb) {}

    /**
     * @brief Get source embedding for given indices.
     *
     * @param indices Vertex indices.
     * @return Embeddings tensor [N, embedding_dim].
     */
    const torch::Tensor source(const torch::Tensor& indices) {
        return m_source_embeddings(indices);
    }

    /**
     * @brief Get destination embedding for given indices.
     *
     * @param indices Vertex indices.
     * @return Embeddings tensor [N, embedding_dim].
     */
    const torch::Tensor destination(const torch::Tensor& indices) {
        return m_destination_embeddings(indices);
    }

    /**
     * @brief Get embedding dimension.
     * @return Embedding dimension.
     */
    int64_t embedding_dim() const { return m_source_embeddings->options.embedding_dim(); }

private:
    torch::nn::Embedding m_source_embeddings;
    torch::nn::Embedding m_destination_embeddings;
};
