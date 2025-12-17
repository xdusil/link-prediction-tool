#pragma once

#include "exceptions/exceptions.hpp"
#include <memory>
#include <torch/nn/modules/embedding.h>
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
        : m_source_embeddings(std::move(source_emb)),
          m_destination_embeddings(std::move(dest_emb)) {
        if (m_source_embeddings->options.embedding_dim() !=
            m_destination_embeddings->options.embedding_dim()) {
            throw EmbeddingException(
                "Source and destination embeddings must have the same dimension.");
        }
    }

    /**
     * @brief Get source embedding for given indices.
     *
     * @param indices Vertex indices [N].
     * @return Embeddings tensor [N, embedding_dim].
     */
    torch::Tensor forward_src(const torch::Tensor& indices) {
        return m_source_embeddings->forward(indices);
    }

    /**
     * @brief Get destination embedding for given indices.
     *
     * @param indices Vertex indices [N].
     * @return Embeddings tensor [N, embedding_dim].
     */
    torch::Tensor forward_dst(const torch::Tensor& indices) {
        return m_destination_embeddings->forward(indices);
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
