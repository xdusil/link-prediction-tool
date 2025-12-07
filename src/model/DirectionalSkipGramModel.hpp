#pragma once
#include "DirectionalEmbeddings.hpp"
#include "model/IModel.hpp"
#include <string>
#include <torch/torch.h>
#include <vector>

/**
 * @brief Input structure for directional skip-gram training.
 *
 * Represents directed pairs: sources[i] -> destinations[i]
 * where sources contacted destinations.
 */
struct DirectionalTrainingBatch {
    torch::Tensor sources;      //  Source node indices [batch_size]
    torch::Tensor destinations; // Destination node indices [batch_size]
    torch::Tensor negatives; //  Negative destination samples [batch_size, num_negatives]

    DirectionalTrainingBatch() = default;
    DirectionalTrainingBatch(const torch::Tensor& src, const torch::Tensor& dst,
                             const torch::Tensor& neg)
        : sources(src), destinations(dst), negatives(neg) {}

    /**
     * @brief Conversion from tuple to DirectionalTrainingBatch.
     *
     * @param batch Tuple of (sources, destinations, negatives).
     */
    DirectionalTrainingBatch(
        const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& batch)
        : sources(std::get<0>(batch)), destinations(std::get<1>(batch)),
          negatives(std::get<2>(batch)) {}
};

/**
 * @brief Directional Skip-Gram model with dual embeddings (NERD-style).
 *
 * Each node v has two separate embedding vectors:
 * - source_emb[v]: used when v acts as a source/caller/client (initiates connection)
 * - dest_emb[v]: used when v acts as a destination/callee/server (receives connection)
 *
 * For a directed edge (u -> v) meaning "u contacted v" / "u depends on v":
 * - Training maximizes: dot(source_emb[u], dest_emb[v])
 * - This learns asymmetric roles in client-server relationships
 *
 * Training uses SGNS (Skip-Gram with Negative Sampling):
 * - Positive pairs: maximize log σ(src[u] · dst[v])
 * - Negative pairs: maximize log σ(-src[u] · dst[neg])
 */
class DirectionalSkipGramModel
    : public IModel<DirectionalTrainingBatch,  // Input type
                    torch::Tensor,             // Output type
                    torch::Tensor,             // Loss type
                    DirectionalEmbeddings,     // Embedding type
                    std::vector<torch::Tensor> // Parameters type
                    >,
      public torch::nn::Module {
public:
    /**
     * @brief Construct the model.
     *
     * @param num_vertices Number of vertices/nodes in the graph.
     * @param embedding_dim Dimension of embedding vectors.
     */
    DirectionalSkipGramModel(int64_t num_vertices, int64_t embedding_dim);

    /**
     * @brief Compute scores for a training batch.
     *
     * @param batch The training batch with sources, destinations, and negatives.
     * @return Tensor [batch_size, 1 + num_negatives] with positive score first,
     *         then negative scores.
     */
    torch::Tensor forward(const DirectionalTrainingBatch& batch) override;

    /**
     * @brief Compute Skip-Gram negative sampling loss.
     *
     * @param scores Output from forward() [batch_size, 1 + num_negatives].
     * @param input The input data. - unused
     * @return Scalar loss tensor.
     */
    torch::Tensor loss(const torch::Tensor& scores,
                       const DirectionalTrainingBatch& /*input*/) override;

    /// @brief Get source embeddings module.
    torch::nn::Embedding& source_embeddings() { return m_source_embeddings; }

    /// @brief Get destination embeddings module.
    torch::nn::Embedding& destination_embeddings() { return m_destination_embeddings; }

    /**
     * @brief Get directional embeddings wrapper for feature generation.
     * @return DirectionalEmbeddings wrapper providing both source and dest embeddings.
     */
    DirectionalEmbeddings get_directional_embeddings() {
        return DirectionalEmbeddings(m_source_embeddings, m_destination_embeddings);
    }

    /**
     * @brief Get number of vertices.
     *
     * @return Number of vertices/nodes in the graph.
     */
    int64_t num_vertices() const { return m_num_vertices; }

    /**
     * @brief Get embedding dimension.
        *
     * @return Embedding vector dimension.
     */
    int64_t embedding_dim() const { return m_embedding_dim; }

    /**
     * @brief Save model to file.
     *
     * @param path File path.
     */
    void save(const std::string& path) const override;

    /**
     * @brief Load model from file.
     *
     * @param path File path.
     */
    void load(const std::string& path) override;

private:
    torch::nn::Embedding m_source_embeddings{nullptr};
    torch::nn::Embedding m_destination_embeddings{nullptr};
    int64_t m_num_vertices;
    int64_t m_embedding_dim;
};

