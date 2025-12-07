#include "DirectionalSkipGramModel.hpp"
#include <torch/serialize.h>

DirectionalSkipGramModel::DirectionalSkipGramModel(int64_t num_vertices,
                                                   int64_t embedding_dim)
    : m_num_vertices(num_vertices), m_embedding_dim(embedding_dim) {

    m_source_embeddings = register_module(
        "source_embeddings", torch::nn::Embedding(num_vertices, embedding_dim));
    m_destination_embeddings = register_module(
        "destination_embeddings", torch::nn::Embedding(num_vertices, embedding_dim));
}

torch::Tensor DirectionalSkipGramModel::forward(const DirectionalTrainingBatch& batch) {
    // batch.sources      : source node indices      [B]
    // batch.destinations : destination node indices [B]
    // batch.negatives    : negative samples         [B, k]

    // Source embeddings for source nodes
    auto src_emb = m_source_embeddings(batch.sources); // [B, E]

    // Destination embeddings for positive targets
    auto dst_pos_emb = m_destination_embeddings(batch.destinations); // [B, E]

    // Destination embeddings for negative samples
    auto dst_neg_emb = m_destination_embeddings(batch.negatives); // [B, k, E]

    // Positive scores: dot(src[u], dst[v]) -> [B, 1]
    auto pos_scores = (src_emb * dst_pos_emb).sum(/*dim=*/-1, /*keepdim=*/true);

    // Negative scores: dot(src[u], dst[neg_i]) -> [B, k]
    auto neg_scores = (dst_neg_emb * src_emb.unsqueeze(1)).sum(/*dim=*/-1);

    // Concatenate: [B, 1+k]
    return torch::cat({pos_scores, neg_scores}, /*dim=*/-1);
}

torch::Tensor DirectionalSkipGramModel::loss(const torch::Tensor& scores,
                                             const DirectionalTrainingBatch& /*input*/) {
    // SGNS loss: maximize log σ(pos) + log σ(-neg)
    // Equivalent to minimizing -log σ(pos) - log σ(-neg)

    auto pos_scores = scores.slice(/*dim=*/-1, /*start=*/0, /*end=*/1).squeeze(-1); // [B]
    auto neg_scores = scores.slice(/*dim=*/-1, /*start=*/1);                // [B,k]

    auto pos_loss = -torch::log_sigmoid(pos_scores);          // [B]
    auto neg_loss = -torch::log_sigmoid(-neg_scores).sum(/*dim=*/-1); // [B]

    return (pos_loss + neg_loss).mean(); // scalar
}

void DirectionalSkipGramModel::save(const std::string& path) const {
    torch::serialize::OutputArchive archive;

    archive.write("num_vertices", torch::tensor(m_num_vertices));
    archive.write("embedding_dim", torch::tensor(m_embedding_dim));

    m_source_embeddings->save(archive);
    m_destination_embeddings->save(archive);

    archive.save_to(path);
}

void DirectionalSkipGramModel::load(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    torch::Tensor num_v_tensor, dim_tensor;
    archive.read("num_vertices", num_v_tensor);
    archive.read("embedding_dim", dim_tensor);

    int64_t loaded_num_v = num_v_tensor.item<int64_t>();
    int64_t loaded_dim = dim_tensor.item<int64_t>();

    if (loaded_num_v != m_num_vertices || loaded_dim != m_embedding_dim) {
        throw std::runtime_error(
            "Model dimension mismatch: expected num_vertices=" +
            std::to_string(m_num_vertices) +
            ", embedding_dim=" + std::to_string(m_embedding_dim) +
            " but loaded num_vertices=" + std::to_string(loaded_num_v) +
            ", embedding_dim=" + std::to_string(loaded_dim));
    }

    m_source_embeddings->load(archive);
    m_destination_embeddings->load(archive);
}
