#include "SkipGramModel.hpp"

SkipGramModel::SkipGramModel(int64_t vocab_size, int64_t embedding_dim)
    : m_vocab_size(vocab_size), m_embedding_dim(embedding_dim),
      m_embeddings(register_module("embeddings",
                                   torch::nn::Embedding(vocab_size, embedding_dim))) {}

torch::Tensor SkipGramModel::forward(const SkipGramInput &input) {
    // context, positive  :  [B]
    // negatives          :  [B, k]

    auto ctx = m_embeddings(input.context);   // [B, E]
    auto pos = m_embeddings(input.positive);  // [B, E]
    auto neg = m_embeddings(input.negatives); // [B, k, E]

    // positive dot products  → [B,1]
    auto pos_scores = (ctx * pos).sum(/*dim=*/-1, /*keepdim=*/true);

    // negative dot products  → [B,k]
    auto neg_scores = (neg * ctx.unsqueeze(1)) // broadcast ctx
                          .sum(/*dim=*/-1);

    return torch::cat({pos_scores, neg_scores}, -1); // [B, 1+k]
}

torch::Tensor SkipGramModel::loss(const torch::Tensor &scores,
                                  const SkipGramInput & /*input*/) {
    auto pos_scores = scores.slice(/*dim=*/-1, 0, 1); // [B,1]
    auto neg_scores = scores.slice(/*dim=*/-1, 1);    // [B,k]

    auto pos_loss = torch::log_sigmoid(pos_scores).neg();  // -log σ
    auto neg_loss = torch::log_sigmoid(-neg_scores).neg(); // -log σ(-·)

    return (pos_loss + neg_loss).mean();
}

// Returns the model's embeddings
torch::nn::Embedding &SkipGramModel::get_embeddings() { return m_embeddings; }

// Returns the model's parameters
std::vector<torch::Tensor> SkipGramModel::get_parameters() {
    return m_embeddings->parameters();
}

// Saves the model to a file
void SkipGramModel::save(const std::string &path) const {
    torch::save(m_embeddings, path);
}

// Loads the model from a file
void SkipGramModel::load(const std::string &path) { torch::load(m_embeddings, path); }