#include "SkipGramModel.hpp"

SkipGramModel::SkipGramModel(int64_t vocab_size, int64_t embedding_dim)
    : m_vocab_size(vocab_size), m_embedding_dim(embedding_dim),
      m_embeddings(register_module("embeddings",
                                   torch::nn::Embedding(vocab_size, embedding_dim))) {}

// Forward method: Processes input data and produces predictions.
torch::Tensor SkipGramModel::forward(const SkipGramInput &input) {
    // Get context word embeddings
    auto context_embeds = m_embeddings(input.context);

    // Get positive word embeddings
    auto positive_embeds = m_embeddings(input.positive);

    // Compute dot product between context and positive embeddings
    auto positive_scores =
        torch::matmul(context_embeds, positive_embeds.transpose(-2, -1));

    // Compute dot product between context and negative embeddings
    std::vector<torch::Tensor> negative_scores_list;
    for (const auto &neg : input.negatives) {
        auto negative_embeds = m_embeddings(neg);
        auto negative_scores =
            torch::matmul(context_embeds, negative_embeds.transpose(-2, -1));
        negative_scores_list.push_back(negative_scores);
    }

    // Concatenate positive and negative scores
    if (negative_scores_list.empty()) {
        return positive_scores;
    }
    auto negative_scores = torch::cat(negative_scores_list, -1);
    return torch::cat({positive_scores, negative_scores}, -1);
}

// Computes the loss for SkipGram
torch::Tensor SkipGramModel::loss(const torch::Tensor &predictions,
                                  const SkipGramInput &input) {
    // Determine the split point for positive and negative scores
    auto positive_size = input.positive.size(0);
    auto total_size = predictions.size(-1);
    auto negative_size = total_size - positive_size;

    // Split predictions into positive and negative scores
    auto positive_scores =
        predictions.slice(/*dim=*/-1, /*start=*/0, /*end=*/positive_size);
    auto negative_scores =
        predictions.slice(/*dim=*/-1, /*start=*/positive_size, /*end=*/total_size);

    // Compute positive loss
    auto positive_loss = torch::binary_cross_entropy_with_logits(
        positive_scores, torch::ones_like(positive_scores), torch::Tensor(),
        torch::Tensor(), torch::Reduction::Mean);

    // Compute negative loss
    auto negative_loss = torch::binary_cross_entropy_with_logits(
        negative_scores, torch::zeros_like(negative_scores), torch::Tensor(),
        torch::Tensor(), torch::Reduction::Mean);

    // Return total loss
    return positive_loss + negative_loss;
}

// Returns the model's embeddings
const torch::nn::Embedding &SkipGramModel::get_embeddings() const { return m_embeddings; }

// Saves the model to a file
void SkipGramModel::save(const std::string &path) const {
    torch::save(m_embeddings, path);
}

// Loads the model from a file
void SkipGramModel::load(const std::string &path) { torch::load(m_embeddings, path); }