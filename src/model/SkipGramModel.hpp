#pragma once
#include "IModel.hpp"
#include <torch/torch.h>

/**
 * @brief Input type for SkipGram model
 */
struct SkipGramInput {
    torch::Tensor context;                // Context tensor
    torch::Tensor positive;               // Positive tensor
    std::vector<torch::Tensor> negatives; // List of negative tensors
};

/**
 * @brief SkipGram model for learning embeddings
 */
class SkipGramModel : public IModel<SkipGramInput,             // Input type
                                    torch::Tensor,             // Output type
                                    torch::Tensor,             // Loss type
                                    torch::nn::Embedding,      // Embedding type
                                    std::vector<torch::Tensor> // Parameters type
                                    >,
                      public torch::nn::Module {
public:
    /**
     * @brief Constructor
     *
     * @param vocab_size Size of the vocabulary - e.g. number of vertices
     * @param embedding_dim Dimension of embeddings
     */
    SkipGramModel(int64_t vocab_size, int64_t embedding_dim);

    /**
     * @brief Forward method: Processes input data and produces predictions.
     *
     * @param input The input data.
     * @return The output predictions.
     */
    torch::Tensor forward(const SkipGramInput &input) override;

    /**
     * @brief Computes the loss for SkipGram
     *
     * @param predictions The model's predictions.
     * @param input The input data.
     * @return The loss value.
     */
    torch::Tensor loss(const torch::Tensor &predictions,
                       const SkipGramInput &input) override;

    /**
     * @brief Returns the model's embeddings.
     *
     * @return The model's embeddings.
     */
    torch::nn::Embedding &get_embeddings() override;

    /**
     * @brief Returns the model's parameters.
     *
     * @return The model's parameters.
     */
    std::vector<torch::Tensor> get_parameters() override;

    /**
     * @brief Saves the model to a file.
     *
     * @param path The file path to save the model.
     */
    void save(const std::string &path) const override;

    /**
     * @brief Loads the model from a file.
     *
     * @param path The file path to load the model from.
     */
    void load(const std::string &path) override;

private:
    torch::nn::Embedding m_embeddings; // Embedding layer
    int64_t m_vocab_size;              // Vocabulary size - e.g. number of vertices
    int64_t m_embedding_dim;           // Embedding dimension
};
