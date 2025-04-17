#pragma once
#include <string>

/**
 * @brief Generic interface for machine learning models.

 * @tparam TInput Type of input data
 * @tparam TOutput Type of output predictions
 * @tparam TLoss Type of loss value
 * @tparam TEmbedding Type of embeddings
 * @tparam TParameters Type of model parameters
 */
template <typename TInput, typename TOutput, typename TLoss, typename TEmbedding,
          typename TParameters>
class IModel {
public:
    virtual ~IModel() = default;

    /**
     * @brief Forward method: Processes input data and produces predictions.
     *
     * @param input The input data.
     * @return The output predictions.
     */
    virtual TOutput forward(const TInput &input) = 0;

    /**
     * @brief Computes the loss for the model.
     *
     * @param predictions The model's predictions.
     * @param input The input data.
     * @return The loss value.
     */
    virtual TLoss loss(const TOutput &predictions, const TInput &input) = 0;

    /**
     * @brief Returns the model's embeddings.
     *
     * @return The model's embeddings.
     */
    virtual TEmbedding &get_embeddings() = 0;

    /**
     * @brief Returns the model's parameters.
     *
     * @return The model's parameters.
     */
    virtual TParameters get_parameters() = 0;

    /**
     * @brief Saves the model to a file.
     *
     * @param path The file path to save the model.
     */
    virtual void save(const std::string &path) const = 0;

    /**
     * @brief Loads the model from a file.
     *
     * @param path The file path to load the model from.
     */
    virtual void load(const std::string &path) = 0;
};