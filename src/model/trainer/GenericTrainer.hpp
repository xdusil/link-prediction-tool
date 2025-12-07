#pragma once
#include "../data/IDataLoader.hpp"
#include "../optimizer/IOptimizer.hpp"
#include "ITrainer.hpp"
#include <chrono>
#include <concepts>
#include <iostream>
#include <torch/torch.h>

/**
 * @brief Concept that checks if a type has a forward() method.
 *
 * @tparam M The model type to check.
 * @tparam InputBatch The batch type accepted by forward().
 */
template <typename M, typename InputBatch>
concept HasForward = requires(M model, const InputBatch& batch) {
    { model.forward(batch) } -> std::same_as<torch::Tensor>;
};

/**
 * @brief Concept that checks if a type has a loss() method.
 *
 * @tparam M The model type to check.
 */
template <typename M>
concept HasLoss = requires(M model, const torch::Tensor& scores) {
    { model.loss(scores) } -> std::same_as<torch::Tensor>;
};

/**
 * @brief Concept that checks if a type can be used as a training model.
 *
 * A valid training model must provide:
 * - forward(const InputBatch&) -> torch::Tensor
 * - loss(const torch::Tensor&) -> torch::Tensor
 *
 * These methods enable the standard training loop:
 * 1. scores = model.forward(batch)
 * 2. loss = model.loss(scores)
 * 3. loss.backward()
 *
 * @tparam M The model type to check.
 * @tparam InputBatch The batch type accepted by forward().
 */
template <typename M, typename InputBatch>
concept TrainableModel = HasForward<M, InputBatch> && HasLoss<M>;

/**
 * @brief Generic trainer for models satisfying the TrainableModel concept.
 *
 * @see TrainableModel
 * @see IDataLoader
 * @see IOptimizer
 *
 * The trainer handles the training loop, gradient zeroing, backward pass,
 * and optimizer step.
 *
 * @tparam Model The model type.
 * @tparam InputBatch The batch type returned by the data loader.
 */
template <typename Model, typename InputBatch>
    requires TrainableModel<Model, InputBatch>
class GenericTrainer : public ITrainer {
public:
    /**
     * @brief Construct generic trainer.
     *
     * @param model The model to train.
     * @param data_loader Data loader providing training batches.
     * @param optimizer The optimizer for parameter updates.
     * @param verbose Print training progress if true.
     */
    GenericTrainer(Model& model, IDataLoader<InputBatch>& data_loader,
                   IOptimizer& optimizer, bool verbose = true);

    /**
     * @brief Train the model for specified number of epochs.
     *
     * Each epoch:
     * - Reset data loader for new epoch
     * - While there are batches:
     *    1. Get batch from data loader
     *    2. Zero gradients
     *    3. Forward pass
     *    4. Compute loss
     *    5. Backward pass
     *    6. Optimizer step
     *
     * @param num_epochs Number of training epochs.
     */
    void train(int num_epochs) override;

    /**
     * @brief Get the last training loss.
     *
     * @return The last recorded loss value.
     */
    float last_loss() const { return m_last_loss; }

private:
    Model& m_model;
    IDataLoader<InputBatch>& m_data_loader;
    IOptimizer& m_optimizer;
    bool m_verbose;
    float m_last_loss = 0.0f;
};

#include "GenericTrainer.tpp"