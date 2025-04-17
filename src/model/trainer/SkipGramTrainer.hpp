#pragma once
#include "../SkipGramModel.hpp"
#include "../data/IDataLoader.hpp"
#include "../optimizer/IOptimizer.hpp"
#include "ITrainer.hpp"
#include "model/optimizer/IOptimizer.hpp"

/**
 * @brief Trainer for SkipGram model.
 *
 * @tparam T Data type for generating training data.
 */
template <typename T>
class SkipGramTrainer : public ITrainer {
public:
    /**
     * @brief Constructor for SkipGramTrainer.
     *
     * @param model The SkipGram model to train.
     * @param data_loader The data loader for generating training data.
     * @param optimizer The optimizer to use for training.
     */
    SkipGramTrainer(auto &model, IDataLoader<T> &data_loader, IOptimizer &optimizer);

    /**
     * Train the model for a given number of epochs.
     *
     * @param num_epochs The number of epochs to train the model.
     */
    void train(int num_epochs) override;

private:
    IModel<SkipGramInput, at::Tensor, at::Tensor, torch::nn::Embedding,
           std::vector<torch::Tensor>> &m_model; // SkipGram model
    IDataLoader<T> &m_data_loader;               // Data loader
    IOptimizer &m_optimizer;                     // Optimizer
};

template class SkipGramTrainer<
    std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>>;

#include "SkipGramTrainer.tpp"