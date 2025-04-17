#pragma once

#include "IOptimizer.hpp"
#include <memory>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>

/**
 * @brief Optimizer for training models.
 */
class Optimizer : public IOptimizer {
public:

    /**
     * @brief Constructor for Optimizer.
     *
     * @param model The model to optimize.
     * @param learning_rate The learning rate for the optimizer.
     */
    Optimizer(auto& model, double learning_rate);
    
    /**
     * @brief Zero the gradients of the optimizer.
     */
    inline void zero_grad() override;

    /**
     * @brief Perform a single optimization step.
     */
    inline void step() override;

private:
    std::unique_ptr<torch::optim::Optimizer> m_optimizer; // Optimizer object
};

#include "Optimizer.tpp"