#include "Optimizer.hpp"
#include <torch/optim/adam.h>

Optimizer::Optimizer(torch::nn::Module &model, double learning_rate)
    : m_optimizer(std::make_unique<torch::optim::Adam>(
          model.parameters(), torch::optim::AdamOptions(learning_rate))) {}

void Optimizer::zero_grad() { m_optimizer->zero_grad(); }

void Optimizer::step() { m_optimizer->step(); }