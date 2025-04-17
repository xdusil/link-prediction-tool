#pragma once

#include "SkipGramTrainer.hpp"
#include <iostream>

template <typename T>
SkipGramTrainer<T>::SkipGramTrainer(auto &model, IDataLoader<T> &data_loader,
                                    IOptimizer &optimizer)
    : m_model(model), m_data_loader(data_loader), m_optimizer(optimizer) {}

// Train the model for a given number of epochs.
template <typename T>
void SkipGramTrainer<T>::train(int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        SkipGramInput input;
        // Get the next batch of training data
        std::tie(input.context, input.positive, input.negatives) =
            m_data_loader.next_batch();

        // Forward pass
        auto predictions = m_model.forward(input);

        // Compute loss
        auto loss = m_model.loss(predictions, input);

        // Zero gradients
        m_optimizer.zero_grad();

        // Backward pass
        loss.backward();

        // Update weights
        m_optimizer.step();

        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                  << "], Loss: " << loss.item<float>() << std::endl;
    }
}
