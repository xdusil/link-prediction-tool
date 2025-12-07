#pragma once

#include "GenericTrainer.hpp"

template <typename Model, typename InputBatch>
    requires TrainableModel<Model, InputBatch>
GenericTrainer<Model, InputBatch>::GenericTrainer(Model& model,
                                                  IDataLoader<InputBatch>& data_loader,
                                                  IOptimizer& optimizer, bool verbose)
    : m_model{model}, m_data_loader{data_loader}, m_optimizer{optimizer},
      m_verbose{verbose} {}

template <typename Model, typename InputBatch>
    requires TrainableModel<Model, InputBatch>
void GenericTrainer<Model, InputBatch>::train(int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Reset data loader for new epoch
        m_data_loader.reset();

        float epoch_loss = 0.0f;
        int num_batches = 0;

        while (m_data_loader.has_next()) {
            auto batch = m_data_loader.next_batch();

            // Zero gradients
            m_optimizer.zero_grad();

            // Forward pass: compute scores
            auto scores = m_model.forward(batch);

            // Compute loss
            auto loss = m_model.loss(scores);

            // Backward pass
            loss.backward();

            // Update weights
            m_optimizer.step();

            float batch_loss = loss.template item<float>();
            epoch_loss += batch_loss;
            num_batches++;

            m_last_loss = batch_loss;
        }

        if (m_verbose) {
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                epoch_end - epoch_start);

            float avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0f;

            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "]";
            if (num_batches > 0) {
                std::cout << ", Batches: " << num_batches << ", Avg Loss: " << avg_loss;
            } else {
                std::cout << " - No training data generated";
            }
            std::cout << ", Time: " << duration_ms.count() << "ms" << std::endl;
        }
    }
}
