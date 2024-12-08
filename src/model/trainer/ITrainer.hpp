#pragma once
#include "../IModel.hpp"

/**
 * @brief Interface for a trainer.
 */
class ITrainer {
public:
    virtual ~ITrainer() = default;

    /**
     * @brief Train the model for a given number of epochs.
     *
     * @param num_epochs The number of epochs to train the model.
     */
    virtual void train(int num_epochs) = 0;
};
