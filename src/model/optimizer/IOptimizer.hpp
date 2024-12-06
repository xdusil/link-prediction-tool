#pragma once

/**
 * @brief Interface for an optimizer.
 */
class IOptimizer {
public:

    virtual ~IOptimizer() = default;

    /**
     * @brief Zero the gradients of the optimizer.
     */
    virtual void zero_grad() = 0;

    /**
     * @brief Perform a single optimization step.
     */
    virtual void step() = 0;

};