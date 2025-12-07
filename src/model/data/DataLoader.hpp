#pragma once
#include "../../generators/context/IContextGenerator.hpp"
#include "../../generators/dependency/IDependencyGenerator.hpp"
#include "../../random_walk/manager/IRandomWalkManager.hpp"
#include "IDataLoader.hpp"
#include <torch/torch.h>
#include <tuple>

/**
 * @brief Data loader for generating batches of data.
 *
 * This class generates batches of data by performing the following steps:
 * 1. Generate random walks using a RandomWalkManager @see IRandomWalkManager.
 * 2. Generate contexts from the random walks using a ContextGenerator @see
 * IContextGenerator.
 * 3. Generate dependencies from the contexts using a DependencyGenerator @see
 * IDependencyGenerator.
 *
 * @tparam T The type of elements in the random walks.
 */
template <typename T>
class DataLoader
    : public IDataLoader<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> {
public:
    /**
     * @brief Constructs a new DataLoader object.
     *
     * @param rw_manager The RandomWalkManager used to generate random walks.
     * @param context_generator The ContextGenerator used to create contexts.
     * @param dependency_generator The DependencyGenerator for generating dependencies.
     */
    DataLoader(IRandomWalkManager<T>& rw_manager, IContextGenerator<T>& context_generator,
               IDependencyGenerator<T>& dependency_generator);

    /**
     * @brief Get the next batch of data.
     *
     * @return The next batch of data.
     *         A tuple containing:
     *         - Context tensor
     *         - Positive target tensor
     *         - 2D tensor of negative samples
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> next_batch() override;

    /**
     * @brief Check if there are more batches in current epoch.
     *
     * DataLoader generates one batch per epoch (all walks at once).
     *
     * @return true if batch hasn't been consumed yet in this epoch.
     */
    bool has_next() const override;

    /**
     * @brief Reset for new epoch.
     */
    void reset() override;

private:
    IRandomWalkManager<T>& m_rw_manager;       // RWManager used to generate random walks
    IContextGenerator<T>& m_context_generator; // ContextGenerator used to create contexts
    IDependencyGenerator<T>&
        m_dependency_generator;   // DependencyGenerator for generating dependencies
    bool m_batch_consumed = true; // Track if current epoch's batch was consumed
};

#include "DataLoader.tpp"