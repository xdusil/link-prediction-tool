#pragma once
#include "DataLoader.hpp"
#include <tuple>

template <typename T>
DataLoader<T>::DataLoader(IRandomWalkManager<T>& rw_manager,
                          IContextGenerator<T>& context_generator,
                          IDependencyGenerator<T>& dependency_generator)
    : m_rw_manager(rw_manager), m_context_generator(context_generator),
      m_dependency_generator(dependency_generator) {}

// Generate the next batch of data
template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DataLoader<T>::next_batch() {
    // Step 1: Generate new random walks
    auto walks = m_rw_manager.generate_random_walks();

    // Step 2: Generate contexts dynamically
    auto contexts = m_context_generator.generate_contexts(walks);

    // Step 3: Generate dependencies from contexts
    auto dependencies = m_dependency_generator.generate_dependencies(contexts);

    m_batch_consumed = true;
    return dependencies;
}

template <typename T>
bool DataLoader<T>::has_next() const {
    return !m_batch_consumed;
}

template <typename T>
void DataLoader<T>::reset() {
    m_batch_consumed = false;
}
