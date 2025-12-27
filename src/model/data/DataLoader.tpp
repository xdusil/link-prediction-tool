#pragma once
#include "DataLoader.hpp"
#include "exceptions/exceptions.hpp"
#include <iostream>
#include <sstream>
#include <tuple>

template <typename T>
DataLoader<T>::DataLoader(IRandomWalkManager<T>& rw_manager,
                          IContextGenerator<T>& context_generator,
                          IDependencyGenerator<T>& dependency_generator,
                          std::optional<std::size_t> batch_size /*= std::nullopt*/,
                          bool verbose /*= false*/)
    : m_rw_manager(rw_manager), m_context_generator(context_generator),
      m_dependency_generator(dependency_generator), m_batch_size(batch_size),
      m_current_batch_idx(0), m_total_batches(0), m_verbose(verbose) {
    if (batch_size.has_value() && *batch_size == 0) {
        throw DataLoaderException("Batch size cannot be zero.");
    }
}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DataLoader<T>::next_batch() {
    if (!has_next()) {
        throw DataLoaderException("No more batches available. Call reset() first.");
    }

    // Calculate batch range for contexts
    std::size_t effective_batch_size = m_batch_size.value_or(m_contexts.size());
    std::size_t start_idx = m_current_batch_idx * effective_batch_size;
    std::size_t end_idx = std::min(start_idx + effective_batch_size, m_contexts.size());

    // Extract contexts for this batch
    std::vector<std::vector<T>> batch_contexts(m_contexts.begin() + start_idx,
                                               m_contexts.begin() + end_idx);

    // Generate dependencies from this batch of contexts
    auto dependencies = m_dependency_generator.generate_dependencies(batch_contexts);

    ++m_current_batch_idx;
    return dependencies;
}

template <typename T>
bool DataLoader<T>::has_next() const {
    return m_current_batch_idx < m_total_batches;
}

template <typename T>
void DataLoader<T>::reset() {
    // Generate all random walks for this epoch
    auto walks = m_rw_manager.generate_random_walks();

    if (m_verbose) {
        double mean_walk_length = 0.0;
        for (const auto& walk : walks) {
            mean_walk_length += static_cast<double>(walk.size());
        }
        if (!walks.empty()) {
            mean_walk_length /= static_cast<double>(walks.size());
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << mean_walk_length;
        std::cout << "    Walks: " << walks.size() << " (avg len: " << oss.str() << ")";
    }

    // Generate all contexts from walks
    m_contexts = m_context_generator.generate_contexts(walks);

    // Calculate batching
    if (m_contexts.empty()) {
        m_total_batches = 0;
    } else if (!m_batch_size.has_value() || *m_batch_size >= m_contexts.size()) {
        m_total_batches = 1;
    } else {
        m_total_batches = (m_contexts.size() + *m_batch_size - 1) / *m_batch_size;
    }

    if (m_verbose) {
        std::cout << ", Contexts: " << m_contexts.size()
                  << ", Batches: " << m_total_batches << std::endl;
    }

    m_current_batch_idx = 0;
}
