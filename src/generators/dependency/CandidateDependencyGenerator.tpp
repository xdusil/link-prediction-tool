#pragma once

#include "CandidateDependencyGenerator.hpp"
#include <iostream>

template <typename T>
CandidateDependencyGenerator<T>::CandidateDependencyGenerator(
    std::function<T(const std::vector<T> &)> initial_selector, std::size_t total_vertices,
    std::function<long(T)> to_long, int num_negative_samples)
    : initial_selector(std::move(initial_selector)), total_vertices(total_vertices),
      to_long(std::move(to_long)), num_negative_samples(num_negative_samples),
      rng(std::random_device{}()) {}


template <typename T>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CandidateDependencyGenerator<T>::generate_dependencies(
    const std::vector<std::vector<T>> &context_list) {
    // First pass to count the number of elements
    size_t num_pairs = 0;
    for (const auto &context : context_list) {
        if (context.empty()) continue;
        
        T target = initial_selector(context);
        for (const auto &item : context) {
            if (item != target) num_pairs++;
        }
    }
    
    // Create tensors directly with the right size
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt64)
                       .device(torch::kCPU)
                       .memory_format(torch::MemoryFormat::Contiguous);
    
    auto contexts_tensor = torch::empty({static_cast<long>(num_pairs)}, options);
    auto positives_tensor = torch::empty({static_cast<long>(num_pairs)}, options);
    auto negatives_tensor = torch::empty({static_cast<long>(num_pairs), 
                                          static_cast<long>(num_negative_samples)}, options);
    
    // Create accessors for direct indexing
    auto contexts_accessor = contexts_tensor.accessor<int64_t, 1>();
    auto positives_accessor = positives_tensor.accessor<int64_t, 1>();
    auto negatives_accessor = negatives_tensor.accessor<int64_t, 2>();
    
    std::uniform_int_distribution<long> dist(0, static_cast<long>(total_vertices) - 1);
    
    // Second pass to fill the tensors
    size_t idx = 0;
    for (const auto &context : context_list) {
        if (context.empty()) continue;
        
        T target = initial_selector(context);
        long target_id = to_long(target);
        
        for (size_t i = 0; i < context.size(); ++i) {
            if (context[i] == target) continue;
           
            long context_id = to_long(context[i]);
            
            contexts_accessor[idx] = context_id;
            positives_accessor[idx] = target_id;
            
            // Generate negative samples
            for (int j = 0; j < num_negative_samples; ++j) {
                long negative;
                int attempts = 0;
                const int max_attempts = 10;
                
                do {
                    negative = dist(rng);
                    attempts++;
                    if (attempts >= max_attempts) {
                        // Fallback
                        negative = (target_id + j + 1) % total_vertices;
                        break;
                    }
                } while (negative == target_id || negative == context_id);
                
                negatives_accessor[idx][j] = negative;
            }
            
            idx++;
        }
    }
    
    return {contexts_tensor, positives_tensor, negatives_tensor};
}