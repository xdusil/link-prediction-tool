#pragma once

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Configuration for the features
 *
 * Every feature must be commutative, i.e., f(v1, v2) == f(v2, v1).
 * This is important because we want to have the same feature vector
 * for both (v1, v2) and (v2, v1) pairs.
 */
class FeatureConfig {
public:
    // Embedding similarity features
    bool cosine_similarity = true;
    bool dot_product = true;
    bool l1_distance = true; //  Manhattan distance
    bool l2_distance = true; //  Euclidean distance

    // Embedding statistical features
    bool embedding_std = true;   // For both nodes v1 and v2
    bool embedding_abs_mean = true; // For both nodes v1 and v2
    bool embedding_norm_ratio = true;

    // Hadamard product derived features
    bool hadamard_product_sum = true;
    bool hadamard_product_mean = true;
    bool hadamard_product_components = true;

    // Network structure features
    bool common_neighbors_count = true;
    bool jaccard_coefficient = true;
    bool adamic_adar_index = true;
    bool preferential_attachment = true;
    bool resource_allocation_index = true;
    
    // Node-level features with min/max pairs
    bool node_degree = true; // For both nodes v1 and v2

    FeatureConfig() = default;

    /**
     * @brief Get the dimension of the feature vector based on the enabled features.
     *
     * @param embedding_dim The dimension of the embedding.
     * @return The total dimension of the feature vector.
     */
    std::size_t get_dimension(std::size_t embedding_dim) const {
        std::size_t dim = 0;
        dim += cosine_similarity ? 1 : 0;
        dim += l2_distance ? 1 : 0;
        dim += dot_product ? 1 : 0;
        dim += hadamard_product_sum ? 1 : 0;
        dim += hadamard_product_mean ? 1 : 0;
        dim += l1_distance ? 1 : 0;
        dim += common_neighbors_count ? 1 : 0;
        dim += jaccard_coefficient ? 1 : 0;
        dim += node_degree ? 2 : 0; // Two features: degree(v1), degree(v2)
        dim += embedding_std ? 2 : 0;   // Two features: std(v1_emb), std(v2_emb)
        dim += adamic_adar_index ? 1 : 0;
        dim += preferential_attachment ? 1 : 0;
        dim += resource_allocation_index ? 1 : 0;
        dim += embedding_norm_ratio ? 1 : 0;
        dim += embedding_abs_mean ? 2
                                  : 0; // Two features: abs_mean(v1_emb), abs_mean(v2_emb)
        dim += hadamard_product_components ? embedding_dim : 0;
        return dim;
    }

    /**
     * @brief Get the names of the features based on the enabled features.
     *
     * @param embedding_dim The dimension of the embedding.
     * @return A vector of feature names.
     */
    std::vector<std::string> get_feature_names(std::size_t embedding_dim) const {
        std::vector<std::string> names;
        
        // Embedding similarity features
        if (cosine_similarity)
            names.push_back("cosine_similarity");
        if (dot_product)
            names.push_back("dot_product");
        if (l1_distance)
            names.push_back("l1_distance");
        if (l2_distance)
            names.push_back("l2_distance");
        
        // Embedding statistical features
        if (embedding_std) {
            names.push_back("embedding_std_min");
            names.push_back("embedding_std_max");
        }
        if (embedding_abs_mean) {
            names.push_back("embedding_abs_mean_min");
            names.push_back("embedding_abs_mean_max");
        }
        if (embedding_norm_ratio)
            names.push_back("embedding_norm_ratio");
        
        // Hadamard product derived features
        if (hadamard_product_sum)
            names.push_back("hadamard_product_sum");
        if (hadamard_product_mean)
            names.push_back("hadamard_product_mean");
        if (hadamard_product_components) {
            for (std::size_t i = 0; i < embedding_dim; ++i) {
                names.push_back("hadamard_product_component_" + std::to_string(i));
            }
        }

        // Network structure features
        if (common_neighbors_count)
            names.push_back("common_neighbors_count");
        if (jaccard_coefficient)
            names.push_back("jaccard_coefficient");
        if (adamic_adar_index)
            names.push_back("adamic_adar_index");
        if (preferential_attachment)
            names.push_back("preferential_attachment");
        if (resource_allocation_index)
            names.push_back("resource_allocation_index");

        // Node-level features with min/max pairs
        if (node_degree) {
            names.push_back("node_degree_min");
            names.push_back("node_degree_max");
        }

        return names;
    }
};