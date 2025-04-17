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
    bool cosine_similarity = true;
    bool euclidean_distance = true;
    bool dot_product = true;
    bool hadamard_sum = true;
    bool hadamard_mean = true;
    bool l1_distance = true;
    bool common_neighbors = true;
    bool jaccard_coefficient = true;
    bool node_degree = true; // For both nodes v1 and v2
    bool embed_std = true;   // For both nodes v1 and v2
    bool adamic_adar = true;
    bool preferential_attachment = true;
    bool resource_allocation = true;
    bool embedding_ratio = true;
    bool embedding_abs_mean = true; // For both nodes v1 and v2
    bool element_wise_product = true;

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
        dim += euclidean_distance ? 1 : 0;
        dim += dot_product ? 1 : 0;
        dim += hadamard_sum ? 1 : 0;
        dim += hadamard_mean ? 1 : 0;
        dim += l1_distance ? 1 : 0;
        dim += common_neighbors ? 1 : 0;
        dim += jaccard_coefficient ? 1 : 0;
        dim += node_degree ? 2 : 0; // Two features: degree(v1), degree(v2)
        dim += embed_std ? 2 : 0;   // Two features: std(v1_emb), std(v2_emb)
        dim += adamic_adar ? 1 : 0;
        dim += preferential_attachment ? 1 : 0;
        dim += resource_allocation ? 1 : 0;
        dim += embedding_ratio ? 1 : 0;
        dim += embedding_abs_mean ? 2
                                  : 0; // Two features: abs_mean(v1_emb), abs_mean(v2_emb)
        dim += element_wise_product ? embedding_dim : 0;
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
        if (cosine_similarity)
            names.push_back("cosine_similarity");
        if (euclidean_distance)
            names.push_back("euclidean_distance");
        if (dot_product)
            names.push_back("dot_product");
        if (hadamard_sum)
            names.push_back("hadamard_sum");
        if (hadamard_mean)
            names.push_back("hadamard_mean");
        if (l1_distance)
            names.push_back("l1_distance");
        if (common_neighbors)
            names.push_back("common_neighbors");
        if (jaccard_coefficient)
            names.push_back("jaccard_coefficient");
        if (node_degree) {
            names.push_back("min(degree_v1, degree_v2)");
            names.push_back("max(degree_v1, degree_v2)");
        }
        if (embed_std) {
            names.push_back("min(embed_std_v1, embed_std_v2)");
            names.push_back("max(embed_std_v1, embed_std_v2)");
        }
        if (adamic_adar)
            names.push_back("adamic_adar");
        if (preferential_attachment)
            names.push_back("preferential_attachment");
        if (resource_allocation)
            names.push_back("resource_allocation");
        if (embedding_ratio)
            names.push_back("embedding_ratio");
        if (embedding_abs_mean) {
            names.push_back("min(embed_abs_mean_v1, embed_abs_mean_v2)");
            names.push_back("max(embed_abs_mean_v1, embed_abs_mean_v2)");
        }
        if (element_wise_product) {
            for (std::size_t i = 0; i < embedding_dim; ++i) {
                names.push_back("element_wise_product_" + std::to_string(i));
            }
        }
        return names;
    }
};