#pragma once
#include <cstddef>
#include <vector>
#include <string>

class FeatureConfig {
public:
    // Feature flags - defaulting to what your code currently uses
    bool cosine_similarity = true;
    bool euclidean_distance = true;
    bool dot_product = true;
    bool hadamard_sum = true;
    bool hadamard_mean = true;
    bool l1_distance = true;
    bool common_neighbors = true;
    bool jaccard_coefficient = true;
    bool node_degree = true;        // For both nodes v1 and v2
    bool embed_std = true;          // For both nodes v1 and v2
    bool adamic_adar = true;
    bool preferential_attachment = true;
    bool resource_allocation = true;
    bool embedding_ratio = true;
    bool embedding_abs_mean = true; // For both nodes v1 and v2

    // Calculate feature dimension based on enabled features
    std::size_t get_dimension() const {
        std::size_t dim = 0;
        dim += cosine_similarity ? 1 : 0;
        dim += euclidean_distance ? 1 : 0;
        dim += dot_product ? 1 : 0;
        dim += hadamard_sum ? 1 : 0;
        dim += hadamard_mean ? 1 : 0;
        dim += l1_distance ? 1 : 0;
        dim += common_neighbors ? 1 : 0;
        dim += jaccard_coefficient ? 1 : 0;
        dim += node_degree ? 2 : 0;       // Two features: degree(v1), degree(v2)
        dim += embed_std ? 2 : 0;         // Two features: std(v1_emb), std(v2_emb)
        dim += adamic_adar ? 1 : 0;
        dim += preferential_attachment ? 1 : 0;
        dim += resource_allocation ? 1 : 0;
        dim += embedding_ratio ? 1 : 0;
        dim += embedding_abs_mean ? 2 : 0; // Two features: abs_mean(v1_emb), abs_mean(v2_emb)
        return dim;
    }
    
    // Get names of enabled features (for debugging)
    std::vector<std::string> get_feature_names() const {
        std::vector<std::string> names;
        if (cosine_similarity) names.push_back("cosine_similarity");
        if (euclidean_distance) names.push_back("euclidean_distance");
        if (dot_product) names.push_back("dot_product");
        if (hadamard_sum) names.push_back("hadamard_sum");
        if (hadamard_mean) names.push_back("hadamard_mean");
        if (l1_distance) names.push_back("l1_distance");
        if (common_neighbors) names.push_back("common_neighbors");
        if (jaccard_coefficient) names.push_back("jaccard_coefficient");
        if (node_degree) {
            names.push_back("degree_v1");
            names.push_back("degree_v2");
        }
        if (embed_std) {
            names.push_back("embed_std_v1");
            names.push_back("embed_std_v2");
        }
        if (adamic_adar) names.push_back("adamic_adar");
        if (preferential_attachment) names.push_back("preferential_attachment");
        if (resource_allocation) names.push_back("resource_allocation");
        if (embedding_ratio) names.push_back("embedding_ratio");
        if (embedding_abs_mean) {
            names.push_back("embed_abs_mean_v1");
            names.push_back("embed_abs_mean_v2");
        }
        return names;
    }
};