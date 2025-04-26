#pragma once

#include "FeatureGenerator.hpp"
#include "exceptions/exceptions.hpp"
#include <c10/core/TensorOptions.h>
#include <cstddef>

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::FeatureGenerator(
    const IGraphAnalytics<GraphTraits> &graph_analytics,
    const EmbeddingModule &embedding_module,
    const FeatureConfig &config /*= FeatureConfig()*/)
    : m_graph_analytics(graph_analytics), m_embedding_module(embedding_module),
      m_feature_config(config) {}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
std::tuple<torch::Tensor, arma::Row<size_t>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_labeled_features(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) {
    auto [features, labels, _] = generate_features_and_labels_impl<true, false>(
        vertex_to_index, ground_truth_dependencies);
    return {features, labels};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
std::tuple<torch::Tensor, arma::Row<size_t>, std::vector<std::pair<IPAddress, IPAddress>>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_labeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) {
    return generate_features_and_labels_impl<true, true>(vertex_to_index,
                                                         ground_truth_dependencies);
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index) {
    auto [features, _, vertex_pairs] =
        generate_features_and_labels_impl<false, true>(vertex_to_index, {});
    return {features, vertex_pairs};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <bool WithLabels /*= true */, bool WithVertexPairs /*= false */>
std::tuple<torch::Tensor, arma::Row<size_t>, std::vector<std::pair<IPAddress, IPAddress>>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_features_and_labels_impl(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) {

    const std::size_t num_vertices = vertex_to_index.size();
    // Avoid self-loops and duplicate pairs
    const std::size_t num_pairs = num_vertices * (num_vertices - 1) / 2;
    const std::size_t feature_dim = m_feature_config.get_dimension(
        m_embedding_module->options.embedding_dim());

    if (num_pairs == 0) {
        throw FeatureGeneratorException("No vertex pairs to process.");
    }

    // Prepare result containers
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor all_features = torch::empty(
        {static_cast<int64_t>(num_pairs), static_cast<int64_t>(feature_dim)}, opts);
    arma::Row<size_t> arma_labels((WithLabels) ? num_pairs : 1, arma::fill::none);
    std::vector<std::pair<IPAddress, IPAddress>> vertex_pairs(
        (WithVertexPairs) ? num_pairs : 1);

    // Create a vector of vertex IPs for consistent ordering
    std::vector<IPAddress> vertex_ips;
    vertex_ips.reserve(vertex_to_index.size());
    for (const auto& [ip, _] : vertex_to_index) {
        vertex_ips.push_back(ip);
    }
    
    // Sort to ensure consistent iteration order
    std::sort(vertex_ips.begin(), vertex_ips.end());

    std::size_t i = 0;
    // Only process each pair once with idx1 < idx2 to avoid duplicates
    for (size_t idx1 = 0; idx1 < vertex_ips.size(); ++idx1) {
        const auto& ip1 = vertex_ips[idx1];
        const auto v1 = vertex_to_index.at(ip1);
        
        for (size_t idx2 = idx1 + 1; idx2 < vertex_ips.size(); ++idx2) {
            const auto& ip2 = vertex_ips[idx2];
            const auto v2 = vertex_to_index.at(ip2);
            
            // Get embeddings
            torch::Tensor v1_emb =
                m_embedding_module->forward(torch::tensor({static_cast<int64_t>(v1)}))
                    .squeeze();
            torch::Tensor v2_emb =
                m_embedding_module->forward(torch::tensor({static_cast<int64_t>(v2)}))
                    .squeeze();

            // Create features and set to tensor
            create_features_and_set_to_tensor<float>(v1, v2, v1_emb, v2_emb, all_features,
                                                     i);

            if constexpr (WithLabels) {
                if (ground_truth_dependencies.contains({ip1, ip2}) ||
                    ground_truth_dependencies.contains({ip2, ip1})) {
                    arma_labels[i] = 1;
                } else {
                    arma_labels[i] = 0;
                }
            }

            if constexpr (WithVertexPairs) {
                vertex_pairs[i] = {ip1, ip2};
            }

            i++;
        }
    }

    return {all_features, arma_labels, vertex_pairs};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    create_features_and_set_to_tensor(Vertex v1, Vertex v2, const torch::Tensor &v1_emb,
                                      const torch::Tensor &v2_emb,
                                      torch::Tensor &features_tensor,
                                      std::size_t row_index) {
    // assert tensor has the correct shape and correct type T
    assert(features_tensor.dim() == 2);
    assert(features_tensor.size(1) == m_feature_config.get_dimension(
        m_embedding_module->options.embedding_dim()));
    if constexpr (std::is_same_v<T, float>) {
        assert(features_tensor.scalar_type() == c10::ScalarType::Float);
    } else if constexpr (std::is_same_v<T, double>) {
        assert(features_tensor.scalar_type() == c10::ScalarType::Double);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float and double types are supported");
    }

    // Get an accessor for efficient tensor access
    auto features_accessor = features_tensor.accessor<T, 2>();
    std::size_t j = 0;

    // Embedding similarity features
    if (m_feature_config.cosine_similarity) {
        features_accessor[row_index][j++] = cosine_similarity<T>(v1_emb, v2_emb);
    }
    if (m_feature_config.dot_product) {
        features_accessor[row_index][j++] = dot_product<T>(v1_emb, v2_emb);
    }
    if (m_feature_config.l1_distance) {
        features_accessor[row_index][j++] = (v1_emb - v2_emb).abs().sum().item<T>();
    }
    if (m_feature_config.l2_distance) {
        features_accessor[row_index][j++] = euclidean_distance<T>(v1_emb, v2_emb);
    }
    
    // Embedding statistical features
    if (m_feature_config.embedding_std) {
        T std1 = v1_emb.std().item<T>();
        T std2 = v2_emb.std().item<T>();

        features_accessor[row_index][j++] = std::min(std1, std2);
        features_accessor[row_index][j++] = std::max(std1, std2);
    }
    if (m_feature_config.embedding_abs_mean) {
        T mean1 = v1_emb.abs().mean().item<T>();
        T mean2 = v2_emb.abs().mean().item<T>();

        features_accessor[row_index][j++] = std::min(mean1, mean2);
        features_accessor[row_index][j++] = std::max(mean1, mean2);
    }
    if (m_feature_config.embedding_norm_ratio) {
        T v1_norm = v1_emb.norm().item<T>();
        T v2_norm = v2_emb.norm().item<T>();

        if (v1_norm > 0 && v2_norm > 0) {
            features_accessor[row_index][j++] =
                std::min(v1_norm, v2_norm) / std::max(v1_norm, v2_norm);
        } else {
            features_accessor[row_index][j++] = 0.0;
        }
    }

    // Hadamard product derived features
    auto hadamard = v1_emb * v2_emb;
    if (m_feature_config.hadamard_product_sum) {
        features_accessor[row_index][j++] = hadamard.sum().item<T>();
    }
    if (m_feature_config.hadamard_product_mean) {
        features_accessor[row_index][j++] = hadamard.mean().item<T>();
    }
    if (m_feature_config.hadamard_product_components) {
        for (std::size_t i = 0; i < hadamard.size(0); ++i) {
            features_accessor[row_index][j++] = hadamard[i].item<T>();
        }
    }

    // Network structure features
    if (m_feature_config.common_neighbors_count) {
        features_accessor[row_index][j++] =
            m_graph_analytics.normalized_common_neighbors_count(v1, v2);
    }
    if (m_feature_config.jaccard_coefficient) {
        features_accessor[row_index][j++] = m_graph_analytics.jaccard_coefficient(v1, v2);
    }
    if (m_feature_config.adamic_adar_index) {
        features_accessor[row_index][j++] = m_graph_analytics.adamic_adar_index(v1, v2);
    }
    if (m_feature_config.preferential_attachment) {
        features_accessor[row_index][j++] =
            m_graph_analytics.preferential_attachment(v1, v2);
    }
    if (m_feature_config.resource_allocation_index) {
        features_accessor[row_index][j++] = m_graph_analytics.resource_allocation_index(v1, v2);
    }

    // Node-level features with min/max pairs
    if (m_feature_config.node_degree) {
        T deg1 = m_graph_analytics.degree(v1) / get_set_avg_degree<T>();
        T deg2 = m_graph_analytics.degree(v2) / get_set_avg_degree<T>();

        features_accessor[row_index][j++] = std::min(deg1, deg2);
        features_accessor[row_index][j++] = std::max(deg1, deg2);
    }
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
T FeatureGenerator<GraphTraits, EmbeddingModule,
                   GroundTruthDependencies>::cosine_similarity(const torch::Tensor &a,
                                                               const torch::Tensor &b) {
    // Calculate norms
    T norm_a = a.norm().item<T>();
    T norm_b = b.norm().item<T>();

    // Avoid division by zero
    if (norm_a < 1e-8 || norm_b < 1e-8) {
        return 0.0;
    }

    // Calculate cosine similarity: dot(a,b) / (||a|| * ||b||)
    return torch::dot(a, b).item<T>() / (norm_a * norm_b);
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
T FeatureGenerator<GraphTraits, EmbeddingModule,
                   GroundTruthDependencies>::euclidean_distance(const torch::Tensor &a,
                                                                const torch::Tensor &b) {
    // Calculate Euclidean (L2) distance
    return torch::pairwise_distance(a.unsqueeze(0), b.unsqueeze(0)).item<T>();
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
T FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::dot_product(
    const torch::Tensor &a, const torch::Tensor &b) {
    // Calculate dot product: a·b
    return torch::dot(a, b).item<T>();
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
T FeatureGenerator<GraphTraits, EmbeddingModule,
                   GroundTruthDependencies>::get_set_avg_degree() {
    if (m_avg_degree.has_value()) {
        return m_avg_degree.value();
    }

    m_avg_degree = m_graph_analytics.avg_degree();
    return m_avg_degree.value();
}