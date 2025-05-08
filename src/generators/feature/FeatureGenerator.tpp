#pragma once

#include "FeatureGenerator.hpp"
#include "exceptions/exceptions.hpp"
#include <c10/core/TensorOptions.h>
#include <cstddef>
#include <functional>

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::FeatureGenerator(
    const IGraphAnalytics<GraphTraits> &graph_analytics,
    const EmbeddingModule &embedding_module,
    const FeatureConfig &config /*= FeatureConfig()*/)
    : m_graph_analytics(graph_analytics), m_embedding_module(embedding_module),
      m_feature_config(config) {
    if (!m_feature_config.is_any_feature_enabled()) {
        throw FeatureGeneratorException("No features enabled in the configuration.");
    }
}

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

    // Setup and validation
    const std::size_t num_vertices = vertex_to_index.size();
    const std::size_t num_pairs =
        num_vertices * (num_vertices - 1) / 2; // Avoid self-loops and duplicate pairs
    const std::size_t feature_dim =
        m_feature_config.get_dimension(m_embedding_module->options.embedding_dim());

    if (num_pairs == 0) {
        throw FeatureGeneratorException("No vertex pairs to process.");
    }

    // Prepare tensor options and containers
    torch::TensorOptions opts = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .memory_format(torch::MemoryFormat::Contiguous);
    torch::TensorOptions idx_opts = torch::TensorOptions()
                                        .dtype(torch::kInt64)
                                        .device(torch::kCPU)
                                        .memory_format(torch::MemoryFormat::Contiguous);

    torch::Tensor all_features = torch::empty(
        {static_cast<int64_t>(num_pairs), static_cast<int64_t>(feature_dim)}, opts);
    arma::Row<size_t> arma_labels((WithLabels) ? num_pairs : 1, arma::fill::none);
    std::vector<std::pair<IPAddress, IPAddress>> vertex_pairs(
        (WithVertexPairs) ? num_pairs : 1);

    // Extract and sort vertices for consistent ordering
    auto [vertex_ips, all_indices] = extract_sorted_vertices(vertex_to_index);

    // Get all embeddings in a single forward pass
    torch::Tensor all_indices_tensor = torch::tensor(all_indices, idx_opts);
    torch::Tensor all_embeddings = m_embedding_module->forward(all_indices_tensor);

    // Generate features for all vertex pairs
    process_vertex_pairs<WithLabels, WithVertexPairs>(
        vertex_ips, all_embeddings, vertex_to_index, ground_truth_dependencies,
        all_features, arma_labels, vertex_pairs);

    return {all_features, arma_labels, vertex_pairs};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
std::pair<std::vector<IPAddress>, std::vector<int64_t>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    extract_sorted_vertices(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index) {

    // Create a vector of vertex IPs and their corresponding indices
    std::vector<std::pair<IPAddress, int64_t>> vertex_data;
    vertex_data.reserve(vertex_to_index.size());

    for (const auto &[ip, vertex_idx] : vertex_to_index) {
        vertex_data.push_back({ip, static_cast<int64_t>(vertex_idx)});
    }

    // Sort by IP address for consistent iteration order
    std::sort(vertex_data.begin(), vertex_data.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Extract sorted IPs and indices into separate vectors
    std::vector<IPAddress> vertex_ips;
    std::vector<int64_t> all_indices;
    vertex_ips.reserve(vertex_data.size());
    all_indices.reserve(vertex_data.size());

    for (const auto &[ip, idx] : vertex_data) {
        vertex_ips.push_back(ip);
        all_indices.push_back(idx);
    }

    return {vertex_ips, all_indices};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <bool WithLabels, bool WithVertexPairs>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    process_vertex_pairs(const std::vector<IPAddress> &vertex_ips,
                         const torch::Tensor &all_embeddings,
                         const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
                         const GroundTruthDependencies &ground_truth_dependencies,
                         torch::Tensor &all_features, arma::Row<size_t> &arma_labels,
                         std::vector<std::pair<IPAddress, IPAddress>> &vertex_pairs) {

    std::size_t pair_index = 0;

    // Process each pair once (with idx1 < idx2 to avoid duplicates)
    for (size_t idx1 = 0; idx1 < vertex_ips.size(); ++idx1) {
        const auto &ip1 = vertex_ips[idx1];
        const auto v1 = vertex_to_index.at(ip1);
        const torch::Tensor &v1_emb = all_embeddings[idx1];

        for (size_t idx2 = idx1 + 1; idx2 < vertex_ips.size(); ++idx2) {
            const auto &ip2 = vertex_ips[idx2];
            const auto v2 = vertex_to_index.at(ip2);
            const torch::Tensor &v2_emb = all_embeddings[idx2];

            // Create features
            create_features_and_set_to_tensor<float>(v1, v2, v1_emb, v2_emb, all_features,
                                                     pair_index);

            // Handle labels if needed
            if constexpr (WithLabels) {
                arma_labels[pair_index] =
                    (ground_truth_dependencies.contains({ip1, ip2}) ||
                     ground_truth_dependencies.contains({ip2, ip1}))
                        ? 1
                        : 0;
            }

            // Store vertex pairs if needed
            if constexpr (WithVertexPairs) {
                vertex_pairs[pair_index] = {ip1, ip2};
            }

            pair_index++;
        }
    }
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    create_features_and_set_to_tensor(Vertex v1, Vertex v2, const torch::Tensor &v1_emb,
                                      const torch::Tensor &v2_emb,
                                      torch::Tensor &features_tensor,
                                      std::size_t row_index) {
    // Verify tensor dimensions and type
    assert(features_tensor.dim() == 2);
    assert(features_tensor.size(1) ==
           m_feature_config.get_dimension(m_embedding_module->options.embedding_dim()));
    assert((std::is_same_v<T, float> &&
            features_tensor.scalar_type() == c10::ScalarType::Float) ||
           (std::is_same_v<T, double> &&
            features_tensor.scalar_type() == c10::ScalarType::Double));

    auto features_accessor = features_tensor.accessor<T, 2>();
    std::size_t col_index = 0;

    // Apply feature generators by category
    col_index = add_similarity_features<T>(v1_emb, v2_emb, features_accessor, row_index,
                                           col_index);
    col_index = add_statistical_features<T>(v1_emb, v2_emb, features_accessor, row_index,
                                            col_index);
    col_index =
        add_hadamard_features<T>(v1_emb, v2_emb, features_accessor, row_index, col_index);
    col_index = add_network_features<T>(v1, v2, features_accessor, row_index, col_index);
    add_node_features<T>(v1, v2, features_accessor, row_index, col_index);
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    add_similarity_features(const torch::Tensor &v1_emb, const torch::Tensor &v2_emb,
                            torch::TensorAccessor<T, 2> &accessor, std::size_t row,
                            std::size_t col) {
    if (m_feature_config.cosine_similarity) {
        accessor[row][col++] = cosine_similarity<T>(v1_emb, v2_emb);
    }
    if (m_feature_config.dot_product) {
        accessor[row][col++] = dot_product<T>(v1_emb, v2_emb);
    }
    if (m_feature_config.l1_distance) {
        accessor[row][col++] = (v1_emb - v2_emb).abs().sum().item<T>();
    }
    if (m_feature_config.l2_distance) {
        accessor[row][col++] = euclidean_distance<T>(v1_emb, v2_emb);
    }
    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    add_statistical_features(const torch::Tensor &v1_emb, const torch::Tensor &v2_emb,
                             torch::TensorAccessor<T, 2> &accessor, std::size_t row,
                             std::size_t col) {
    if (m_feature_config.embedding_std) {
        T std1 = v1_emb.std().item<T>();
        T std2 = v2_emb.std().item<T>();
        accessor[row][col++] = std::min(std1, std2);
        accessor[row][col++] = std::max(std1, std2);
    }
    if (m_feature_config.embedding_abs_mean) {
        T mean1 = v1_emb.abs().mean().item<T>();
        T mean2 = v2_emb.abs().mean().item<T>();
        accessor[row][col++] = std::min(mean1, mean2);
        accessor[row][col++] = std::max(mean1, mean2);
    }
    if (m_feature_config.embedding_norm_ratio) {
        T v1_norm = v1_emb.norm().item<T>();
        T v2_norm = v2_emb.norm().item<T>();

        accessor[row][col++] =
            (v1_norm > 0 && v2_norm > 0)
                ? std::min(v1_norm, v2_norm) / std::max(v1_norm, v2_norm)
                : 0.0;
    }
    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    add_hadamard_features(const torch::Tensor &v1_emb, const torch::Tensor &v2_emb,
                          torch::TensorAccessor<T, 2> &accessor, std::size_t row,
                          std::size_t col) {
    auto hadamard = v1_emb * v2_emb;
    if (m_feature_config.hadamard_product_sum) {
        accessor[row][col++] = hadamard.sum().item<T>();
    }
    if (m_feature_config.hadamard_product_mean) {
        accessor[row][col++] = hadamard.mean().item<T>();
    }
    if (m_feature_config.hadamard_product_components) {
        for (std::size_t i = 0; i < hadamard.size(0); ++i) {
            accessor[row][col++] = hadamard[i].item<T>();
        }
    }
    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    add_network_features(Vertex v1, Vertex v2, torch::TensorAccessor<T, 2> &accessor,
                         std::size_t row, std::size_t col) {
    if (m_feature_config.common_neighbors_count) {
        accessor[row][col++] =
            m_graph_analytics.normalized_common_neighbors_count(v1, v2);
    }
    if (m_feature_config.jaccard_coefficient) {
        accessor[row][col++] = m_graph_analytics.jaccard_coefficient(v1, v2);
    }
    if (m_feature_config.adamic_adar_index) {
        accessor[row][col++] = m_graph_analytics.adamic_adar_index(v1, v2);
    }
    if (m_feature_config.preferential_attachment) {
        accessor[row][col++] = m_graph_analytics.preferential_attachment(v1, v2);
    }
    if (m_feature_config.resource_allocation_index) {
        accessor[row][col++] = m_graph_analytics.resource_allocation_index(v1, v2);
    }
    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    add_node_features(Vertex v1, Vertex v2, torch::TensorAccessor<T, 2> &accessor,
                      std::size_t row, std::size_t col) {
    if (m_feature_config.node_degree) {
        T avg_degree = get_set_avg_degree<T>();
        T deg1 = m_graph_analytics.degree(v1) / avg_degree;
        T deg2 = m_graph_analytics.degree(v2) / avg_degree;

        accessor[row][col++] = std::min(deg1, deg2);
        accessor[row][col++] = std::max(deg1, deg2);
    }
    return col;
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
    if (!m_avg_degree.has_value()) {
        m_avg_degree = m_graph_analytics.avg_degree();
    }
    return m_avg_degree.value();
}