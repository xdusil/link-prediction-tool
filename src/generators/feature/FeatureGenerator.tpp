#pragma once

#include "FeatureGenerator.hpp"
#include "exceptions/exceptions.hpp"
#include <c10/core/TensorOptions.h>

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::FeatureGenerator(
    const IGraphAnalytics<GraphTraits>& graph_analytics,
    EmbeddingModule& embedding_module, const FeatureConfig& config)
    : m_graph_analytics{graph_analytics}, m_embedding_module{embedding_module},
      m_config{config} {

    if (!m_config.is_any_feature_enabled()) {
        throw FeatureGeneratorException("No features enabled in configuration.");
    }
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
std::tuple<torch::Tensor, arma::Row<std::size_t>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_labeled_features(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const GroundTruthDependencies& ground_truth) {

    auto [features, labels, _] =
        generate_impl<true, false>(vertex_to_index, ground_truth);
    return {std::move(features), std::move(labels)};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
std::tuple<torch::Tensor, arma::Row<std::size_t>,
           std::vector<std::pair<IPAddress, IPAddress>>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_labeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const GroundTruthDependencies& ground_truth) {

    return generate_impl<true, true>(vertex_to_index, ground_truth);
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index) {

    auto [features, _, pairs] = generate_impl<false, true>(vertex_to_index, {});
    return {std::move(features), std::move(pairs)};
}

// ============================================================================
// Core Implementation
// ============================================================================

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <bool WithLabels, bool WithPairs>
std::tuple<torch::Tensor, arma::Row<std::size_t>,
           std::vector<std::pair<IPAddress, IPAddress>>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::generate_impl(
    const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
    const GroundTruthDependencies& ground_truth) {

    const std::size_t num_vertices = vertex_to_index.size();
    const std::size_t num_pairs = num_vertices * (num_vertices - 1);
    const std::size_t feature_dim = m_config.get_dimension();

    if (num_pairs == 0)
        throw FeatureGeneratorException("No vertex pairs to process.");

    auto tensor_opts = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .memory_format(torch::MemoryFormat::Contiguous);

    torch::Tensor features =
        torch::empty({static_cast<int64_t>(num_pairs), static_cast<int64_t>(feature_dim)},
                     tensor_opts);

    arma::Row<std::size_t> labels(WithLabels ? num_pairs : 1, arma::fill::none);
    std::vector<std::pair<IPAddress, IPAddress>> pairs;
    if constexpr (WithPairs) {
        pairs.resize(num_pairs);
    }

    auto [vertex_ips, indices] = prepare_sorted_vertices(vertex_to_index);

    // Pre-compute expensive features if needed
    if (m_config.struct_shortest_path || m_config.struct_transitive_reachability) {
        std::vector<Vertex> vertices;
        vertices.reserve(vertex_to_index.size());
        for (const auto& [ip, v] : vertex_to_index) {
            vertices.push_back(v);
        }
        precompute_expensive_features(vertices);
    }

    torch::Tensor source_embeddings;
    torch::Tensor dest_embeddings;
    if (m_config.are_embedding_features_enabled()) {
        auto idx_opts = torch::TensorOptions()
                            .dtype(torch::kInt64)
                            .device(torch::kCPU)
                            .memory_format(torch::MemoryFormat::Contiguous);

        torch::Tensor indices_tensor = torch::tensor(indices, idx_opts);

        source_embeddings = m_embedding_module.forward_src(indices_tensor);
        dest_embeddings = m_embedding_module.forward_dst(indices_tensor);
    }

    // Compute graph statistics for normalization
    compute_graph_statistics(vertex_to_index, source_embeddings, dest_embeddings);

    process_all_pairs<WithLabels, WithPairs>(vertex_ips, source_embeddings,
                                             dest_embeddings, vertex_to_index,
                                             ground_truth, features, labels, pairs);

    // Print profiling results
    m_profiler.print_results();
    m_profiler.reset();

    return {std::move(features), std::move(labels), std::move(pairs)};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
std::pair<std::vector<IPAddress>, std::vector<int64_t>>
FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    prepare_sorted_vertices(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index) const {

    std::vector<std::pair<IPAddress, int64_t>> data;
    data.reserve(vertex_to_index.size());

    for (const auto& [ip, vertex] : vertex_to_index) {
        data.emplace_back(ip, static_cast<int64_t>(vertex));
    }

    std::sort(data.begin(), data.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<IPAddress> ips;
    std::vector<int64_t> indices;
    ips.reserve(data.size());
    indices.reserve(data.size());

    for (auto& [ip, idx] : data) {
        ips.push_back(std::move(ip));
        indices.push_back(idx);
    }

    return {std::move(ips), std::move(indices)};
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <bool WithLabels, bool WithPairs>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    process_all_pairs(const std::vector<IPAddress>& vertex_ips,
                      const torch::Tensor& source_embeddings,
                      const torch::Tensor& dest_embeddings,
                      const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
                      const GroundTruthDependencies& ground_truth,
                      torch::Tensor& features, arma::Row<std::size_t>& labels,
                      std::vector<std::pair<IPAddress, IPAddress>>& pairs) {
    const std::size_t directed_pairs = vertex_ips.size() * (vertex_ips.size() - 1);
    if (WithLabels && labels.n_elem != directed_pairs) {
        throw FeatureGeneratorException(
            "Labels size does not match number of vertex pairs.");
    }
    if (WithPairs && pairs.size() != directed_pairs) {
        throw FeatureGeneratorException(
            "Pairs size does not match number of vertex pairs.");
    }
    if (features.size(0) != directed_pairs) {
        throw FeatureGeneratorException(
            "Features size does not match number of vertex pairs.");
    }
    if (features.size(1) != static_cast<int64_t>(m_config.get_dimension())) {
        throw FeatureGeneratorException(
            "Features dimension does not match configuration.");
    }

    const bool use_embeddings = m_config.are_embedding_features_enabled();
    if (use_embeddings) {
        if (source_embeddings.size(0) != static_cast<int64_t>(vertex_ips.size()) ||
            dest_embeddings.size(0) != static_cast<int64_t>(vertex_ips.size())) {
            throw FeatureGeneratorException(
                "Embedding sizes do not match number of vertices.");
        }
    }

    const torch::Tensor empty_tensor;
    auto accessor = features.accessor<float, 2>();

    std::size_t pair_idx = 0;

    for (std::size_t i = 0; i < vertex_ips.size(); ++i) {
        const auto& ip_src = vertex_ips[i];
        const auto v_src = vertex_to_index.at(ip_src);
        // Use SOURCE embedding for source node (caller/client role)
        const auto& src_emb = use_embeddings ? source_embeddings[i] : empty_tensor;

        for (std::size_t j = 0; j < vertex_ips.size(); ++j) {
            if (i == j)
                continue;

            const auto& ip_dst = vertex_ips[j];
            const auto v_dst = vertex_to_index.at(ip_dst);
            // Use DESTINATION embedding for destination node (callee/server role)
            const auto& dst_emb = use_embeddings ? dest_embeddings[j] : empty_tensor;

            write_pair_features<float>(v_src, v_dst, src_emb, dst_emb, accessor,
                                       pair_idx);

            if constexpr (WithLabels) {
                labels[pair_idx] = ground_truth.contains({ip_src, ip_dst}) ? 1 : 0;
            }

            if constexpr (WithPairs) {
                pairs[pair_idx] = {ip_src, ip_dst};
            }

            ++pair_idx;
        }
    }
}

// ============================================================================
// Feature Writers
// ============================================================================

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_pair_features(Vertex src, Vertex dst, const torch::Tensor& src_emb,
                        const torch::Tensor& dst_emb,
                        torch::TensorAccessor<T, 2>& accessor, std::size_t row) {

    std::size_t col = 0;

    col = write_embedding_similarity_features<T>(src_emb, dst_emb, accessor, row, col);
    col = write_embedding_asymmetry_features<T>(src_emb, dst_emb, accessor, row, col);
    col = write_hadamard_features<T>(src_emb, dst_emb, accessor, row, col);
    col = write_structural_features<T>(src, dst, accessor, row, col);
    col = write_temporal_features<T>(src, dst, accessor, row, col);
    col = write_flow_features<T>(src, dst, accessor, row, col);
    col = write_protocol_features<T>(src, dst, accessor, row, col);
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_embedding_similarity_features(const torch::Tensor& src_emb,
                                        const torch::Tensor& dst_emb,
                                        torch::TensorAccessor<T, 2>& accessor,
                                        std::size_t row, std::size_t col) {

    if (m_config.emb_dot_src_dst) {
        PROFILE_FEATURE(m_profiler, "emb_dot_src_dst");
        accessor[row][col++] = math::dot_product<T>(src_emb, dst_emb);
    }
    if (m_config.emb_cosine_src_dst) {
        PROFILE_FEATURE(m_profiler, "emb_cosine_src_dst");
        accessor[row][col++] = math::cosine_similarity<T>(src_emb, dst_emb);
    }
    if (m_config.emb_l1_src_dst) {
        PROFILE_FEATURE(m_profiler, "emb_l1_src_dst");
        accessor[row][col++] = math::l1_distance<T>(src_emb, dst_emb);
    }
    if (m_config.emb_l2_src_dst) {
        PROFILE_FEATURE(m_profiler, "emb_l2_src_dst");
        accessor[row][col++] = math::l2_distance<T>(src_emb, dst_emb);
    }

    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_embedding_asymmetry_features(const torch::Tensor& src_emb,
                                       const torch::Tensor& dst_emb,
                                       torch::TensorAccessor<T, 2>& accessor,
                                       std::size_t row, std::size_t col) {

    T src_norm = static_cast<T>(0);
    T dst_norm = static_cast<T>(0);

    if (m_config.emb_src_norm || m_config.emb_norm_ratio) {
        src_norm = math::tensor_norm<T>(src_emb);
    }
    if (m_config.emb_dst_norm || m_config.emb_norm_ratio) {
        dst_norm = math::tensor_norm<T>(dst_emb);
    }

    if (m_config.emb_src_norm) {
        PROFILE_FEATURE(m_profiler, "emb_src_norm");
        T percentile = math::get_percentile_rank<T, double>(src_norm, m_graph_stats.sorted_embedding_norms);
        accessor[row][col++] = percentile;
    }
    if (m_config.emb_dst_norm) {
        PROFILE_FEATURE(m_profiler, "emb_dst_norm");
        T percentile = math::get_percentile_rank<T, double>(dst_norm, m_graph_stats.sorted_embedding_norms);
        accessor[row][col++] = percentile;
    }
    if (m_config.emb_norm_ratio) {
        PROFILE_FEATURE(m_profiler, "emb_norm_ratio");
        accessor[row][col++] = math::safe_ratio<T>(src_norm, dst_norm);
    }

    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_hadamard_features(const torch::Tensor& src_emb, const torch::Tensor& dst_emb,
                            torch::TensorAccessor<T, 2>& accessor, std::size_t row,
                            std::size_t col) {

    if (m_config.emb_hadamard_sum) {
        PROFILE_FEATURE(m_profiler, "emb_hadamard_sum");
        accessor[row][col++] = math::hadamard_sum<T>(src_emb, dst_emb);
    }
    if (m_config.emb_hadamard_mean) {
        PROFILE_FEATURE(m_profiler, "emb_hadamard_mean");
        accessor[row][col++] = math::hadamard_mean<T>(src_emb, dst_emb);
    }

    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_structural_features(Vertex src, Vertex dst,
                              torch::TensorAccessor<T, 2>& accessor, std::size_t row,
                              std::size_t col) {

    if (!m_config.are_structural_features_enabled()) {
        return col;
    }

    // Degree features
    if (m_config.struct_in_degree_src) {
        PROFILE_FEATURE(m_profiler, "struct_in_degree_src");
        T log_degree = std::log1p(static_cast<T>(m_graph_analytics.in_degree(src)));
        T percentile = math::get_percentile_rank<T, double>(log_degree, m_graph_stats.sorted_degrees);
        accessor[row][col++] = percentile;
    }
    if (m_config.struct_out_degree_src) {
        PROFILE_FEATURE(m_profiler, "struct_out_degree_src");
        T log_degree = std::log1p(static_cast<T>(m_graph_analytics.out_degree(src)));
        T percentile = math::get_percentile_rank<T, double>(log_degree, m_graph_stats.sorted_degrees);
        accessor[row][col++] = percentile;
    }
    if (m_config.struct_in_degree_dst) {
        PROFILE_FEATURE(m_profiler, "struct_in_degree_dst");
        T log_degree = std::log1p(static_cast<T>(m_graph_analytics.in_degree(dst)));
        T percentile = math::get_percentile_rank<T, double>(log_degree, m_graph_stats.sorted_degrees);
        accessor[row][col++] = percentile;
    }
    if (m_config.struct_out_degree_dst) {
        PROFILE_FEATURE(m_profiler, "struct_out_degree_dst");
        T log_degree = std::log1p(static_cast<T>(m_graph_analytics.out_degree(dst)));
        T percentile = math::get_percentile_rank<T, double>(log_degree, m_graph_stats.sorted_degrees);
        accessor[row][col++] = percentile;
    }
    if (m_config.struct_degree_ratio) {
        PROFILE_FEATURE(m_profiler, "struct_degree_ratio");
        T out_src = static_cast<T>(m_graph_analytics.out_degree(src));
        T in_dst = static_cast<T>(m_graph_analytics.in_degree(dst));
        accessor[row][col++] = math::safe_ratio<T>(out_src, in_dst);
    }

    // Cache common neighbors once and reuse for next features
    const bool need_common_neighbors =
        m_config.struct_common_neighbors || m_config.struct_jaccard_coefficient ||
        m_config.struct_adamic_adar_index || m_config.struct_resource_allocation;

    std::vector<Vertex> common_neighbors;
    std::size_t out_deg_src = 0, in_deg_dst = 0;

    if (need_common_neighbors) {
        common_neighbors = m_graph_analytics.get_common_neighbors(src, dst);
        out_deg_src = m_graph_analytics.out_degree(src);
        in_deg_dst = m_graph_analytics.in_degree(dst);
    }

    // Common neighbors (normalized)
    if (m_config.struct_common_neighbors) {
        PROFILE_FEATURE(m_profiler, "struct_common_neighbors");
        std::size_t max_possible = std::min(out_deg_src, in_deg_dst);
        double normalized = math::safe_ratio<T>(common_neighbors.size(), max_possible);
        accessor[row][col++] = static_cast<T>(normalized);
    }

    // Jaccard coefficient
    if (m_config.struct_jaccard_coefficient) {
        PROFILE_FEATURE(m_profiler, "struct_jaccard_coefficient");
        // |out_neighbors(src) ∪ in_neighbors(dst)| = out_deg + in_deg - common
        std::size_t union_size = out_deg_src + in_deg_dst - common_neighbors.size();
        double jaccard = math::safe_ratio<T>(common_neighbors.size(), union_size);
        accessor[row][col++] = static_cast<T>(jaccard);
    }

    // Adamic-Adar index
    if (m_config.struct_adamic_adar_index) {
        PROFILE_FEATURE(m_profiler, "struct_adamic_adar_index");
        double score = 0.0;
        for (const auto& w : common_neighbors) {
            // Use out_degree(w) - w acts as intermediary forwarding connections
            std::size_t out_deg_w = m_graph_analytics.out_degree(w);
            if (out_deg_w > 1) {
                score += 1.0 / std::log(static_cast<double>(out_deg_w));
            }
        }
        accessor[row][col++] = static_cast<T>(std::log1p(score));
    }

    // Preferential attachment
    if (m_config.struct_preferential_attachment) {
        PROFILE_FEATURE(m_profiler, "struct_preferential_attachment");
        double pa_value = static_cast<double>(m_graph_analytics.preferential_attachment(src, dst));
        accessor[row][col++] = static_cast<T>(std::log1p(pa_value));
    }

    // Resource allocation index
    if (m_config.struct_resource_allocation) {
        PROFILE_FEATURE(m_profiler, "struct_resource_allocation");
        double score = 0.0;
        for (const auto& w : common_neighbors) {
            // Use out_degree(w) - w's capacity to forward resources
            std::size_t out_deg_w = m_graph_analytics.out_degree(w);
            if (out_deg_w > 0) {
                score += 1.0 / out_deg_w;
            }
        }
        accessor[row][col++] = static_cast<T>(std::log1p(score));
    }

    // Transitive reachability (2-hop paths)
    if (m_config.struct_transitive_reachability) {
        PROFILE_FEATURE(m_profiler, "struct_transitive_reachability");
        std::size_t count = m_transitive_path_cache.get(src, dst).value_or(0);
        accessor[row][col++] = static_cast<T>(std::log1p(static_cast<double>(count)));
    }

    // Shortest path (inverse distance, 0 for unreachable)
    if (m_config.struct_shortest_path) {
        PROFILE_FEATURE(m_profiler, "struct_shortest_path");
        std::size_t dist = m_shortest_path_cache.get(src, dst).value_or(999);
        // Transform to inverse distance: closer = higher value, unreachable = 0
        if (dist == 999) {
            accessor[row][col++] = static_cast<T>(0);
        } else {
            accessor[row][col++] = static_cast<T>(1.0 / static_cast<double>(dist));
        }
    }

    // Hierarchy difference
    if (m_config.struct_hierarchy_diff) {
        PROFILE_FEATURE(m_profiler, "struct_hierarchy_diff");
        T in_src = static_cast<T>(m_graph_analytics.in_degree(src));
        T out_src = static_cast<T>(m_graph_analytics.out_degree(src));
        T in_dst = static_cast<T>(m_graph_analytics.in_degree(dst));
        T out_dst = static_cast<T>(m_graph_analytics.out_degree(dst));

        T level_src = std::log(in_src + static_cast<T>(1.0)) -
                      std::log(out_src + static_cast<T>(1.0));
        T level_dst = std::log(in_dst + static_cast<T>(1.0)) -
                      std::log(out_dst + static_cast<T>(1.0));
        accessor[row][col++] = level_dst - level_src;
    }

    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_temporal_features(Vertex src, Vertex dst, torch::TensorAccessor<T, 2>& accessor,
                            std::size_t row, std::size_t col) {

    if (!m_config.are_temporal_features_enabled()) {
        return col;
    }

    decltype(TemporalFeatureExtractor::extract(m_graph_analytics.get_graph_manager(), src,
                                               dst, m_config)) features;
    {
        PROFILE_FEATURE(m_profiler, "TemporalFeatureExtractor::extract");
        features = TemporalFeatureExtractor::extract(
            m_graph_analytics.get_graph_manager(), src, dst, m_config);
    }

    if (m_config.time_avg_duration) {
        PROFILE_FEATURE(m_profiler, "time_avg_duration");
        double duration = features.avg_duration.value_or(0.0);
        accessor[row][col++] = static_cast<T>(std::log1p(duration));
    }
    if (m_config.time_avg_interarrival) {
        PROFILE_FEATURE(m_profiler, "time_avg_interarrival");
        double interarrival = features.avg_interarrival.value_or(0.0);
        accessor[row][col++] = static_cast<T>(std::log1p(interarrival));
    }
    if (m_config.time_regularity) {
        PROFILE_FEATURE(m_profiler, "time_regularity");
        accessor[row][col++] = static_cast<T>(features.regularity.value_or(0.0));
    }
    if (m_config.time_direction_bias) {
        PROFILE_FEATURE(m_profiler, "time_direction_bias");
        accessor[row][col++] = static_cast<T>(features.direction_bias.value_or(0.0));
    }
    if (m_config.time_initiation_order) {
        PROFILE_FEATURE(m_profiler, "time_initiation_order");
        accessor[row][col++] = static_cast<T>(features.initiation_order.value_or(0.0));
    }
    if (m_config.time_crosscorr_peak) {
        PROFILE_FEATURE(m_profiler, "time_crosscorr_peak");
        accessor[row][col++] = static_cast<T>(features.crosscorr_max.value_or(0.0));
        accessor[row][col++] = static_cast<T>(features.crosscorr_lag.value_or(0.0));
    }
    if (m_config.time_spike_score) {
        PROFILE_FEATURE(m_profiler, "time_spike_score");
        accessor[row][col++] = static_cast<T>(features.spike_score.value_or(0.0));
    }

    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_flow_features(Vertex src, Vertex dst, torch::TensorAccessor<T, 2>& accessor,
                        std::size_t row, std::size_t col) {

    if (!m_config.are_flow_features_enabled()) {
        return col;
    }

    decltype(BidirectionalFeatureExtractor::extract(m_graph_analytics.get_graph_manager(),
                                                    src, dst, m_config)) features;
    {
        PROFILE_FEATURE(m_profiler, "BidirectionalFeatureExtractor::extract");
        features = BidirectionalFeatureExtractor::extract(
            m_graph_analytics.get_graph_manager(), src, dst, m_config);
    }

    if (m_config.flow_response_time) {
        PROFILE_FEATURE(m_profiler, "flow_response_time");
        double response_time = features.response_time.value_or(0.0);
        accessor[row][col++] = static_cast<T>(std::log1p(response_time));
    }
    if (m_config.flow_request_ratio) {
        PROFILE_FEATURE(m_profiler, "flow_request_ratio");
        accessor[row][col++] = static_cast<T>(features.request_ratio.value_or(0.0));
    }
    if (m_config.flow_direction_asymmetry) {
        PROFILE_FEATURE(m_profiler, "flow_direction_asymmetry");
        accessor[row][col++] = static_cast<T>(features.direction_asymmetry.value_or(0.0));
    }
    if (m_config.flow_causality_score) {
        PROFILE_FEATURE(m_profiler, "flow_causality_score");
        accessor[row][col++] = static_cast<T>(features.causality_score.value_or(0.0));
    }

    return col;
}

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
template <typename T>
std::size_t FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    write_protocol_features(Vertex src, Vertex dst, torch::TensorAccessor<T, 2>& accessor,
                            std::size_t row, std::size_t col) {

    if (!m_config.are_network_features_enabled()) {
        return col;
    }

    decltype(ProtocolFeatureExtractor::extract(m_graph_analytics.get_graph_manager(), src,
                                               dst, m_config)) features;
    {
        PROFILE_FEATURE(m_profiler, "ProtocolFeatureExtractor::extract");
        features = ProtocolFeatureExtractor::extract(
            m_graph_analytics.get_graph_manager(), src, dst, m_config);
    }

    if (m_config.net_protocol_role) {
        PROFILE_FEATURE(m_profiler, "net_protocol_role");
        accessor[row][col++] = static_cast<T>(features.protocol_role.value_or(0.0));
    }
    if (m_config.net_port_role) {
        PROFILE_FEATURE(m_profiler, "net_port_role");
        accessor[row][col++] = static_cast<T>(features.port_role.value_or(0.0));
    }
    if (m_config.net_top_port) {
        PROFILE_FEATURE(m_profiler, "net_top_port");
        accessor[row][col++] = static_cast<T>(features.top_port.value_or(0.0));
    }

    return col;
}

// ============================================================================
// Helper Utilities
// ============================================================================

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    precompute_expensive_features(const std::vector<Vertex>& vertices) const {

    std::cout << "[FeatureGenerator] Pre-computing expensive features for "
              << vertices.size() << " vertices..." << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    const auto& gm = m_graph_analytics.get_graph_manager();

    // Pre-compute shortest paths
    if (m_config.struct_shortest_path) {
        auto start = std::chrono::high_resolution_clock::now();
        m_shortest_path_cache.clear();

        std::size_t total_paths = 0;
        for (const auto& src : vertices) {
            // BFS from src to all other vertices using unordered_set for visited
            std::queue<std::pair<Vertex, std::size_t>> queue;
            std::unordered_map<Vertex, std::size_t> distances;
            distances.reserve(vertices.size()); // Pre-allocate

            queue.push({src, 0});
            distances[src] = 0;

            while (!queue.empty()) {
                auto [current, dist] = queue.front();
                queue.pop();

                for (const auto& neighbor : gm.get_out_neighbors(current)) {
                    if (distances.find(neighbor) == distances.end()) {
                        distances[neighbor] = dist + 1;
                        queue.push({neighbor, dist + 1});
                    }
                }
            }

            // Batch insert: store only indirect paths (dist > 1)
            for (const auto& [dst, dist] : distances) {
                if (dst != src && dist > 1) { // Exclude self and direct edges
                    m_shortest_path_cache.put(src, dst, dist);
                    ++total_paths;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  [Shortest Path] " << elapsed << "ms -> cached " << total_paths
                  << " paths" << std::endl;
    }

    // Pre-compute transitive paths (2-hop counts)
    if (m_config.struct_transitive_reachability) {
        auto start = std::chrono::high_resolution_clock::now();
        m_transitive_path_cache.clear();

        std::size_t total_paths = 0;
        for (const auto& src : vertices) {
            std::unordered_map<Vertex, std::size_t> two_hop_counts;
            two_hop_counts.reserve(vertices.size());

            // Follow src -> intermediate -> dst (directed 2-hop paths)
            const auto src_out_neighbors = gm.get_out_neighbors(src);
            for (const auto& intermediate : src_out_neighbors) {
                const auto intermediate_out_neighbors = gm.get_out_neighbors(intermediate);
                for (const auto& dst : intermediate_out_neighbors) {
                    if (dst != src) { // Don't count back to source
                        two_hop_counts[dst]++;
                    }
                }
            }

            // Batch insert: store all 2-hop counts for this source at once
            for (const auto& [dst, count] : two_hop_counts) {
                m_transitive_path_cache.put(src, dst, count);
                ++total_paths;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  [Transitive Paths] " << elapsed << "ms -> cached " << total_paths
                  << " paths" << std::endl;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start)
            .count();
    std::cout << "[FeatureGenerator] Pre-computation complete: " << total_elapsed
              << "ms total" << std::endl;
}

// ============================================================================
// Graph Statistics for Normalization
// ============================================================================

template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies> && HasDirectionalForward<EmbeddingModule>
void FeatureGenerator<GraphTraits, EmbeddingModule, GroundTruthDependencies>::
    compute_graph_statistics(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const torch::Tensor& source_embeddings,
        const torch::Tensor& dest_embeddings) const {

    if (m_graph_stats.is_initialized) {
        return; // Already computed
    }

    std::cout << "[FeatureGenerator] Computing graph statistics for normalization..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Collect all degrees (for degree feature normalization)
    bool need_degree_stats = m_config.struct_in_degree_src || m_config.struct_out_degree_src ||
                             m_config.struct_in_degree_dst || m_config.struct_out_degree_dst;
    
    if (need_degree_stats) {
        std::vector<double> all_degrees;
        all_degrees.reserve(vertex_to_index.size() * 2); // in + out degrees
        
        for (const auto& [ip, vertex] : vertex_to_index) {
            all_degrees.push_back(std::log1p(static_cast<double>(m_graph_analytics.in_degree(vertex))));
            all_degrees.push_back(std::log1p(static_cast<double>(m_graph_analytics.out_degree(vertex))));
        }
        
        std::sort(all_degrees.begin(), all_degrees.end());
        m_graph_stats.sorted_degrees = std::move(all_degrees);
    }

    // Collect all embedding norms (for embedding norm normalization)
    bool need_emb_stats = m_config.emb_src_norm || m_config.emb_dst_norm;
    
    if (need_emb_stats && m_config.are_embedding_features_enabled()) {
        std::vector<double> all_norms;
        all_norms.reserve(source_embeddings.size(0) + dest_embeddings.size(0));
        
        auto src_accessor = source_embeddings.accessor<float, 2>();
        auto dst_accessor = dest_embeddings.accessor<float, 2>();
        
        for (int64_t i = 0; i < source_embeddings.size(0); ++i) {
            double src_norm = 0.0;
            double dst_norm = 0.0;
            
            for (int64_t j = 0; j < source_embeddings.size(1); ++j) {
                src_norm += src_accessor[i][j] * src_accessor[i][j];
                dst_norm += dst_accessor[i][j] * dst_accessor[i][j];
            }
            
            all_norms.push_back(std::sqrt(src_norm));
            all_norms.push_back(std::sqrt(dst_norm));
        }
        
        std::sort(all_norms.begin(), all_norms.end());
        m_graph_stats.sorted_embedding_norms = std::move(all_norms);
    }

    m_graph_stats.is_initialized = true;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[FeatureGenerator] Statistics computed in " << elapsed << "ms" << std::endl;
}

