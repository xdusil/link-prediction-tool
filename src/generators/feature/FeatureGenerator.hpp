#pragma once

#include "BidirectionalFeatureExtractor.hpp"
#include "FeatureConfig.hpp"
#include "IFeatureGenerator.hpp"
#include "ProtocolFeatureExtractor.hpp"
#include "TemporalFeatureExtractor.hpp"
#include "graph/IGraphAnalytics.hpp"
#include "graph/network/INetworkGraphManager.hpp"
#include "utils/cache/FeatureCache.hpp"
#include "utils/math/TensorMath.hpp"
#include "utils/timers/FeatureProfiler.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>
#include <queue>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/**
 * @brief Feature generator for directed link prediction.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 * @tparam EmbeddingModule The type of the embedding module.
 * @tparam GroundTruthDependencies The type of the ground truth dependencies.
 */
template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
    requires HasContains<GroundTruthDependencies>
class FeatureGenerator
    : public IFeatureGenerator<typename GraphTraits::Vertex, GroundTruthDependencies> {
public:
    using Vertex = typename GraphTraits::Vertex;

    /**
     * @brief Construct a new Feature Generator object.
     *
     * @param graph_analytics The graph analytics.
     * @param embedding_module The embedding module.
     * @param config The feature configuration.
     */
    FeatureGenerator(const IGraphAnalytics<GraphTraits>& graph_analytics,
                     EmbeddingModule& embedding_module, const FeatureConfig& config = {});

    /**
     * @brief Generate feature tensors with corresponding labels.
     *
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @param ground_truth_dependencies Ground truth dependency pairs.
     * @return Tuple of (features, labels):
     *         - features: tensor [num_pairs, feature_dim]
     *         - labels: row vector [num_pairs] (1=dependency, 0=no dependency)
     */
    std::tuple<torch::Tensor, arma::Row<std::size_t>> generate_labeled_features(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const GroundTruthDependencies& ground_truth_dependencies) override;

    /**
     * @brief Generate feature tensors with labels and vertex pairs.
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @param ground_truth_dependencies Ground truth dependency pairs.
     * @return Tuple of (features, labels, vertex_pairs):
     *         - features: tensor [num_pairs, feature_dim]
     *         - labels: row vector [num_pairs] (1=dependency, 0=no dependency)
     *         - vertex_pairs: vector [num_pairs] of (src_ip, dst_ip) pairs
     */
    std::tuple<torch::Tensor, arma::Row<std::size_t>,
               std::vector<std::pair<IPAddress, IPAddress>>>
    generate_labeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
        const GroundTruthDependencies& ground_truth_dependencies) override;

    /**
     * @brief Generate feature tensors for prediction (no labels).
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @return Tuple of (features, vertex_pairs):
     *         - features: tensor [num_pairs, feature_dim]
     *         - vertex_pairs: vector [num_pairs] of (src_ip, dst_ip) pairs
     */
    std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index) override;

private:
    const IGraphAnalytics<GraphTraits>& m_graph_analytics;
    EmbeddingModule& m_embedding_module;
    FeatureConfig m_config;
    mutable FeatureProfiler m_profiler;

    // Caches for expensive graph computations
    mutable FeatureCache<Vertex, std::size_t> m_shortest_path_cache;
    mutable FeatureCache<Vertex, std::size_t> m_transitive_path_cache;

    /**
     * @brief Core implementation for feature generation.
     *
     * @tparam WithLabels Whether to generate labels.
     * @tparam WithPairs Whether to generate vertex pairs.
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @param ground_truth Ground truth dependency pairs.
     * @return Tuple of (features, labels, vertex_pairs) depending on template params.
     */
    template <bool WithLabels, bool WithPairs>
    std::tuple<torch::Tensor, arma::Row<std::size_t>,
               std::vector<std::pair<IPAddress, IPAddress>>>
    generate_impl(const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
                  const GroundTruthDependencies& ground_truth);

    /**
     * @brief Process all vertex pairs to generate features, labels, and pairs.
     *
     * @tparam WithLabels Whether to generate labels.
     * @tparam WithPairs Whether to generate vertex pairs.
     * @param vertex_ips Vector of vertex IP addresses [N]
     * @param source_embeddings Tensor of source node embeddings [N, embedding_dim].
     * @param dest_embeddings Tensor of destination node embeddings [N, embedding_dim].
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @param ground_truth Ground truth dependency pairs.
     * @param features Output tensor for generated features - must already be of shape
     * [N*(N-1), feature_dim].
     * @param labels Output vector for generated labels - must already be of size
     * [N*(N-1)].
     * @param pairs Output vector for generated vertex pairs - must already be of size
     * [N*(N-1)].
     */
    template <bool WithLabels, bool WithPairs>
    void process_all_pairs(const std::vector<IPAddress>& vertex_ips,
                           const torch::Tensor& source_embeddings,
                           const torch::Tensor& dest_embeddings,
                           const std::unordered_map<IPAddress, Vertex>& vertex_to_index,
                           const GroundTruthDependencies& ground_truth,
                           torch::Tensor& features, arma::Row<std::size_t>& labels,
                           std::vector<std::pair<IPAddress, IPAddress>>& pairs);
    /**
     * @brief Prepare sorted vertex IPs and their corresponding indices.
     *
     * Sorts the IP addresses in ascending order.
     *
     * @param vertex_to_index Mapping from IP addresses to vertex indices.
     * @return Pair of (sorted IP addresses, corresponding vertex indices).
     */
    std::pair<std::vector<IPAddress>, std::vector<int64_t>> prepare_sorted_vertices(
        const std::unordered_map<IPAddress, Vertex>& vertex_to_index) const;

    void precompute_expensive_features(const std::vector<Vertex>& vertices) const;

    // ========================================================================
    // Feature Writers
    // ========================================================================

    template <typename T>
    void write_pair_features(Vertex src, Vertex dst, const torch::Tensor& src_emb,
                             const torch::Tensor& dst_emb,
                             torch::TensorAccessor<T, 2>& accessor, std::size_t row);

    template <typename T>
    std::size_t write_embedding_similarity_features(const torch::Tensor& src_emb,
                                                    const torch::Tensor& dst_emb,
                                                    torch::TensorAccessor<T, 2>& accessor,
                                                    std::size_t row, std::size_t col);

    template <typename T>
    std::size_t write_embedding_asymmetry_features(const torch::Tensor& src_emb,
                                                   const torch::Tensor& dst_emb,
                                                   torch::TensorAccessor<T, 2>& accessor,
                                                   std::size_t row, std::size_t col);

    template <typename T>
    std::size_t write_hadamard_features(const torch::Tensor& src_emb,
                                        const torch::Tensor& dst_emb,
                                        torch::TensorAccessor<T, 2>& accessor,
                                        std::size_t row, std::size_t col);

    template <typename T>
    std::size_t write_structural_features(Vertex src, Vertex dst,
                                          torch::TensorAccessor<T, 2>& accessor,
                                          std::size_t row, std::size_t col);

    template <typename T>
    std::size_t write_temporal_features(Vertex src, Vertex dst,
                                        torch::TensorAccessor<T, 2>& accessor,
                                        std::size_t row, std::size_t col);

    template <typename T>
    std::size_t write_flow_features(Vertex src, Vertex dst,
                                    torch::TensorAccessor<T, 2>& accessor,
                                    std::size_t row, std::size_t col);

    template <typename T>
    std::size_t write_protocol_features(Vertex src, Vertex dst,
                                        torch::TensorAccessor<T, 2>& accessor,
                                        std::size_t row, std::size_t col);
};

#include "FeatureGenerator.tpp"