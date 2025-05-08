#pragma once

#include "IFeatureGenerator.hpp"
#include "generators/feature/FeatureConfig.hpp"
#include "graph/IGraphAnalytics.hpp"
#include "graph/network/NetworkGraphManager.hpp"

/**
 * @brief Class for generating features.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 * @tparam EmbeddingModule The type of the embedding module.
 * @tparam GroundTruthDependencies The type of the ground truth dependencies.
 */
template <typename GraphTraits, typename EmbeddingModule,
          typename GroundTruthDependencies>
class FeatureGenerator
    : public IFeatureGenerator<typename GraphTraits::Vertex, EmbeddingModule,
                               GroundTruthDependencies> {
public:
    /**
     * @brief Construct a new Feature Generator object.
     *
     * @param graph_analytics The graph analytics.
     * @param embedding_module The embedding module.
     * @param config The feature configuration.
     */
    FeatureGenerator(const IGraphAnalytics<GraphTraits> &graph_analytics,
                     const EmbeddingModule &embedding_module,
                     const FeatureConfig &config = FeatureConfig());

    /**
     * @brief Generate feature tensors with corresponding labels.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of the features and labels.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs =
     * n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *
     * @note Since features are symmetric, each pair is processed once (avoiding
     * duplicates).
     */
    std::tuple<torch::Tensor, arma::Row<size_t>> generate_labeled_features(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) override;

    /**
     * @brief Generate feature tensors with corresponding labels and vertex pairs.
     *
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of features, labels, and vertex pairs.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs =
     * n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since features are symmetric, each pair is processed once (avoiding
     * duplicates).
     */
    std::tuple<torch::Tensor, arma::Row<size_t>,
               std::vector<std::pair<IPAddress, IPAddress>>>
    generate_labeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies) override;

    /**
     * @brief Generate feature tensors for prediction with corresponding vertex pairs.
     *
     * @param vertex_to_index The map of vertex to index.
     * @return The tuple of features and vertex pairs.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs =
     * n*(n-1)/2
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since features are symmetric, each pair is processed once (avoiding
     * duplicates).
     */
    std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index) override;

private:
    const IGraphAnalytics<GraphTraits> &m_graph_analytics;
    const EmbeddingModule &m_embedding_module;
    FeatureConfig m_feature_config;
    std::optional<double> m_avg_degree; // Cached graph statistics to avoid recalculation

    /**
     * @brief Generate feature tensors with corresponding labels, and vertex pairs.
     *
     * @tparam WithLabels Whether to include labels.
     * @tparam WithVertexPairs Whether to include vertex pairs.
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of features, labels, and vertex pairs.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs =
     * n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since all features are commutative (f(v1,v2) = f(v2,v1)), each pair is
     * processed only once. The implementation avoids duplicates by only processing
     * pairs where idx1 < idx2. However, when checking for dependencies, both (ip1,ip2)
     * and (ip2,ip1) are considered because dependencies can be stored in either
     * direction.
     */
    template <bool WithLabels = true, bool WithVertexPairs = false>
    std::tuple<torch::Tensor, arma::Row<size_t>,
               std::vector<std::pair<IPAddress, IPAddress>>>
    generate_features_and_labels_impl(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies);

    /**
     * @brief Extracts and sorts vertices from the vertex_to_index map.
     *
     * @param vertex_to_index The map of vertex IPs to their indices.
     * @return A pair of vectors containing sorted vertex IPs and their corresponding
     * indices.
     */
    std::pair<std::vector<IPAddress>, std::vector<int64_t>>
    extract_sorted_vertices(const std::unordered_map<IPAddress, Vertex> &vertex_to_index);

    /**
     * @brief Process vertex pairs to generate features and labels.
     *
     * @tparam WithLabels Whether to include labels in the output.
     * @tparam WithVertexPairs Whether to include vertex pairs in the output.
     * @param vertex_ips The sorted vector of vertex IPs.
     * @param all_embeddings The tensor containing all embeddings.
     * @param vertex_to_index The map of vertex IPs to their indices.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @param all_features The tensor to store generated features.
     * @param arma_labels The vector to store labels (if applicable).
     * @param vertex_pairs The vector to store vertex pairs (if applicable).
     */
    template <bool WithLabels, bool WithVertexPairs>
    void
    process_vertex_pairs(const std::vector<IPAddress> &vertex_ips,
                         const torch::Tensor &all_embeddings,
                         const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
                         const GroundTruthDependencies &ground_truth_dependencies,
                         torch::Tensor &all_features, arma::Row<size_t> &arma_labels,
                         std::vector<std::pair<IPAddress, IPAddress>> &vertex_pairs);

    /**
     * @brief Add similarity features to the feature tensor.
     *
     * @tparam T The type of elements in the tensor.
     * @param v1_emb The embedding of the first vertex.
     * @param v2_emb The embedding of the second vertex.
     * @param accessor The tensor accessor for the feature tensor.
     * @param row The row index in the feature tensor.
     * @param col The column index in the feature tensor.
     * @return The updated column index after adding the features.
     */
    template <typename T>
    std::size_t add_similarity_features(const torch::Tensor &v1_emb,
                                        const torch::Tensor &v2_emb,
                                        torch::TensorAccessor<T, 2> &accessor,
                                        std::size_t row, std::size_t col);

    /**
     * @brief Add statistical features to the feature tensor.
     *
     * @tparam T The type of elements in the tensor.
     * @param v1_emb The embedding of the first vertex.
     * @param v2_emb The embedding of the second vertex.
     * @param accessor The tensor accessor for the feature tensor.
     * @param row The row index in the feature tensor.
     * @param col The column index in the feature tensor.
     * @return The updated column index after adding the features.
     */
    template <typename T>
    std::size_t add_statistical_features(const torch::Tensor &v1_emb,
                                         const torch::Tensor &v2_emb,
                                         torch::TensorAccessor<T, 2> &accessor,
                                         std::size_t row, std::size_t col);

    /**
     * @brief Add hadamard features to the feature tensor.
     *
     * @tparam T The type of elements in the tensor.
     * @param v1_emb The embedding of the first vertex.
     * @param v2_emb The embedding of the second vertex.
     * @param accessor The tensor accessor for the feature tensor.
     * @param row The row index in the feature tensor.
     * @param col The column index in the feature tensor.
     * @return The updated column index after adding the features.
     */
    template <typename T>
    std::size_t add_hadamard_features(const torch::Tensor &v1_emb,
                                      const torch::Tensor &v2_emb,
                                      torch::TensorAccessor<T, 2> &accessor,
                                      std::size_t row, std::size_t col);

    /**
     * @brief Add network features to the feature tensor.
     *
     * @tparam T The type of elements in the tensor.
     * @param v1 The first vertex.
     * @param v2 The second vertex.
     * @param accessor The tensor accessor for the feature tensor.
     * @param row The row index in the feature tensor.
     * @param col The column index in the feature tensor.
     * @return The updated column index after adding the features.
     */
    template <typename T>
    std::size_t add_network_features(Vertex v1, Vertex v2,
                                     torch::TensorAccessor<T, 2> &accessor,
                                     std::size_t row, std::size_t col);

    /**
     * @brief Add node features to the feature tensor.
     *
     * @tparam T The type of elements in the tensor.
     * @param v1 The first vertex.
     * @param v2 The second vertex.
     * @param accessor The tensor accessor for the feature tensor.
     * @param row The row index in the feature tensor.
     * @param col The column index in the feature tensor.
     * @return The updated column index after adding the features.
     */
    template <typename T>
    std::size_t add_node_features(Vertex v1, Vertex v2,
                                  torch::TensorAccessor<T, 2> &accessor, std::size_t row,
                                  std::size_t col);
    /**
     * @brief Calculate cosine similarity between two tensors.
     *
     * @tparam T The type of elements in the tensor.
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The cosine similarity.
     */
    template <typename T>
    T cosine_similarity(const torch::Tensor &a, const torch::Tensor &b);
    /**
     * @brief Calculate Euclidean distance between two tensors.
     *
     * @tparam T The type of elements in the tensor.
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The Euclidean distance.
     */
    template <typename T>
    T euclidean_distance(const torch::Tensor &a, const torch::Tensor &b);

    /**
     * @brief Calculate dot product between two tensors.
     *
     * @tparam T The type of elements in the tensor.
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The dot product.
     */
    template <typename T>
    T dot_product(const torch::Tensor &a, const torch::Tensor &b);

    /**
     * @brief Get the average degree of the graph and cache it.
     *
     * @return The average degree.
     */
    template <typename T>
    T get_set_avg_degree();

    /**
     * @brief Create features and set them to a tensor.
     *
     * @tparam T The type of elements in the tensor.
     * @param v1 The first vertex.
     * @param v2 The second vertex.
     * @param v1_emb The embedding of the first vertex.
     * @param v2_emb The embedding of the second vertex.
     * @param features_tensor The tensor to set the features to.
     * @param row_index The row index in the tensor.
     */
    template <typename T>
    void create_features_and_set_to_tensor(Vertex v1, Vertex v2,
                                           const torch::Tensor &v1_emb,
                                           const torch::Tensor &v2_emb,
                                           torch::Tensor &features_tensor,
                                           std::size_t row_index);
};

#include "FeatureGenerator.tpp"