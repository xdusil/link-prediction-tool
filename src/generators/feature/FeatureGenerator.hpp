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
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *
     * @note Since features are symmetric, each pair is processed once (avoiding duplicates).
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
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since features are symmetric, each pair is processed once (avoiding duplicates).
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
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     *
     * @note Since features are symmetric, each pair is processed once (avoiding duplicates).
     */
    std::tuple<torch::Tensor, std::vector<std::pair<IPAddress, IPAddress>>>
    generate_unlabeled_features_with_pairs(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index) override;

private:
    const IGraphAnalytics<GraphTraits> &m_graph_analytics;
    const EmbeddingModule &m_embedding_module;
    FeatureConfig m_feature_config;
    std::optional<double> m_avg_degree;  // Cached graph statistics to avoid recalculation

    /**
     * @brief Generate feature tensors with corresponding labels, and vertex pairs.
     *
     * @tparam WithLabels Whether to include labels.
     * @tparam WithVertexPairs Whether to include vertex pairs.
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The tuple of features, labels, and vertex pairs.
     *        - features: tensor of shape [num_pairs, feature_dim] where num_pairs = n*(n-1)/2
     *        - labels: row vector of size num_pairs (1=dependency, 0=no dependency)
     *        - vertex_pairs: vector of IP address pairs corresponding to each feature
     * 
     * @note Since all features are commutative (f(v1,v2) = f(v2,v1)), each pair is 
     * processed only once. The implementation avoids duplicates by only processing 
     * pairs where idx1 < idx2. However, when checking for dependencies, both (ip1,ip2) 
     * and (ip2,ip1) are considered because dependencies can be stored in either direction.
     */
    template <bool WithLabels = true, bool WithVertexPairs = false>
    std::tuple<torch::Tensor, arma::Row<size_t>,
               std::vector<std::pair<IPAddress, IPAddress>>>
    generate_features_and_labels_impl(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const GroundTruthDependencies &ground_truth_dependencies);

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