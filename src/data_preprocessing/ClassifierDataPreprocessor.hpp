#pragma once

#include "../Types.hpp"
#include "../model/IModel.hpp"
#include "Types.hpp"
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <cstdlib>
#include <torch/torch.h>
#include <unordered_set>
#include <vector>

/**
 * @brief ClassifierDataPreprocessor class that prepares data for the classifier.
 */
template <typename Embeddings = torch::nn::Embedding>
class ClassifierDataPreprocessor {
public:

    /**
     * @brief Construct a new Classifier Data Preprocessor object.
     *
     * @param embeddings The embeddings to use.
     */
    explicit ClassifierDataPreprocessor(const Embeddings &embeddings) : m_embeddings(embeddings) {}

    /**
     * @brief Prepare the data for the classifier training.
     *
     * @tparam Vertex The type of the vertex.
     * @param vertex_to_index The map of vertex to index.
     * @param ground_truth_dependencies The ground truth dependencies.
     * @return The pair of the features and labels.
     */
    template <typename Vertex>
    std::pair<arma::fmat, arma::Row<size_t>> prepare_data_for_classifier_training(
        const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
        const auto &ground_truth_dependencies) {
        const std::size_t num_vertices = vertex_to_index.size();
        const std::size_t num_pairs = num_vertices * num_vertices;

        const std::size_t embedding_dim =
            m_embeddings->forward(torch::tensor(static_cast<int64_t>(0))).numel();

        //  1) Gather all (v1, v2) in Tensors for a single batch forward
        // -----------------------------
        //   - all_v1: shape [num_pairs] of int64
        //   - all_v2: shape [num_pairs] of int64
        //   - We'll also keep track of (ip1, ip2) so we can set labels accordingly
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt64);
        torch::Tensor all_v1 = torch::empty({static_cast<long>(num_pairs)}, opts);
        torch::Tensor all_v2 = torch::empty({static_cast<long>(num_pairs)}, opts);

        // 2) Fill the Tensors and labels
        arma::Row<size_t> arma_labels(num_pairs, arma::fill::none);
        std::size_t i = 0;
        for (const auto &[ip1, v1] : vertex_to_index) {
            for (const auto &[ip2, v2] : vertex_to_index) {
                all_v1[i] = static_cast<int64_t>(v1);
                all_v2[i] = static_cast<int64_t>(v2);

                if (ground_truth_dependencies.contains({ip1, ip2})) {
                    arma_labels[i] = 1;
                } else {
                    arma_labels[i] = 0;
                }
                
                ++i;
            }
        }

        //  3) Single pass forward for all pairs: emb1, emb2 => combined
        // -----------------------------
        // shapes:
        //   emb1       => [num_pairs, embedding_dim]
        //   emb2       => [num_pairs, embedding_dim]
        //   combined   => [num_pairs, embedding_dim]
        auto emb1 = m_embeddings->forward(all_v1);
        auto emb2 = m_embeddings->forward(all_v2);
        auto combined = emb1 * emb2; // elementwise product

        // 4) Transpose the tensor
        // ----------------------------
        // shape:
        //   combined_transposed => [embedding_dim, num_pairs]
        auto combined_transposed = combined.transpose(0, 1).detach().cpu().contiguous();
        
        // Check if tensor contains float data
        assert(combined_transposed.scalar_type() == at::ScalarType::Float 
               && "Tensor must contain float data");
        
        float* combined_ptr = combined_transposed.template data_ptr<float>();

        // 5) Convert to Armadillo matrices
        arma::fmat arma_features(combined_ptr, embedding_dim, num_pairs);

        return {arma_features, arma_labels};
    }


private:
    Embeddings m_embeddings; // The embeddings to use
};
