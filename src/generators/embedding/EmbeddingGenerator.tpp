#pragma once

#include "EmbeddingGenerator.hpp"

template <typename Vertex, typename Dependencies, typename EmbeddingModule>
torch::Tensor EmbeddingGenerator<Vertex, Dependencies, EmbeddingModule>::generate_dependency_embeddings(
    const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
    const Dependencies &dependencies, EmbeddingModule &embedding_module) {
    auto [combined, _] =
        generate_dependency_embeddings_and_labels<false>(
            vertex_to_index, dependencies, embedding_module);
    return combined;
}

template <typename Vertex, typename Dependencies, typename EmbeddingModule>
std::tuple<torch::Tensor, arma::Row<size_t>>
EmbeddingGenerator<Vertex, Dependencies, EmbeddingModule>::generate_dependency_embeddings_and_labels(
    const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
    const Dependencies &dependencies, EmbeddingModule &embedding_module) {
    return generate_dependency_embeddings_and_labels<true>(
        vertex_to_index, dependencies, embedding_module);
}

template <typename Vertex, typename Dependencies, typename EmbeddingModule>
template <bool CreateLabels>
std::tuple<torch::Tensor, arma::Row<size_t>>
EmbeddingGenerator<Vertex, Dependencies, EmbeddingModule>::generate_dependency_embeddings_and_labels(
    const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
    const Dependencies &dependencies, EmbeddingModule &embedding_module) {

    //  1) Gather all (v1, v2) in Tensors for a single batch forward
    //   - v1 and v2 are the Vertices from the vertex_to_index map
    //   - num_pairs is the count of all possible pairs of vertices
    // -----------------------------
    // shapes:
    //   all_v1        => [num_pairs] of int64
    //   all_v2        => [num_pairs] of int64
    //   arma_labels   => [num_pairs] if CreateLabels is true, else [1]
    auto [all_v1, all_v2, arma_labels] =
        create_vertex_pairs_and_labels<CreateLabels>(
            vertex_to_index, dependencies);

    //  2) Single pass forward for all pairs: emb1, emb2 => combined
    // -----------------------------
    // shapes:
    //   emb1       => [num_pairs, embedding_dim]
    //   emb2       => [num_pairs, embedding_dim]
    //   combined   => [num_pairs, embedding_dim]
    auto emb1 = embedding_module->forward(all_v1);
    auto emb2 = embedding_module->forward(all_v2);
    auto combined = emb1 * emb2; // elementwise product

    return {combined, arma_labels};
}

template <typename Vertex, typename Dependencies, typename EmbeddingModule>
template <bool CreateLabels>
std::tuple<torch::Tensor, torch::Tensor, arma::Row<size_t>>
EmbeddingGenerator<Vertex, Dependencies, EmbeddingModule>::create_vertex_pairs_and_labels(
    const std::unordered_map<IPAddress, Vertex> &vertex_to_index,
    const Dependencies &ground_truth_dependencies /*= {} */) {
    const std::size_t num_vertices = vertex_to_index.size();
    const std::size_t num_pairs = num_vertices * num_vertices;

    // 1) Prepare the Tensors
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor all_v1 = torch::empty({static_cast<long>(num_pairs)}, opts);
    torch::Tensor all_v2 = torch::empty({static_cast<long>(num_pairs)}, opts);

    // 2) Fill the Tensors and labels
    arma::Row<size_t> arma_labels((CreateLabels) ? num_pairs : 1, arma::fill::none);
    std::size_t i = 0;
    for (const auto &[ip1, v1] : vertex_to_index) {
        for (const auto &[ip2, v2] : vertex_to_index) {
            all_v1[i] = static_cast<int64_t>(v1);
            all_v2[i] = static_cast<int64_t>(v2);

            if constexpr (CreateLabels) {
                if (ground_truth_dependencies.contains({ip1, ip2})) {
                    arma_labels[i] = 1;
                } else {
                    arma_labels[i] = 0;
                }
            }

            ++i;
        }
    }

    return {all_v1, all_v2, arma_labels};
}