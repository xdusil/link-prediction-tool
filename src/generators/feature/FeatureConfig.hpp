#pragma once

#include <cstddef>
#include <string>
#include <vector>

/**
 * @brief Feature configuration for directed link prediction.
 *
 * Features are directional: f(src, dst) captures the src -> dst dependency direction.
 * Both (u, v) and (v, u) are processed as separate samples.
 *
 * Feature groups:
 *   1. Directional Embedding Similarity - core backbone for dual-space models
 *   2. Embedding Asymmetry - role detection (caller vs callee)
 *   3. Hadamard Aggregates - multiplicative interactions without dimension explosion
 *   4. Structural Graph Features - topology and hierarchy signals
 *   5. Temporal Causality - request-response timing patterns
 *   6. Bidirectional Flow - communication asymmetry
 *   7. Protocol/Port Role - network-layer dependency signals
 */
class FeatureConfig {
public:
    // =========================================================================
    // 1. Directional Embedding Similarity Features
    // =========================================================================

    bool emb_dot_src_dst = true;    // dot(src[u], dst[v])
    bool emb_cosine_src_dst = true; // cosine(src[u], dst[v])
    bool emb_l1_src_dst = true;     // L1 distance between src[u] and dst[v]
    bool emb_l2_src_dst = true;     // L2 distance between src[u] and dst[v]

    // =========================================================================
    // 2. Embedding Asymmetry Features
    // =========================================================================

    bool emb_src_norm = true;   // ||src[u]|| - source embedding magnitude
    bool emb_dst_norm = true;   // ||dst[v]|| - destination embedding magnitude
    bool emb_norm_ratio = true; // ||src[u]|| / ||dst[v]|| - direction strength signal

    // =========================================================================
    // 3. Hadamard Aggregate Features
    // =========================================================================

    bool emb_hadamard_sum = true;  // sum(src[u] * dst[v])
    bool emb_hadamard_mean = true; // mean(src[u] * dst[v])

    // =========================================================================
    // 4. Structural Directed Graph Features
    // =========================================================================

    bool struct_in_degree_src = true;  // in-degree of source node u
    bool struct_out_degree_src = true; // out-degree of source node u
    bool struct_in_degree_dst = true;  // in-degree of destination node v
    bool struct_out_degree_dst = true; // out-degree of destination node v
    bool struct_degree_ratio = true;   // out(u) / in(v) - dependency direction signal

    bool struct_common_neighbors = true;        // normalized common neighbor count
    bool struct_jaccard_coefficient = true;     // Jaccard coefficient
    bool struct_adamic_adar_index = true;       // Adamic-Adar index
    bool struct_preferential_attachment = true; // Preferential attachment score
    bool struct_resource_allocation = true;     // Resource allocation index
    bool struct_transitive_reachability = true; // # of 2-hop paths u -> ... -> v
    bool struct_shortest_path = true;           // directed shortest path length
    bool struct_hierarchy_diff = true;          // hierarchy_level(v) - hierarchy_level(u)

    // =========================================================================
    // 5. Temporal Causality Features
    // =========================================================================

    bool time_avg_duration = true;     // average flow duration
    bool time_avg_interarrival = true; // average inter-arrival time
    bool time_regularity = true;       // coefficient of variation (periodic vs burst)
    bool time_direction_bias = true;   // (forward - reverse) / total flows
    bool time_initiation_order = true; // who initiates communication first
    bool time_crosscorr_peak = true;   // max cross-correlation and lag (causal signal)
    bool time_spike_score = true;      // delay distribution sharpness (tight causality)

    // =========================================================================
    // 6. Bidirectional Flow Asymmetry Features
    // =========================================================================

    bool flow_response_time = true;       // average response latency
    bool flow_request_ratio = true;       // forward_count / reverse_count
    bool flow_direction_asymmetry = true; // combined count + duration asymmetry
    bool flow_causality_score = true;     // temporal ordering pattern score

    // =========================================================================
    // 7. Protocol / Port Role Features
    // =========================================================================

    bool net_protocol_role = true; // TCP/UDP initiator detection
    bool net_port_role = true;     // server port vs ephemeral client port

    // =========================================================================

    constexpr FeatureConfig() = default;

    constexpr bool are_embedding_features_enabled() const noexcept {
        return emb_dot_src_dst || emb_cosine_src_dst || emb_l1_src_dst ||
               emb_l2_src_dst || emb_src_norm || emb_dst_norm || emb_norm_ratio ||
               emb_hadamard_sum || emb_hadamard_mean;
    }

    constexpr bool are_structural_features_enabled() const noexcept {
        return struct_in_degree_src || struct_out_degree_src || struct_in_degree_dst ||
               struct_out_degree_dst || struct_degree_ratio || struct_common_neighbors ||
               struct_jaccard_coefficient || struct_adamic_adar_index ||
               struct_preferential_attachment || struct_resource_allocation ||
               struct_transitive_reachability || struct_shortest_path ||
               struct_hierarchy_diff;
    }

    constexpr bool are_temporal_features_enabled() const noexcept {
        return time_avg_duration || time_avg_interarrival || time_regularity ||
               time_direction_bias || time_initiation_order || time_crosscorr_peak ||
               time_spike_score;
    }

    constexpr bool are_flow_features_enabled() const noexcept {
        return flow_response_time || flow_request_ratio || flow_direction_asymmetry ||
               flow_causality_score;
    }

    constexpr bool are_network_features_enabled() const noexcept {
        return net_protocol_role || net_port_role;
    }

    constexpr bool is_any_feature_enabled() const noexcept {
        return are_embedding_features_enabled() || are_structural_features_enabled() ||
               are_temporal_features_enabled() || are_flow_features_enabled() ||
               are_network_features_enabled();
    }

    constexpr std::size_t get_dimension() const noexcept {
        std::size_t dim = 0;

        // 1. Directional Embedding Similarity (4 features)
        dim += emb_dot_src_dst ? 1 : 0;
        dim += emb_cosine_src_dst ? 1 : 0;
        dim += emb_l1_src_dst ? 1 : 0;
        dim += emb_l2_src_dst ? 1 : 0;

        // 2. Embedding Asymmetry (3 features)
        dim += emb_src_norm ? 1 : 0;
        dim += emb_dst_norm ? 1 : 0;
        dim += emb_norm_ratio ? 1 : 0;

        // 3. Hadamard Aggregates (2 features)
        dim += emb_hadamard_sum ? 1 : 0;
        dim += emb_hadamard_mean ? 1 : 0;

        // 4. Structural Graph Features (13 features)
        dim += struct_in_degree_src ? 1 : 0;
        dim += struct_out_degree_src ? 1 : 0;
        dim += struct_in_degree_dst ? 1 : 0;
        dim += struct_out_degree_dst ? 1 : 0;
        dim += struct_degree_ratio ? 1 : 0;
        dim += struct_common_neighbors ? 1 : 0;
        dim += struct_jaccard_coefficient ? 1 : 0;
        dim += struct_adamic_adar_index ? 1 : 0;
        dim += struct_preferential_attachment ? 1 : 0;
        dim += struct_resource_allocation ? 1 : 0;
        dim += struct_transitive_reachability ? 1 : 0;
        dim += struct_shortest_path ? 1 : 0;
        dim += struct_hierarchy_diff ? 1 : 0;

        // 5. Temporal Causality (8 features - crosscorr gives 2)
        dim += time_avg_duration ? 1 : 0;
        dim += time_avg_interarrival ? 1 : 0;
        dim += time_regularity ? 1 : 0;
        dim += time_direction_bias ? 1 : 0;
        dim += time_initiation_order ? 1 : 0;
        dim += time_crosscorr_peak ? 2 : 0; // max_corr + lag
        dim += time_spike_score ? 1 : 0;

        // 6. Bidirectional Flow (4 features)
        dim += flow_response_time ? 1 : 0;
        dim += flow_request_ratio ? 1 : 0;
        dim += flow_direction_asymmetry ? 1 : 0;
        dim += flow_causality_score ? 1 : 0;

        // 7. Protocol/Port Role (2 features)
        dim += net_protocol_role ? 1 : 0;
        dim += net_port_role ? 1 : 0;

        return dim;
    }

    std::vector<std::string> get_feature_names() const {
        std::vector<std::string> names;
        names.reserve(get_dimension());

        // 1. Directional Embedding Similarity
        if (emb_dot_src_dst)
            names.push_back("emb_dot_src_dst");
        if (emb_cosine_src_dst)
            names.push_back("emb_cosine_src_dst");
        if (emb_l1_src_dst)
            names.push_back("emb_l1_src_dst");
        if (emb_l2_src_dst)
            names.push_back("emb_l2_src_dst");

        // 2. Embedding Asymmetry
        if (emb_src_norm)
            names.push_back("emb_src_norm");
        if (emb_dst_norm)
            names.push_back("emb_dst_norm");
        if (emb_norm_ratio)
            names.push_back("emb_norm_ratio");

        // 3. Hadamard Aggregates
        if (emb_hadamard_sum)
            names.push_back("emb_hadamard_sum");
        if (emb_hadamard_mean)
            names.push_back("emb_hadamard_mean");

        // 4. Structural Graph Features
        if (struct_in_degree_src)
            names.push_back("struct_in_degree_src");
        if (struct_out_degree_src)
            names.push_back("struct_out_degree_src");
        if (struct_in_degree_dst)
            names.push_back("struct_in_degree_dst");
        if (struct_out_degree_dst)
            names.push_back("struct_out_degree_dst");
        if (struct_degree_ratio)
            names.push_back("struct_degree_ratio");
        if (struct_common_neighbors)
            names.push_back("struct_common_neighbors");
        if (struct_jaccard_coefficient)
            names.push_back("struct_jaccard_coefficient");
        if (struct_adamic_adar_index)
            names.push_back("struct_adamic_adar_index");
        if (struct_preferential_attachment)
            names.push_back("struct_preferential_attachment");
        if (struct_resource_allocation)
            names.push_back("struct_resource_allocation");
        if (struct_transitive_reachability)
            names.push_back("struct_transitive_reachability");
        if (struct_shortest_path)
            names.push_back("struct_shortest_path");
        if (struct_hierarchy_diff)
            names.push_back("struct_hierarchy_diff");

        // 5. Temporal Causality
        if (time_avg_duration)
            names.push_back("time_avg_duration");
        if (time_avg_interarrival)
            names.push_back("time_avg_interarrival");
        if (time_regularity)
            names.push_back("time_regularity");
        if (time_direction_bias)
            names.push_back("time_direction_bias");
        if (time_initiation_order)
            names.push_back("time_initiation_order");
        if (time_crosscorr_peak) {
            names.push_back("time_crosscorr_max");
            names.push_back("time_crosscorr_lag");
        }
        if (time_spike_score)
            names.push_back("time_spike_score");

        // 6. Bidirectional Flow
        if (flow_response_time)
            names.push_back("flow_response_time");
        if (flow_request_ratio)
            names.push_back("flow_request_ratio");
        if (flow_direction_asymmetry)
            names.push_back("flow_direction_asymmetry");
        if (flow_causality_score)
            names.push_back("flow_causality_score");

        // 7. Protocol/Port Role
        if (net_protocol_role)
            names.push_back("net_protocol_role");
        if (net_port_role)
            names.push_back("net_port_role");

        return names;
    }
};
