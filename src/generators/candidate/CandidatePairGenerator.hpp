#pragma once

#include "CandidateConfig.hpp"
#include "CandidatePair.hpp"
#include "graph/IGraphAnalytics.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace candidate {

/**
 * @brief Builds the directed IP pair universe for training or prediction.
 *
 * If max_candidates_per_source is set, candidates for each source are ranked by
 * direct-flow support, then by common-neighbor support, then by destination IP.
 */
template <typename GraphTraits>
class CandidatePairGenerator {
public:
    using Vertex = typename GraphTraits::Vertex;

    std::vector<CandidatePair>
    generate(const IGraphAnalytics<GraphTraits>& analytics,
             const std::unordered_map<IPAddress, Vertex>& ip_to_vertex,
             const CandidateConfig& config) const {

        std::vector<VertexEntry> vertices;
        vertices.reserve(ip_to_vertex.size());
        for (const std::pair<const IPAddress, Vertex>& entry : ip_to_vertex) {
            vertices.emplace_back(entry.first, entry.second);
        }
        std::sort(vertices.begin(), vertices.end(),
                  [](const VertexEntry& lhs, const VertexEntry& rhs) {
                      return lhs.first < rhs.first;
                  });

        if (config.mode == CandidateMode::AllPairs &&
            !config.max_candidates_per_source.has_value()) {
            return generate_all_pairs(vertices);
        }

        std::vector<CandidatePair> candidates;
        candidates.reserve(
            vertices.size() *
            std::min(vertices.size(), DEFAULT_EXPECTED_CANDIDATES_PER_SOURCE));
        const bool has_per_source_limit = config.max_candidates_per_source.has_value();

        for (const VertexEntry& src_entry : vertices) {
            std::vector<CandidateWithEvidence> source_candidates;

            for (const VertexEntry& dst_entry : vertices) {
                if (src_entry.second == dst_entry.second) {
                    continue;
                }

                const Evidence evidence = collect_evidence(analytics, src_entry.second,
                                                           dst_entry.second, config);

                if (!accept_candidate(config.mode, evidence, config)) {
                    continue;
                }

                CandidatePair candidate{src_entry.first, dst_entry.first};
                if (has_per_source_limit) {
                    source_candidates.push_back({std::move(candidate), evidence});
                } else {
                    candidates.push_back(std::move(candidate));
                }
            }

            if (!has_per_source_limit) {
                continue;
            }

            std::sort(
                source_candidates.begin(), source_candidates.end(),
                [](const CandidateWithEvidence& lhs, const CandidateWithEvidence& rhs) {
                    if (lhs.evidence.direct_flows != rhs.evidence.direct_flows) {
                        return lhs.evidence.direct_flows > rhs.evidence.direct_flows;
                    }
                    if (lhs.evidence.common_neighbors != rhs.evidence.common_neighbors) {
                        return lhs.evidence.common_neighbors >
                               rhs.evidence.common_neighbors;
                    }
                    return lhs.pair.dst < rhs.pair.dst;
                });

            if (source_candidates.size() > *config.max_candidates_per_source) {
                source_candidates.resize(*config.max_candidates_per_source);
            }

            for (CandidateWithEvidence& candidate : source_candidates) {
                candidates.push_back(std::move(candidate.pair));
            }
        }

        if (has_per_source_limit) {
            std::sort(candidates.begin(), candidates.end(),
                      [](const CandidatePair& lhs, const CandidatePair& rhs) {
                          return std::tie(lhs.src, lhs.dst) <
                                 std::tie(rhs.src, rhs.dst);
                      });
        }

        return candidates;
    }

private:
    static constexpr std::size_t DEFAULT_EXPECTED_CANDIDATES_PER_SOURCE = 32;

    using VertexEntry = std::pair<IPAddress, Vertex>;

    struct Evidence {
        std::size_t direct_flows = 0;
        std::size_t common_neighbors = 0;
    };

    struct CandidateWithEvidence {
        CandidatePair pair;
        Evidence evidence;
    };

    static std::vector<CandidatePair>
    generate_all_pairs(const std::vector<VertexEntry>& vertices) {
        std::vector<CandidatePair> candidates;
        candidates.reserve(directed_pair_count(vertices.size()));

        for (const VertexEntry& src_entry : vertices) {
            for (const VertexEntry& dst_entry : vertices) {
                if (src_entry.second != dst_entry.second) {
                    candidates.push_back({src_entry.first, dst_entry.first});
                }
            }
        }

        return candidates;
    }

    static std::size_t directed_pair_count(std::size_t vertex_count) {
        return vertex_count < 2 ? 0 : vertex_count * (vertex_count - 1);
    }

    static Evidence collect_evidence(const IGraphAnalytics<GraphTraits>& analytics,
                                     Vertex src, Vertex dst,
                                     const CandidateConfig& config) {
        Evidence evidence;

        if (needs_common_neighbors(config)) {
            evidence.common_neighbors = analytics.common_neighbors_count(src, dst);
        }

        if (needs_direct_flows(config)) {
            const auto& graph_manager = analytics.get_graph_manager();
            const auto [begin, end] = graph_manager.get_out_edges(src);
            for (auto it = begin; it != end; ++it) {
                if (graph_manager.get_target_vertex(*it) == dst) {
                    ++evidence.direct_flows;
                }
            }
        }

        return evidence;
    }

    static bool accept_candidate(CandidateMode mode, const Evidence& evidence,
                                 const CandidateConfig& config) {
        const bool direct = evidence.direct_flows >= config.min_direct_edge_support;
        const bool two_hop =
            evidence.common_neighbors >= config.min_common_neighbor_support;

        switch (mode) {
        case CandidateMode::AllPairs:
            return true;
        case CandidateMode::ObservedOrTwoHop:
            return direct || two_hop;
        }

        return false;
    }

    static bool needs_direct_flows(const CandidateConfig& config) {
        return config.mode == CandidateMode::ObservedOrTwoHop ||
               config.max_candidates_per_source.has_value();
    }

    static bool needs_common_neighbors(const CandidateConfig& config) {
        return config.mode == CandidateMode::ObservedOrTwoHop ||
               config.max_candidates_per_source.has_value();
    }
};

} // namespace candidate
