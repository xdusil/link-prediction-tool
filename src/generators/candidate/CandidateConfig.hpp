#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace candidate {

enum class CandidateMode {
    AllPairs,
    ObservedOrTwoHop
};

inline std::string_view to_string(CandidateMode mode) {
    switch (mode) {
    case CandidateMode::AllPairs:
        return "all_pairs";
    case CandidateMode::ObservedOrTwoHop:
        return "observed_or_two_hop";
    }

    return "unknown";
}

inline std::optional<CandidateMode> candidate_mode_from_string(
    std::string_view mode) {
    if (mode == "all_pairs") {
        return CandidateMode::AllPairs;
    }
    if (mode == "observed_or_two_hop") {
        return CandidateMode::ObservedOrTwoHop;
    }

    return std::nullopt;
}

/**
 * @brief Configures which IP pairs are sent to the feature generator and classifier.
 *
 * Candidate generation controls the evaluation universe only.
 */
struct CandidateConfig {
    // Valid modes: all_pairs, observed_or_two_hop.
    CandidateMode mode = CandidateMode::AllPairs;

    // Minimum graph support required by observed_or_two_hop mode.
    std::size_t min_direct_edge_support = 2;
    std::size_t min_common_neighbor_support = 1;

    // Optional truncation after the selected mode has generated candidates.
    std::optional<std::size_t> max_candidates_per_source;
};

} // namespace candidate
