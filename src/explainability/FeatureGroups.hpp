#pragma once

#include <array>
#include <string_view>

namespace explainability {

inline constexpr std::string_view EMBEDDING_FEATURE_PREFIX = "emb_";
inline constexpr std::string_view TOPOLOGY_FEATURE_PREFIX = "struct_";
inline constexpr std::string_view TEMPORAL_FEATURE_PREFIX = "time_";
inline constexpr std::string_view OBSERVED_FLOW_FEATURE_PREFIX = "flow_";
inline constexpr std::string_view NETWORK_FEATURE_PREFIX = "net_";

/**
 * @brief Metadata for one explainability feature group.
 *
 * Features are assigned to a group by matching their configured feature-name
 * prefix. The text is intentionally human-facing and is written to explanation
 * artifacts.
 */
struct FeatureGroup {
    std::string_view name;
    std::string_view prefix;
    std::string_view explanation;
};

/**
 * @brief Feature groups currently used by local ablation explanations.
 *
 * Keep prefixes synchronized with FeatureConfig feature names. If a new feature
 * family is added without one of these prefixes, local ablation rejects the
 * active feature configuration.
 */
inline constexpr std::array<FeatureGroup, 5> FEATURE_GROUPS = {{
    {"embedding",
     EMBEDDING_FEATURE_PREFIX,
     "evidence from learned source and destination graph embeddings"},
    {"topology",
     TOPOLOGY_FEATURE_PREFIX,
     "evidence from the retained graph structure"},
    {"temporal",
     TEMPORAL_FEATURE_PREFIX,
     "evidence from timing patterns in observed flows"},
    {"observed_flow",
     OBSERVED_FLOW_FEATURE_PREFIX,
     "evidence from observed flow direction and volume"},
    {"network",
     NETWORK_FEATURE_PREFIX,
     "evidence from protocol and port roles"},
}};

/**
 * @brief Returns true when value starts with prefix.
 */
inline bool has_prefix(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

/**
 * @brief Returns true when a feature name belongs to any known group.
 */
inline bool belongs_to_known_group(std::string_view feature_name) {
    for (const FeatureGroup& group : FEATURE_GROUPS) {
        if (has_prefix(feature_name, group.prefix)) {
            return true;
        }
    }

    return false;
}

} // namespace explainability
