#pragma once

#include "FeatureConfig.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>

/**
 * @brief Extractor for protocol and port role features.
 *
 * Analyzes network protocol usage and port patterns to infer device roles.
 */
class ProtocolFeatureExtractor {
public:
    /**
     * @brief Result structure containing protocol/port features.
     */
    struct Features {
        std::optional<double> protocol_role;
        std::optional<double> port_role;
        std::optional<double> top_port;
    };

    /**
     * @brief Extract protocol features for a vertex pair.
     *
     * @param graph_manager The graph manager instance.
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param config The feature configuration.
     * @return The extracted protocol features. If a feature is not computable,
     * its value will be std::nullopt.
     */
    template <typename GraphTraits>
    static Features extract(
        const IGraphManager<GraphTraits>& graph_manager,
        const typename GraphTraits::Vertex& src,
        const typename GraphTraits::Vertex& dst,
        const FeatureConfig& config);

private:
    // Well-known ports threshold
    static constexpr uint16_t WELL_KNOWN_PORT_MAX = 1024;
    static constexpr uint16_t REGISTERED_PORT_MAX = 49151;

    // Common protocol identifiers
    static constexpr uint8_t PROTO_TCP = 6;
    static constexpr uint8_t PROTO_UDP = 17;
    static constexpr uint8_t PROTO_ICMP = 1;

    static double classify_port_role(uint16_t port);
    static double classify_protocol_role(uint8_t protocol);
};

// ====================================================================================================

template <typename GraphTraits>
ProtocolFeatureExtractor::Features ProtocolFeatureExtractor::extract(
    const IGraphManager<GraphTraits>& graph_manager,
    const typename GraphTraits::Vertex& src,
    const typename GraphTraits::Vertex& dst,
    const FeatureConfig& config) {
    
    Features result;

    std::unordered_map<uint8_t, std::size_t> protocol_counts;
    std::unordered_map<uint16_t, std::size_t> src_port_counts;
    std::unordered_map<uint16_t, std::size_t> dst_port_counts;
    std::size_t total_edges = 0;

    // Collect from src -> dst edges
    {
        auto [it, end] = graph_manager.get_out_edges(src);
        for (; it != end; ++it) {
            if (graph_manager.get_target_vertex(*it) == dst) {
                const auto& edge = graph_manager.get_edge_properties(*it);

                if (edge.protocol >= 0) {
                    ++protocol_counts[static_cast<uint8_t>(edge.protocol)];
                }

                if (edge.src_port > 0) {
                    ++src_port_counts[static_cast<uint16_t>(edge.src_port)];
                }
                if (edge.dst_port > 0) {
                    ++dst_port_counts[static_cast<uint16_t>(edge.dst_port)];
                }

                ++total_edges;
            }
        }
    }

    // Collect from dst -> src edges
    {
        auto [it, end] = graph_manager.get_out_edges(dst);
        for (; it != end; ++it) {
            if (graph_manager.get_target_vertex(*it) == src) {
                const auto& edge = graph_manager.get_edge_properties(*it);

                if (edge.protocol >= 0) {
                    ++protocol_counts[static_cast<uint8_t>(edge.protocol)];
                }

                // Ports (swapped for reverse direction)
                if (edge.dst_port > 0) {
                    ++src_port_counts[static_cast<uint16_t>(edge.dst_port)];
                }
                if (edge.src_port > 0) {
                    ++dst_port_counts[static_cast<uint16_t>(edge.src_port)];
                }

                ++total_edges;
            }
        }
    }

    if (total_edges == 0) return result;

    // Protocol role
    if (config.net_protocol_role) {
        double weighted_role = 0.0;
        for (const auto& [proto, count] : protocol_counts) {
            weighted_role += classify_protocol_role(proto) * static_cast<double>(count);
        }
        result.protocol_role = weighted_role / static_cast<double>(total_edges);
    }

    // Port role
    if (config.net_port_role) {
        double src_role = 0.0;
        double dst_role = 0.0;
        std::size_t src_total = 0;
        std::size_t dst_total = 0;

        for (const auto& [port, count] : src_port_counts) {
            src_role += classify_port_role(port) * static_cast<double>(count);
            src_total += count;
        }
        for (const auto& [port, count] : dst_port_counts) {
            dst_role += classify_port_role(port) * static_cast<double>(count);
            dst_total += count;
        }

        if (src_total > 0) src_role /= static_cast<double>(src_total);
        if (dst_total > 0) dst_role /= static_cast<double>(dst_total);

        // Port role difference: positive = dst acts as server, negative = src acts
        // as server
        result.port_role = dst_role - src_role;
    }

    // Top port: most frequently used destination port (service identifier)
    if (config.net_top_port && !dst_port_counts.empty()) {
        auto max_it = std::max_element(
            dst_port_counts.begin(), dst_port_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        result.top_port = static_cast<double>(max_it->first);
    }

    return result;
}

inline double ProtocolFeatureExtractor::classify_port_role(uint16_t port) {
    if (port < WELL_KNOWN_PORT_MAX) {
        return 1.0;  // Strong server indicator
    } else if (port < REGISTERED_PORT_MAX) {
        return 0.5;  // Weak server indicator
    } else {
        return 0.0;  // Client indicator
    }
}

inline double ProtocolFeatureExtractor::classify_protocol_role(uint8_t protocol) {
    switch (protocol) {
    case PROTO_TCP:
        return 1.0;  // Connection-oriented, session-based, typical server protocol
    case PROTO_UDP:
        return 0.5;  // Connectionless, used for both client/server patterns
    case PROTO_ICMP:
        return 0.25;  // Control/diagnostic, not a typical dependency indicator
    default:
        return 0.0;  // Unknown protocol
    }
}
