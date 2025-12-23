#pragma once

#include "ServicePortConfig.hpp"
#include "ServiceTypes.hpp"
#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace service {

/**
 * @brief Configuration parameters for service classification.
 */
struct ServiceClassificationConfig {
    bool enabled = false;                // Enable service classification
    std::string port_config_path = "";   // Path to service_ports.json
    uint16_t ephemeral_port_min = 49152; // Minimum ephemeral port
    std::size_t min_flows = 3;           // Minimum flows required for classification
    double min_confidence = 0.1;         // Minimum confidence for non-UNKNOWN result
    double smoothing_alpha = 1.0;        // Laplace smoothing parameter
    std::size_t top_k = 3;               // Number of top services to report
};

/**
 * @brief Result of service classification for an edge.
 */
struct ServiceClassificationResult {
    ServiceType::Value service = ServiceType::UNKNOWN; // Most likely service
    double confidence = 0.0;                           // Probability of the top service
    std::string top_k_string;                          // Formatted top-k string

    /**
     * @brief Check if classification was successful (not UNKNOWN).
     */
    bool is_classified() const { return ServiceType::is_classifiable(service); }
};

/**
 * @brief Classifier for determining service type of network edges.
 *
 * Analyzes destination ports from flows between two vertices to infer
 * what type of service is being provided at the destination.
 *
 * @tparam GraphTraits The graph traits type.
 */
template <typename GraphTraits>
class EdgeServiceClassifier {
public:
    using Vertex = typename GraphTraits::Vertex;

    /**
     * @brief Construct classifier with port configuration.
     * @param port_config The port-to-service mapping configuration.
     * @param config Classification parameters.
     */
    EdgeServiceClassifier(const ServicePortConfig& port_config,
                          const ServiceClassificationConfig& config = {});

    /**
     * @brief Classify the service type for an edge.
     *
     * Iterates through all flows from src to dst, counts destination ports,
     * maps them to services, and computes probabilities.
     *
     * @param graph_manager The graph manager providing access to edges.
     * @param src Source vertex.
     * @param dst Destination vertex.
     * @return Classification result with service type and confidence.
     */
    ServiceClassificationResult classify(const IGraphManager<GraphTraits>& graph_manager,
                                         const Vertex& src, const Vertex& dst) const;

private:
    const ServicePortConfig& m_port_config;
    ServiceClassificationConfig m_config;

    /**
     * @brief Port-protocol pair for classification.
     */
    struct PortProtocol {
        uint16_t port;
        uint8_t protocol; // 6=TCP, 17=UDP, 0=any

        bool operator==(const PortProtocol& other) const {
            return port == other.port && protocol == other.protocol;
        }
    };

    /**
     * @brief Hash function for PortProtocol.
     */
    struct PortProtocolHash {
        std::size_t operator()(const PortProtocol& pp) const {
            return std::hash<uint32_t>{}((static_cast<uint32_t>(pp.port) << 8) |
                                         pp.protocol);
        }
    };

    /**
     * @brief Count destination port+protocol pairs for flows from src to dst.
     * @param graph_manager The graph manager.
     * @param src Source vertex.
     * @param dst Destination vertex.
     * @return Map of (port, protocol) -> count (ephemeral ports filtered).
     */
    std::unordered_map<PortProtocol, std::size_t, PortProtocolHash>
    count_destination_ports(const IGraphManager<GraphTraits>& graph_manager,
                            const Vertex& src, const Vertex& dst) const;

    /**
     * @brief Aggregate port+protocol counts into service counts.
     * @param port_counts Map of (port, protocol) -> count.
     * @return Array of service counts.
     */
    std::array<std::size_t, ServiceType::CLASSIFIABLE_COUNT> aggregate_service_counts(
        const std::unordered_map<PortProtocol, std::size_t, PortProtocolHash>&
            port_counts) const;

    /**
     * @brief Compute probabilities with Laplace smoothing.
     */
    std::array<double, ServiceType::CLASSIFIABLE_COUNT> compute_probabilities(
        const std::array<std::size_t, ServiceType::CLASSIFIABLE_COUNT>& counts,
        std::size_t total) const;

    /**
     * @brief Format top-k services as string.
     * @param probs Array of service probabilities.
     * @return Comma-separated string of top-k services with probabilities.
     */
    std::string
    format_top_k(const std::array<double, ServiceType::CLASSIFIABLE_COUNT>& probs) const;
};

} // namespace service

#include "EdgeServiceClassifier.tpp"
