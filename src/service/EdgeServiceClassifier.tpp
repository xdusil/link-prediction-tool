#pragma once

#include "EdgeServiceClassifier.hpp"
#include <iomanip>
#include <numeric>

namespace service {

template <typename GraphTraits>
EdgeServiceClassifier<GraphTraits>::EdgeServiceClassifier(
    const ServicePortConfig& port_config, const ServiceClassificationConfig& config)
    : m_port_config(port_config), m_config(config) {}

template <typename GraphTraits>
ServiceClassificationResult EdgeServiceClassifier<GraphTraits>::classify(
    const IGraphManager<GraphTraits>& graph_manager, const Vertex& src,
    const Vertex& dst) const {

    ServiceClassificationResult result;

    // Count destination port+protocol pairs from src -> dst flows
    auto port_counts = count_destination_ports(graph_manager, src, dst);

    // Check minimum flows requirement
    std::size_t total_flows = 0;
    for (const auto& [pp, count] : port_counts) {
        total_flows += count;
    }

    if (total_flows < m_config.min_flows) {
        result.service = ServiceType::UNKNOWN;
        result.confidence = 0.0;
        result.top_k_string = "";
        return result;
    }

    // Aggregate into service counts
    auto service_counts = aggregate_service_counts(port_counts);

    // Compute total mapped flows
    std::size_t total_mapped =
        std::accumulate(service_counts.begin(), service_counts.end(), std::size_t{0});

    if (total_mapped == 0) {
        result.service = ServiceType::UNKNOWN;
        result.confidence = 0.0;
        result.top_k_string = "";
        return result;
    }

    // Compute probabilities with smoothing
    auto probs = compute_probabilities(service_counts, total_mapped);

    // Find argmax
    std::size_t max_idx = 0;
    double max_prob = 0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }

    // Check confidence threshold
    result.service = (max_prob >= m_config.min_confidence)
                         ? ServiceType::from_index(max_idx)
                         : ServiceType::UNKNOWN;

    result.confidence = max_prob;
    result.top_k_string = format_top_k(probs);

    return result;
}

template <typename GraphTraits>
std::unordered_map<typename EdgeServiceClassifier<GraphTraits>::PortProtocol, std::size_t,
                   typename EdgeServiceClassifier<GraphTraits>::PortProtocolHash>
EdgeServiceClassifier<GraphTraits>::count_destination_ports(
    const IGraphManager<GraphTraits>& graph_manager, const Vertex& src,
    const Vertex& dst) const {

    std::unordered_map<PortProtocol, std::size_t, PortProtocolHash> port_counts;

    auto [it, end] = graph_manager.get_out_edges(src);
    for (; it != end; ++it) {
        if (graph_manager.get_target_vertex(*it) == dst) {
            const auto& edge = graph_manager.get_edge_properties(*it);

            // Skip if port is not set or invalid
            if (edge.dst_port <= 0)
                continue;

            uint16_t port = static_cast<uint16_t>(edge.dst_port);

            // Filter ephemeral ports
            if (port >= m_config.ephemeral_port_min)
                continue;

            uint8_t protocol = static_cast<uint8_t>(edge.protocol);
            ++port_counts[PortProtocol{port, protocol}];
        }
    }

    return port_counts;
}

template <typename GraphTraits>
std::array<std::size_t, ServiceType::CLASSIFIABLE_COUNT>
EdgeServiceClassifier<GraphTraits>::aggregate_service_counts(
    const std::unordered_map<PortProtocol, std::size_t, PortProtocolHash>& port_counts)
    const {

    std::array<std::size_t, ServiceType::CLASSIFIABLE_COUNT> service_counts{};

    for (const auto& [pp, count] : port_counts) {
        std::optional<uint8_t> proto =
            (pp.protocol != 0) ? std::optional<uint8_t>(pp.protocol) : std::nullopt;

        auto service_opt = m_port_config.lookup(pp.port, proto);
        if (service_opt.has_value()) {
            ServiceType::Value type = *service_opt;
            if (ServiceType::is_classifiable(type)) {
                service_counts[ServiceType::to_index(type)] += count;
            }
        }
        // Unmapped port+protocol pairs are ignored (they don't contribute to any service)
    }

    return service_counts;
}

template <typename GraphTraits>
std::array<double, ServiceType::CLASSIFIABLE_COUNT>
EdgeServiceClassifier<GraphTraits>::compute_probabilities(
    const std::array<std::size_t, ServiceType::CLASSIFIABLE_COUNT>& counts,
    std::size_t total) const {

    std::array<double, ServiceType::CLASSIFIABLE_COUNT> probs{};

    // Laplace smoothing: P(s) = (count(s) + alpha) / (total + alpha * K)
    double alpha = m_config.smoothing_alpha;
    double denominator =
        static_cast<double>(total) + alpha * ServiceType::CLASSIFIABLE_COUNT;

    for (std::size_t i = 0; i < ServiceType::CLASSIFIABLE_COUNT; ++i) {
        probs[i] = (static_cast<double>(counts[i]) + alpha) / denominator;
    }

    return probs;
}

template <typename GraphTraits>
std::string EdgeServiceClassifier<GraphTraits>::format_top_k(
    const std::array<double, ServiceType::CLASSIFIABLE_COUNT>& probs) const {

    // Create (probability, index) pairs and sort descending
    std::vector<std::pair<double, std::size_t>> sorted_probs;
    sorted_probs.reserve(ServiceType::CLASSIFIABLE_COUNT);

    for (std::size_t i = 0; i < ServiceType::CLASSIFIABLE_COUNT; ++i) {
        sorted_probs.emplace_back(probs[i], i);
    }

    std::partial_sort(sorted_probs.begin(),
                      sorted_probs.begin() +
                          std::min(m_config.top_k, sorted_probs.size()),
                      sorted_probs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Format as "SERVICE:0.XX;SERVICE:0.XX;..."
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    std::size_t count = std::min(m_config.top_k, sorted_probs.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (i > 0)
            oss << ";";
        oss << ServiceType::to_string(ServiceType::from_index(sorted_probs[i].second))
            << ":" << sorted_probs[i].first;
    }

    return oss.str();
}

} // namespace service
