#include "FlowProcessor.hpp"
#include "exceptions/exceptions.hpp"
#include "utils/flow/FlowTimestampExtractor.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <algorithm>
#include <exception>
#include <iostream>
#include <tuple>

FlowProcessor::FlowProcessor(IEvictingCounter<IPAddress> &internal_counter,
                             IEvictingCounter<IPAddress> &external_counter,
                             ICapacityLimitedReservoir<IPAddress, IPEdge> &reservoir,
                             const IIPChecker &allowed_ips_checker,
                             const IIPChecker &internal_ips_checker,
                             std::size_t temporal_bucket_count)
    : m_internal_counter(internal_counter), m_external_counter(external_counter),
      m_reservoir(reservoir), m_allowed_ips_checker(allowed_ips_checker),
      m_internal_ips_checker(internal_ips_checker),
      m_temporal_bucket_count(std::max<std::size_t>(1, temporal_bucket_count)) {}

void FlowProcessor::process_flow_file(const std::string &filename) {
    FileReader reader(filename);
    std::string line;

    while (reader.get_next_line(line)) {
        try {
            auto data = JsonHelper::parse_json(line);
            auto src_ip =
                JsonHelper::extract_value<std::string>(data, "sourceIPv4Address");
            auto dst_ip =
                JsonHelper::extract_value<std::string>(data, "destinationIPv4Address");
            auto protocol = JsonHelper::extract_value<int64_t>(data, "protocolIdentifier");

            if (!src_ip || !dst_ip || !protocol) {
                log_missing_keys(line);
                continue;
            }

            const bool src_allowed = m_allowed_ips_checker.check_ip(*src_ip);
            const bool dst_allowed = m_allowed_ips_checker.check_ip(*dst_ip);
            if (src_allowed && dst_allowed) {
                const std::optional<utils::flow::FlowTimestamps> timestamps =
                    utils::flow::extract_forward_timestamps(data);
                if (!timestamps) {
                    continue;
                }

                update_endpoint_stats(*src_ip, *dst_ip, static_cast<int>(*protocol),
                                      timestamps->start, timestamps->end);
                update_endpoint_stats(*dst_ip, *src_ip, static_cast<int>(*protocol),
                                      timestamps->start, timestamps->end);

                m_observation_start = std::min(m_observation_start, timestamps->start);
                m_observation_end = std::max(m_observation_end, timestamps->end);
            }

            ++m_total_flows;

        } catch (const std::exception &) {
            std::throw_with_nested(FlowProcessorException(
                "Error processing flow data from file: " + filename));
        }
    }

    finalize_endpoint_selection();
}

void FlowProcessor::process_filtered_flows(const std::string &filename) {
    FileReader reader(filename);
    std::string line;
    std::size_t line_no = 0;

    while (reader.get_next_line(line)) {
        try {
            auto data = JsonHelper::parse_json(line);
            auto src_ip =
                JsonHelper::extract_value<std::string>(data, "sourceIPv4Address");
            auto dst_ip =
                JsonHelper::extract_value<std::string>(data, "destinationIPv4Address");

            ++line_no;
            if (!src_ip || !dst_ip) {
                continue;
            }

            if (is_retained_endpoint(*src_ip) && is_retained_endpoint(*dst_ip)) {
                try {
                    const std::optional<IPEdge> edge = parse_flow_from_json(data);
                    if (!edge) {
                        continue;
                    }

                    add_edge_to_reservoir(*src_ip, *dst_ip, *edge);

                    if (has_reverse_flow_data(data)) {
                        const std::optional<IPEdge> rev_edge =
                            parse_rev_flow_from_json(data);
                        if (rev_edge) {
                            add_edge_to_reservoir(*dst_ip, *src_ip, *rev_edge);
                        }
                    }
                } catch (const std::exception &) {
                    log_missing_keys(line, line_no);
                    continue;
                }
            }

        } catch (const std::exception &) {
            std::throw_with_nested(FlowProcessorException(
                "Error processing filtered flow data from file: " + filename));
        }
    }
}

void FlowProcessor::update_endpoint_stats(
    const std::string &ip, const std::string &peer, int protocol,
    std::chrono::milliseconds start_timestamp,
    std::chrono::milliseconds end_timestamp) {
    auto &endpoint_stats =
        m_internal_ips_checker.check_ip(ip) ? m_internal_endpoint_stats[ip]
                                            : m_external_endpoint_stats[ip];

    ++endpoint_stats.flow_count;
    endpoint_stats.peers.insert(peer);
    endpoint_stats.protocols.insert(protocol);
    endpoint_stats.first_seen = std::min(endpoint_stats.first_seen, start_timestamp);
    endpoint_stats.last_seen = std::max(endpoint_stats.last_seen, end_timestamp);
}

void FlowProcessor::finalize_endpoint_selection() {
    m_selected_internal_ips =
        select_top_endpoints(m_internal_endpoint_stats, m_internal_counter.get_limit());
    m_selected_external_ips =
        select_top_endpoints(m_external_endpoint_stats, m_external_counter.get_limit());

    std::cout << "Endpoint sampling complete.\n"
              << "  Internal retained: " << m_selected_internal_ips.size() << " / "
              << m_internal_endpoint_stats.size() << "\n"
              << "  External retained: " << m_selected_external_ips.size() << " / "
              << m_external_endpoint_stats.size() << "\n"
              << "  Observation window: "
              << (m_observation_start == std::chrono::milliseconds::max()
                      ? 0
                      : m_observation_start.count())
              << " -> " << m_observation_end.count() << " ms" << std::endl;
}

std::unordered_set<IPAddress> FlowProcessor::select_top_endpoints(
    const std::unordered_map<IPAddress, EndpointStats> &endpoint_stats,
    std::size_t limit) {
    struct RankedEndpoint {
        IPAddress ip;
        std::size_t flow_count = 0;
        std::size_t peer_count = 0;
        std::size_t protocol_count = 0;
        std::chrono::milliseconds temporal_span = std::chrono::milliseconds::zero();
    };

    std::vector<RankedEndpoint> ranked_endpoints;
    ranked_endpoints.reserve(endpoint_stats.size());

    for (const auto &[ip, stats] : endpoint_stats) {
        const auto temporal_span =
            stats.first_seen == std::chrono::milliseconds::max()
                ? std::chrono::milliseconds::zero()
                : stats.last_seen - stats.first_seen;
        ranked_endpoints.push_back({ip, stats.flow_count, stats.peers.size(),
                                    stats.protocols.size(), temporal_span});
    }

    std::sort(ranked_endpoints.begin(), ranked_endpoints.end(),
              [](const RankedEndpoint &lhs, const RankedEndpoint &rhs) {
                  // Rank endpoint statistics descending, then IP ascending.
                  return std::tie(lhs.peer_count, lhs.flow_count, lhs.protocol_count,
                                  lhs.temporal_span, rhs.ip) >
                         std::tie(rhs.peer_count, rhs.flow_count, rhs.protocol_count,
                                  rhs.temporal_span, lhs.ip);
              });

    std::unordered_set<IPAddress> selected_endpoints;
    selected_endpoints.reserve(std::min(limit, ranked_endpoints.size()));
    for (std::size_t i = 0; i < ranked_endpoints.size() && i < limit; ++i) {
        selected_endpoints.insert(ranked_endpoints[i].ip);
    }

    return selected_endpoints;
}

bool FlowProcessor::is_retained_endpoint(const std::string &ip) const {
    if (m_internal_ips_checker.check_ip(ip)) {
        return m_selected_internal_ips.contains(ip);
    }

    return m_selected_external_ips.contains(ip);
}

std::optional<IPEdge>
FlowProcessor::parse_flow_from_json(const boost::json::object &data) {
    const std::optional<utils::flow::FlowTimestamps> timestamps =
        utils::flow::extract_forward_timestamps(data);
    if (!timestamps) {
        return std::nullopt;
    }

    return IPEdge{
        data.at("sourceIPv4Address").as_string().c_str(),
        data.at("destinationIPv4Address").as_string().c_str(),
        data.contains("sourceTransportPort")
            ? std::optional<int>{static_cast<int>(
                  data.at("sourceTransportPort").as_int64())}
            : std::nullopt,
        data.contains("destinationTransportPort")
            ? std::optional<int>{static_cast<int>(
                  data.at("destinationTransportPort").as_int64())}
            : std::nullopt,
        static_cast<int>(data.at("protocolIdentifier").as_int64()),
        timestamps->start,
        timestamps->end};
}

std::optional<IPEdge>
FlowProcessor::parse_rev_flow_from_json(const boost::json::object &data) {
    const std::optional<utils::flow::FlowTimestamps> timestamps =
        utils::flow::extract_reverse_timestamps(data);
    if (!timestamps) {
        return std::nullopt;
    }

    return IPEdge{
        data.at("destinationIPv4Address").as_string().c_str(),
        data.at("sourceIPv4Address").as_string().c_str(),
        data.contains("destinationTransportPort")
            ? std::optional<int>{static_cast<int>(
                  data.at("destinationTransportPort").as_int64())}
            : std::nullopt,
        data.contains("sourceTransportPort")
            ? std::optional<int>{static_cast<int>(
                  data.at("sourceTransportPort").as_int64())}
            : std::nullopt,
        static_cast<int>(data.at("protocolIdentifier").as_int64()),
        timestamps->start,
        timestamps->end};
}

void FlowProcessor::add_edge_to_reservoir(const std::string &src_ip,
                                          const std::string &dst_ip,
                                          const IPEdge &edge) {
    m_reservoir.add(make_edge_bucket_key(src_ip, dst_ip, edge), edge);
}

std::string FlowProcessor::make_edge_bucket_key(const std::string &src_ip,
                                                const std::string &dst_ip,
                                                const IPEdge &edge) const {
    return src_ip + "->" + dst_ip + "#" +
           std::to_string(get_temporal_bucket(edge.start_timestamp));
}

std::size_t FlowProcessor::get_temporal_bucket(
    std::chrono::milliseconds timestamp) const {
    if (m_temporal_bucket_count <= 1 ||
        m_observation_start == std::chrono::milliseconds::max() ||
        m_observation_end <= m_observation_start) {
        return 0;
    }

    const auto observation_span = m_observation_end - m_observation_start;
    const auto clamped_offset =
        std::clamp(timestamp - m_observation_start, std::chrono::milliseconds::zero(),
                   observation_span);

    const auto bucket = static_cast<std::size_t>(
        (clamped_offset.count() * static_cast<long long>(m_temporal_bucket_count)) /
        std::max<long long>(1, observation_span.count() + 1));

    return std::min(bucket, m_temporal_bucket_count - 1);
}

bool FlowProcessor::has_reverse_flow_data(const boost::json::object &data) {
    if (data.contains("flowStartMilliseconds_Rev") &&
        data.contains("flowEndMilliseconds_Rev")) {
        return true;
    }

    if (data.contains("biFlowStartMilliseconds_Rev") &&
        data.contains("biFlowEndMilliseconds_Rev")) {
        return true;
    }

    return false;
}

void FlowProcessor::log_missing_keys(const std::string &line,
                                     std::size_t line_no /*= 0*/) {
    std::cerr << "Missing keys in JSON data"
              << (line_no > 0 ? " at line " + std::to_string(line_no) : "")
              << "\nData: " << line << std::endl;
}

std::size_t FlowProcessor::get_internal_addresses_count() const {
    return m_selected_internal_ips.size();
}

std::size_t FlowProcessor::get_external_addresses_count() const {
    return m_selected_external_ips.size();
}

std::size_t FlowProcessor::get_total_edges_count() const {
    std::size_t total_edges = 0;
    for (const auto &key : m_reservoir.get_keys()) {
        total_edges += m_reservoir.get_size(key);
    }
    return total_edges;
}

std::size_t FlowProcessor::get_total_flows_count() const { return m_total_flows; }
