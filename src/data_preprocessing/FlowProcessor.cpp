#include "FlowProcessor.hpp"

// Constructor
FlowProcessor::FlowProcessor(const std::unordered_set<std::string>& internal_addresses,
                             IEvictingCounter<IPAddress>& internal_counter,
                             IEvictingCounter<IPAddress>& external_counter,
                             ICapacityLimitedReservoir<IPAddress, IPEdge>& reservoir)
        : m_internal_addresses(internal_addresses),
          m_internal_counter(internal_counter),
          m_external_counter(external_counter),
          m_reservoir(reservoir) {}

// Process initial flows
void FlowProcessor::process_initial_flows(const std::string& filename) {
    FileReader reader(filename);
    std::string line;

    while (reader.get_next_line(line)) {
        try {
            auto data = JsonHelper::parse_json_line(line);
            auto src_ip = JsonHelper::extract_value<std::string>(data, "sourceIPv4Address");
            auto dst_ip = JsonHelper::extract_value<std::string>(data, "destinationIPv4Address");

            if (!src_ip || !dst_ip) {
                log_missing_keys(line);
                continue;
            }

            update_counters(*src_ip);
            update_counters(*dst_ip);

        } catch (const std::exception&) {
            // Exception already logged by FlowReader or JsonHelper
            continue;
        }
    }
}

// Process filtered flows
void FlowProcessor::process_filtered_flows(const std::string& filename) {
    FileReader reader(filename);
    std::string line;

    while (reader.get_next_line(line)) {
        try {
            auto data = JsonHelper::parse_json_line(line);
            auto src_ip = JsonHelper::extract_value<std::string>(data, "sourceIPv4Address");
            auto dst_ip = JsonHelper::extract_value<std::string>(data, "destinationIPv4Address");

            if (!src_ip || !dst_ip) {
                log_missing_keys(line);
                continue;
            }

            if ((m_internal_counter.contains(*src_ip) || m_external_counter.contains(*src_ip)) &&
                (m_internal_counter.contains(*dst_ip) || m_external_counter.contains(*dst_ip))) {
                IPEdge edge = parse_flow_from_json(data);
                add_edge_to_reservoir(*src_ip, *dst_ip, edge);
            }

        } catch (const std::exception&) {
            continue;
        }
    }
}

// Update counters
inline void FlowProcessor::update_counters(const std::string& ip) {
    if (m_internal_addresses.contains(ip)) {
        m_internal_counter.add_or_decrement(ip);
    } else {
        m_external_counter.add_or_decrement(ip);
    }
}

// Parse flow from JSON
IPEdge FlowProcessor::parse_flow_from_json(const boost::json::object& data) const {
    return {
        data.at("sourceIPv4Address").as_string().c_str(),
        data.at("destinationIPv4Address").as_string().c_str(),
        data.contains("sourceTransportPort") ? std::optional<int>{static_cast<int>(data.at("sourceTransportPort").as_int64())} : std::nullopt,
        data.contains("destinationTransportPort") ? std::optional<int>{static_cast<int>(data.at("destinationTransportPort").as_int64())} : std::nullopt,
        static_cast<int>(data.at("protocolIdentifier").as_int64()),
        std::chrono::milliseconds{data.contains("flowStartMilliseconds") ? data.at("flowStartMilliseconds").as_int64() : data.at("biFlowStartMilliseconds").as_int64()},
        std::chrono::milliseconds{data.contains("flowEndMilliseconds") ? data.at("flowEndMilliseconds").as_int64() : data.at("biFlowEndMilliseconds").as_int64()}
    };
}

// Add edge to reservoir
void FlowProcessor::add_edge_to_reservoir(const std::string& src_ip, const std::string& dst_ip, IPEdge& edge) {
    m_reservoir.add(src_ip, edge);
    std::swap(edge.src_ip, edge.dst_ip);
    std::swap(edge.src_port, edge.dst_port);
    m_reservoir.add(dst_ip, edge);
}

// Log missing keys
void FlowProcessor::log_missing_keys(const std::string& line) const {
    std::cerr << "Missing keys in JSON object: " << line << std::endl;
}

