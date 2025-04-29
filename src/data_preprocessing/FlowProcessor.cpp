#include "FlowProcessor.hpp"
#include "exceptions/exceptions.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <exception>

FlowProcessor::FlowProcessor(IEvictingCounter<IPAddress> &internal_counter,
                             IEvictingCounter<IPAddress> &external_counter,
                             ICapacityLimitedReservoir<IPAddress, IPEdge> &reservoir,
                             const IIPChecker &allowed_ips_checker,
                             const IIPChecker &internal_ips_checker)
    : m_internal_counter(internal_counter), m_external_counter(external_counter),
      m_reservoir(reservoir), m_allowed_ips_checker(allowed_ips_checker),
      m_internal_ips_checker(internal_ips_checker) {}

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

            if (!src_ip || !dst_ip) {
                log_missing_keys(line);
                continue;
            }

            // Update counters if IP is allowed
            if (m_allowed_ips_checker.check_ip(*src_ip)) {
                update_counters(*src_ip);
            }

            if (m_allowed_ips_checker.check_ip(*dst_ip)) {
                update_counters(*dst_ip);
            }

            ++m_total_flows;

        } catch (const std::exception &) {
            std::throw_with_nested(FlowProcessorException(
                "Error processing flow data from file: " + filename));
        }
    }
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

            // Add edge to reservoir if both IPs are in the counter
            if ((m_internal_counter.contains(*src_ip) ||
                 m_external_counter.contains(*src_ip)) &&
                (m_internal_counter.contains(*dst_ip) ||
                 m_external_counter.contains(*dst_ip))) {
                try {
                    IPEdge edge = parse_flow_from_json(data);
                    add_edge_to_reservoir(*src_ip, *dst_ip, edge);
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

inline void FlowProcessor::update_counters(const std::string &ip) {
    if (m_internal_ips_checker.check_ip(ip)) {
        m_internal_counter.add_or_decrement(ip);
    } else {
        m_external_counter.add_or_decrement(ip);
    }
}

IPEdge FlowProcessor::parse_flow_from_json(const boost::json::object &data) const {
    return {
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
        std::chrono::milliseconds{data.contains("flowStartMilliseconds")
                                      ? data.at("flowStartMilliseconds").as_int64()
                                      : data.at("biFlowStartMilliseconds").as_int64()},
        std::chrono::milliseconds{data.contains("flowEndMilliseconds")
                                      ? data.at("flowEndMilliseconds").as_int64()
                                      : data.at("biFlowEndMilliseconds").as_int64()}};
}

void FlowProcessor::add_edge_to_reservoir(const std::string &src_ip,
                                          const std::string &dst_ip, IPEdge &edge) {
    m_reservoir.add(src_ip, edge);
    std::swap(edge.src_ip, edge.dst_ip);
    std::swap(edge.src_port, edge.dst_port);
    m_reservoir.add(dst_ip, edge);
}

void FlowProcessor::log_missing_keys(const std::string &line,
                                     std::size_t line_no /*= 0*/) const {
    std::cerr << "Missing keys in JSON data"
              << (line_no > 0 ? " at line " + std::to_string(line_no) : "")
              << "\nData: " << line << std::endl;
}

std::size_t FlowProcessor::get_internal_addresses_count() const {
    return m_internal_counter.get_items().size();
}

std::size_t FlowProcessor::get_external_addresses_count() const {
    return m_external_counter.get_items().size();
}

std::size_t FlowProcessor::get_total_edges_count() const {
    // Count total edges
    std::size_t total_edges = 0;
    for (const auto &key : m_reservoir.get_keys()) {
        total_edges += m_reservoir.get_size(key);
    }
    return total_edges;
}

std::size_t FlowProcessor::get_total_flows_count() const { return m_total_flows; }