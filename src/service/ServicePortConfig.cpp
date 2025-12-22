#include "ServicePortConfig.hpp"
#include "exceptions/exceptions.hpp"
#include "io/FileReader.hpp"
#include "json/JsonHelper.hpp"
#include <boost/json.hpp>
#include <stdexcept>

namespace service {

void ServicePortConfig::load(const std::string& filename) {
    FileReader reader(filename);
    std::string content;
    reader.read_all(content);

    // Parse JSON with comments allowed
    boost::json::parse_options opts;
    opts.allow_comments = true;
    opts.allow_trailing_commas = true;

    boost::json::value jv = boost::json::parse(content, {}, opts);

    if (!jv.is_object()) {
        throw ConfigurationException("Service port config must be a JSON object");
    }

    const auto& obj = jv.as_object();
    m_definitions.clear();

    for (const auto& [key, value] : obj) {
        // Skip metadata keys starting with underscore
        if (key.starts_with("_")) {
            continue;
        }

        ServiceDefinition def;
        def.type = ServiceType::from_string(key);

        if (value.is_object()) {
            const auto& service_obj = value.as_object();

            // Get description if present
            if (service_obj.contains("description")) {
                def.description = service_obj.at("description").as_string().c_str();
            }

            // Get ports
            if (service_obj.contains("ports")) {
                const auto& ports_val = service_obj.at("ports");

                if (ports_val.is_array()) {
                    // Simple array of ports: [80, 443, 8080]
                    for (const auto& port_val : ports_val.as_array()) {
                        PortEntry entry;
                        entry.port = static_cast<uint16_t>(port_val.as_int64());
                        def.ports.push_back(entry);
                    }
                } else if (ports_val.is_object()) {
                    // Object with port descriptions: {"3306": "MySQL", "5432": "PostgreSQL"}
                    for (const auto& [port_str, desc_val] : ports_val.as_object()) {
                        PortEntry entry;
                        entry.port = static_cast<uint16_t>(std::stoi(std::string(port_str)));
                        if (desc_val.is_string()) {
                            entry.description = desc_val.as_string().c_str();
                        }
                        def.ports.push_back(entry);
                    }
                }
            }

            // Get protocol-specific ports if present
            if (service_obj.contains("tcp_ports")) {
                for (const auto& port_val : service_obj.at("tcp_ports").as_array()) {
                    PortEntry entry;
                    entry.port = static_cast<uint16_t>(port_val.as_int64());
                    entry.protocol = 6; // TCP
                    def.ports.push_back(entry);
                }
            }

            if (service_obj.contains("udp_ports")) {
                for (const auto& port_val : service_obj.at("udp_ports").as_array()) {
                    PortEntry entry;
                    entry.port = static_cast<uint16_t>(port_val.as_int64());
                    entry.protocol = 17; // UDP
                    def.ports.push_back(entry);
                }
            }

        } else if (value.is_array()) {
            // Simple format: "DNS": [53]
            for (const auto& port_val : value.as_array()) {
                PortEntry entry;
                entry.port = static_cast<uint16_t>(port_val.as_int64());
                def.ports.push_back(entry);
            }
        }

        if (!def.ports.empty()) {
            m_definitions.push_back(std::move(def));
        }
    }

    build_lookup_tables();
    m_loaded = true;
}

void ServicePortConfig::build_lookup_tables() {
    m_port_to_service.clear();
    m_port_proto_to_service.clear();

    for (const auto& def : m_definitions) {
        for (const auto& entry : def.ports) {
            if (entry.protocol.has_value()) {
                // Protocol-specific mapping
                uint32_t key = make_key(entry.port, *entry.protocol);
                m_port_proto_to_service[key] = def.type;
            } else {
                // Protocol-agnostic mapping
                m_port_to_service[entry.port] = def.type;
            }
        }
    }
}

std::optional<ServiceType> ServicePortConfig::lookup(
    uint16_t port, std::optional<uint8_t> protocol) const {

    // First try protocol-specific lookup
    if (protocol.has_value()) {
        uint32_t key = make_key(port, *protocol);
        auto it = m_port_proto_to_service.find(key);
        if (it != m_port_proto_to_service.end()) {
            return it->second;
        }
    }

    // Fall back to protocol-agnostic lookup
    auto it = m_port_to_service.find(port);
    if (it != m_port_to_service.end()) {
        return it->second;
    }

    return std::nullopt;
}

} // namespace service
