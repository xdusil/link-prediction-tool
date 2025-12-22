#pragma once

#include "ServiceTypes.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace service {

/**
 * @brief Configuration for port-to-service mappings.
 *
 * Loads service definitions from a JSON configuration file.
 * Each service type maps to a list of destination ports.
 * Supports optional protocol specification (TCP/UDP).
 */
class ServicePortConfig {
public:
    /**
     * @brief Port entry with optional protocol specification.
     */
    struct PortEntry {
        uint16_t port;
        std::optional<uint8_t> protocol; // 6=TCP, 17=UDP, nullopt=any
        std::string description;         // Optional description (e.g., "PostgreSQL")
    };

    /**
     * @brief Service definition with ports and metadata.
     */
    struct ServiceDefinition {
        ServiceType type;
        std::string description;
        std::vector<PortEntry> ports;
    };

    /**
     * @brief Load configuration from a JSON file.
     * @param filename Path to the JSON configuration file.
     */
    void load(const std::string& filename);

    /**
     * @brief Look up service type for a given port.
     * @param port The destination port number.
     * @param protocol Optional protocol identifier (6=TCP, 17=UDP).
     * @return The service type, or std::nullopt if not mapped.
     */
    std::optional<ServiceType> lookup(uint16_t port,
                                      std::optional<uint8_t> protocol = std::nullopt) const;

    /**
     * @brief Get all service definitions.
     * @return Vector of all loaded service definitions.
     */
    const std::vector<ServiceDefinition>& get_definitions() const { return m_definitions; }

    /**
     * @brief Check if configuration has been loaded.
     * @return True if load() has been called successfully.
     */
    bool is_loaded() const { return m_loaded; }

    /**
     * @brief Get the number of mapped ports.
     * @return Total number of port mappings (protocol-agnostic + protocol-specific).
     */
    std::size_t port_count() const {
        return m_port_to_service.size() + m_port_proto_to_service.size();
    }

private:
    std::vector<ServiceDefinition> m_definitions;
    
    // Fast lookup: port -> service type (for protocol-agnostic lookups)
    std::unordered_map<uint16_t, ServiceType> m_port_to_service;

    // Protocol-specific lookup: (port << 8 | protocol) -> service type
    std::unordered_map<uint32_t, ServiceType> m_port_proto_to_service;
    
    bool m_loaded = false;

    /**
     * @brief Build lookup tables from definitions.
     */
    void build_lookup_tables();

    /**
     * @brief Create combined key for protocol-specific lookup.
     */
    static uint32_t make_key(uint16_t port, uint8_t protocol) {
        return (static_cast<uint32_t>(port) << 8) | protocol;
    }
};

} // namespace service
