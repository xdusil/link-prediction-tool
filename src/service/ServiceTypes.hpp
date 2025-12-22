#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace service {

/**
 * @brief Network service types for edge classification.
 *
 * Categorizes network dependencies by the type of service provided
 * at the destination endpoint.
 */
struct ServiceType {
    enum Value {
        WEB_SERVER,     // HTTP/HTTPS web servers
        DATABASE,       // SQL/NoSQL databases
        CACHE,          // Caching services - Redis, Memcached
        MESSAGE_QUEUE,  // Message brokers - RabbitMQ, Kafka
        DNS,            // Domain Name System
        MAIL,           // Email services - SMTP, POP, IMAP
        FILE_TRANSFER,  // File transfer - FTP, SFTP, NFS, SMB
        REMOTE_ACCESS,  // Remote access - SSH, RDP, VNC
        MONITORING,     // Monitoring - SNMP, Prometheus
        DIRECTORY,      // Directory services - LDAP, Kerberos
        TIME_SYNC,      // Time synchronization - NTP
        OTHER,          // Mapped to known port but miscellaneous category
        UNKNOWN         // Could not classify with sufficient confidence
    };

    // Total number of service types
    static constexpr std::size_t COUNT = 13;

    // Number of classifiable service types (excluding UNKNOWN)
    static constexpr std::size_t CLASSIFIABLE_COUNT = 12;

    // String names for all service types
    static constexpr std::array<std::string_view, COUNT> NAMES = {
        "WEB_SERVER",
        "DATABASE",
        "CACHE",
        "MESSAGE_QUEUE",
        "DNS",
        "MAIL",
        "FILE_TRANSFER",
        "REMOTE_ACCESS",
        "MONITORING",
        "DIRECTORY",
        "TIME_SYNC",
        "OTHER",
        "UNKNOWN"
    };

    /**
     * @brief Convert ServiceType to string.
     * @param type The service type value.
     * @return String representation.
     */
    static std::string_view to_string(Value type) noexcept {
        return NAMES[static_cast<std::size_t>(type)];
    }

    /**
     * @brief Convert string to ServiceType.
     * @param name The service type name.
     * @return The corresponding ServiceType value.
     * @throws std::invalid_argument if name is not recognized.
     */
    static Value from_string(std::string_view name) {
        for (std::size_t i = 0; i < COUNT; ++i) {
            if (NAMES[i] == name) {
                return static_cast<Value>(i);
            }
        }
        throw std::invalid_argument("Unknown service type: " + std::string(name));
    }

    /**
     * @brief Get the index of a ServiceType for array indexing.
     * @param type The service type value.
     * @return Zero-based index.
     */
    static constexpr std::size_t to_index(Value type) noexcept {
        return static_cast<std::size_t>(type);
    }

    /**
     * @brief Convert index to ServiceType.
     * @param index Zero-based index.
     * @return The corresponding ServiceType value.
     */
    static constexpr Value from_index(std::size_t index) noexcept {
        return static_cast<Value>(index);
    }

    /**
     * @brief Check if a service type is classifiable (not UNKNOWN).
     * @param type The service type value.
     * @return True if the service type can be used in classification.
     */
    static constexpr bool is_classifiable(Value type) noexcept {
        return type != UNKNOWN;
    }

    // Default constructor - initializes to UNKNOWN
    ServiceType() : value(UNKNOWN) {}
    
    // Allow implicit construction from Value
    ServiceType(Value v) : value(v) {}
    
    // Allow implicit conversion to Value
    operator Value() const { return value; }
    
    // Underlying value
    Value value;
};

} // namespace service
