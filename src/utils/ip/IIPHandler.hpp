#pragma once

#include <string>

/**
 * @brief Interface for handling IP address operations.
 *
 * This interface provides methods to check if an IP address is allowed or blocked
 * based on predefined IP ranges.
 */
class IIPHandler {
public:
    virtual ~IIPHandler() = default;

    /**
     * @brief Checks if an IP address is allowed.
     *
     * @param ip The IP address to check.
     * @return True if the IP is allowed, false otherwise.
     */
    virtual bool is_ip_allowed(const std::string& ip) const = 0;
};