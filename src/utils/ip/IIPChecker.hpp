#pragma once

#include <string>

/**
 * @brief Interface for checking IP addresses against a specific criterion.
 *
 * This interface defines a contract for classes that can determine whether
 * a given IP address satisfies a particular condition (e.g., being internal,
 * being blocked).
 */
class IIPChecker {
public:

    virtual ~IIPChecker() = default;

    /**
     * @brief Checks if the given IP address satisfies the criterion.
     *
     * @param ip The IP address to check.
     * @return True if the IP address satisfies the criterion, false otherwise.
     */
    virtual bool check_ip(const std::string& ip) const = 0;
};