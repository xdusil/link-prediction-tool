#pragma once

#include "utils/ip/BoostIPHandler.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <boost/asio.hpp>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

/**
 * @brief Class for checking if an IP address is allowed.
 *
 * This class checks if an IP address is allowed based on a list of allowed
 * IP addresses and ranges. It also checks if an IP address is blocked based
 * on a list of blocked IP addresses and ranges. Allowed IPs take precedence
 * over blocked IPs. 
 */
class AllowedIPChecker : public IIPChecker {
public:
    /**
     * @brief Constructor for AllowedIPChecker.
     *
     * @param allowed_ips_and_ranges A vector of allowed IP addresses and ranges.
     * @param blocked_ips_and_ranges A vector of blocked IP addresses and ranges.
     */
    AllowedIPChecker(
        const std::optional<std::vector<std::string>> &allowed_ips_and_ranges,
        const std::optional<std::vector<std::string>> &blocked_ips_and_ranges);

    /**
     * @brief Constructor for AllowedIPChecker.
     *
     * @param allowed_ips_and_ranges_path The path to the file containing allowed IPs and ranges.
     * @param blocked_ips_and_ranges_path The path to the file containing blocked IPs and ranges.
     */
    AllowedIPChecker(std::optional<std::string> allowed_ips_and_ranges_path,
                     std::optional<std::string> blocked_ips_and_ranges_path);

    /**
     * @brief Check if the given IP address is allowed.
     *
     * @param ip The IP address to check.
     * @return True if the IP address is allowed, false otherwise.
     */
    bool check_ip(const std::string &ip) const override;

private:
    std::optional<BoostIPHandler> m_allowed_ips_and_ranges; // Allowed IPs and ranges
    std::optional<BoostIPHandler> m_blocked_ips_and_ranges; // Blocked IPs and ranges
};