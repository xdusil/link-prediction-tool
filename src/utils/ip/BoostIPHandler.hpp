#pragma once

#include "IIPHandler.hpp"
#include <boost/asio.hpp>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

/**
 * @brief Class for handling IP and range operations using Boost.Asio.
 *
 * This class supports mixed inputs of individual IPs and ranges.
 */
class BoostIPHandler : public IIPHandler {
public:
    using IPNetwork = boost::asio::ip::network_v4;
    using IPAddress = boost::asio::ip::address_v4;
    using IPVariant = std::variant<IPAddress, IPNetwork>;

    struct IPContainer {
        std::unordered_set<IPAddress> ips;
        std::vector<IPNetwork> networks;
    };

    /**
     * @brief Constructor for BoostIPHandler.
     *
     * @param allowed_ips_and_ranges A vector of allowed IP addresses and ranges.
     * @param blocked_ips_and_ranges A vector of blocked IP addresses and ranges.
     */
    BoostIPHandler(const std::optional<std::vector<std::string>> &allowed_ips_and_ranges,
                   const std::optional<std::vector<std::string>> &blocked_ips_and_ranges);

    /**
     * @brief Check if an IP is allowed.
     *
     * @param ip The IP to check.
     * @return True if the IP is allowed, false otherwise.
     */
    bool is_ip_allowed(const std::string &ip) const override;

private:
    std::optional<IPContainer> m_allowed_ips_and_ranges; // Allowed IPs and ranges
    std::optional<IPContainer> m_blocked_ips_and_ranges; // Blocked IPs and ranges

    /**
     * @brief Parse an IP or CIDR range from a string.
     *
     * @param input The string representation of an IP or CIDR range.
     * @return A parsed IPVariant or std::nullopt if invalid.
     */
    std::optional<IPVariant> parse_ip_or_range(const std::string &input) const;

    /**
     * @brief Check if an IP is in a list of IPs or ranges.
     *
     * @param address The IP address to check.
     * @param list The list of IPs or ranges.
     * @return True if the IP is in the list, false otherwise.
     */
    bool is_ip_in_list(const IPAddress &address,
                       const std::vector<IPNetwork> &networks) const;

    /**
     * @brief Parse a vector of IP addresses and ranges.
     *
     * @param ip_or_range_vec The vector of IP addresses and ranges.
     * @param container The IPContainer to store the parsed IPs and ranges.
     */
    void parse_ip_or_range_vec(const std::vector<std::string> &ip_or_range_vec,
                               IPContainer &container);
};
