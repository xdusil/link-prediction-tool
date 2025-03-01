#pragma once

#include "utils/ip/IIPChecker.hpp"
#include <boost/asio.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

/**
 * @brief Class for handling IP and range operations using Boost.Asio.
 *
 * This class supports mixed inputs of individual IPs and ranges.
 * This class provides methods for checking if an IP is matched in a list of IPs and
 * ranges. It also provides methods for parsing and checking IP addresses and ranges.
 * It stores a shared pointer to an IPContainer that contains the IP addresses and ranges.
 */
class BoostIPHandler : public IIPChecker {
public:
    using IPNetwork = boost::asio::ip::network_v4;
    using IPAddress = boost::asio::ip::address_v4;
    using IPVariant = std::variant<IPAddress, IPNetwork>;

    struct IPContainer {
        std::unordered_set<IPAddress> ips;
        std::vector<IPNetwork> networks;
    };

    /**
     * @brief Construct a new BoostIPHandler.
     */
    BoostIPHandler() = default;

    /**
     * @brief Constructor for BoostIPHandler.
     *
     * @param ips_and_ranges The optional vector of IPs and ranges.
     */
    BoostIPHandler(const std::optional<std::vector<std::string>> &ips_and_ranges);

    /**
     * @brief Constructor for BoostIPHandler.
     *
     * @param filename The name of the file to parse.
     */
    BoostIPHandler(const std::optional<std::string> &filename);

    /**
     * @brief Check if an IP matches stored IPs and ranges.
     *
     * @param ip The IP address to check.
     * @return True if the IP matches, false otherwise.
     */
    virtual bool check_ip(const std::string &ip) const override;

    /**
     * @brief Parse an IP or CIDR range from a string.
     *
     * @param input The string representation of an IP or CIDR range.
     * @return A parsed IPVariant or std::nullopt if invalid.
     */
    static std::optional<IPVariant> parse_ip_or_range(const std::string &input);

    /**
     * @brief Check if an IP is in a list of networks.
     *
     * @param address The IP address to check.
     * @param networks The list of IP networks.
     * @return True if the IP is in the range of any network, false otherwise.
     */
    bool is_ip_in_list(const IPAddress &address,
                       const std::vector<IPNetwork> &networks) const;

    /**
     * @brief Parse a vector of IP addresses and ranges.
     *
     * @param ip_or_range_vec The vector of IP addresses and ranges.
     * @param container The IPContainer to store the parsed IPs and ranges.
     */
    static void parse_ip_or_range_vec(const std::vector<std::string> &ip_or_range_vec,
                                      IPContainer &container);

    /**
     * @brief Parse a file containing IP addresses and ranges.
     *
     * @param filename The name of the file to parse.
     */
    void parse_file(const std::string &filename);

    /**
     * @brief Insert an IP address or range into the container.
     *
     * @param ip_or_range The IP address or range to insert.
     */
    void insert(const IPVariant &ip_or_range);

    /**
     * @brief Insert an IP address or range into the container.
     *
     * @param ip_or_range The IP address or range to insert.
     * @param container The IPContainer to insert into.
     */
    static void insert(const IPVariant &ip_or_range, IPContainer &container);

private:
    std::unique_ptr<IPContainer> m_ips_and_ranges =
        std::make_unique<IPContainer>(); // IP addresses and ranges
};
