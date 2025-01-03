#include "BoostIPHandler.hpp"
#include <boost/system/error_code.hpp>
#include <iostream>

BoostIPHandler::BoostIPHandler(
    const std::optional<std::vector<std::string>> &allowed_ips_and_ranges,
    const std::optional<std::vector<std::string>> &blocked_ips_and_ranges) {
    // Parse allowed IPs and ranges
    if (allowed_ips_and_ranges) {
        m_allowed_ips_and_ranges = IPContainer();
        parse_ip_or_range_vec(*allowed_ips_and_ranges, *m_allowed_ips_and_ranges);
    }

    // Parse blocked IPs and ranges
    if (blocked_ips_and_ranges) {
        m_blocked_ips_and_ranges = IPContainer();
        parse_ip_or_range_vec(*blocked_ips_and_ranges, *m_blocked_ips_and_ranges);
    }
}

void BoostIPHandler::parse_ip_or_range_vec(
    const std::vector<std::string> &ip_or_range_vec, IPContainer &container) {
    for (const auto &ip_or_range : ip_or_range_vec) {
        auto parsed = parse_ip_or_range(ip_or_range);
        if (parsed) {
            if (std::holds_alternative<IPAddress>(*parsed)) {
                container.ips.insert(std::get<IPAddress>(*parsed));
            } else if (std::holds_alternative<IPNetwork>(*parsed)) {
                container.networks.push_back(std::get<IPNetwork>(*parsed));
            }
        } else {
            std::cerr << "Invalid IP or range: " << ip_or_range << std::endl;
        }
    }
}

bool BoostIPHandler::is_ip_allowed(const std::string &ip) const {
    boost::system::error_code ec;
    auto address = boost::asio::ip::make_address_v4(ip, ec);

    if (ec) {
        std::cerr << "Invalid IP address: " << ip << " - " << ec.message() << std::endl;
        return false;
    }

    // Check if IP is in allowed list
    if (m_allowed_ips_and_ranges) {
        if (m_allowed_ips_and_ranges->ips.contains(address)) {
            return true;
        }

        if (is_ip_in_list(address, m_allowed_ips_and_ranges->networks)) {
            return true;
        }
    }

    // Check if IP is in blocked list
    if (m_blocked_ips_and_ranges) {
        if (m_blocked_ips_and_ranges->ips.contains(address)) {
            return false;
        }

        if (is_ip_in_list(address, m_blocked_ips_and_ranges->networks)) {
            return false;
        }
    }

    // If IP is not in either list, allow it
    return true;
}

std::optional<BoostIPHandler::IPVariant>
BoostIPHandler::parse_ip_or_range(const std::string &input) const {
    boost::system::error_code ec;

    // Try parsing as a network (CIDR range)
    auto network = boost::asio::ip::make_network_v4(input, ec);
    if (!ec) {
        return network;
    }

    // Try parsing as an individual IP address
    auto address = boost::asio::ip::make_address_v4(input, ec);
    if (!ec) {
        return address;
    }

    // If neither worked, return nullopt
    return std::nullopt;
}

bool BoostIPHandler::is_ip_in_list(const IPAddress &address,
                                   const std::vector<IPNetwork> &list) const {
    for (const auto &network : list) {
        if (network.is_host() && network.network() == address) {
            return true;
        }

        if (!network.is_host() &&
            network.is_subnet_of(IPNetwork(address, network.prefix_length()))) {
            return true;
        }
    }

    return false;
}
