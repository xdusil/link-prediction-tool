#include "BoostIPHandler.hpp"
#include <boost/system/error_code.hpp>
#include <iostream>

BoostIPHandler::BoostIPHandler(
    const std::optional<std::vector<std::string>> &ips_and_ranges) {
    // Parse IPs and ranges
    if (ips_and_ranges) {
        parse_ip_or_range_vec(*ips_and_ranges, m_ips_and_ranges);
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

bool BoostIPHandler::check_ip(const std::string &ip) const {
    boost::system::error_code ec;
    auto address = boost::asio::ip::make_address_v4(ip, ec);

    if (ec) {
        std::cerr << "Invalid IP address: " << ip << " - " << ec.message() << std::endl;
        return false;
    }

    return m_ips_and_ranges.ips.contains(address) ||
           is_ip_in_list(address, m_ips_and_ranges.networks);
}

std::optional<BoostIPHandler::IPVariant>
BoostIPHandler::parse_ip_or_range(const std::string &input) {
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
    unsigned long addr = address.to_ulong();

    for (const auto &network : list) {
        auto network_address = network.network().to_ulong();
        auto mask = network.netmask().to_ulong();

        // Check if the address is in the network range
        if ((addr & mask) == (network_address & mask)) {
            return true;
        }
    }

    return false;
}
