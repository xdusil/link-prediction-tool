#include "AllowedIPChecker.hpp"
#include <boost/system/error_code.hpp>
#include <iostream>

AllowedIPChecker::AllowedIPChecker(
    const std::optional<std::vector<std::string>> &allowed_ips_and_ranges,
    const std::optional<std::vector<std::string>> &blocked_ips_and_ranges) {
        if (allowed_ips_and_ranges) {
            m_allowed_ips_and_ranges = BoostIPHandler(*allowed_ips_and_ranges);
        }

        if (blocked_ips_and_ranges) {
            m_blocked_ips_and_ranges = BoostIPHandler(*blocked_ips_and_ranges);
        }
    }

bool AllowedIPChecker::check_ip(const std::string &ip) const {
    boost::system::error_code ec;
    auto address = boost::asio::ip::make_address_v4(ip, ec);

    if (ec) {
        std::cerr << "Invalid IP address: " << ip << " - " << ec.message() << std::endl;
        return false;
    }

    // Check if IP is in allowed list
    if (m_allowed_ips_and_ranges) {
        return m_allowed_ips_and_ranges->check_ip(ip);
    }

    // Check if IP is in blocked list
    if (m_blocked_ips_and_ranges) {
        return !m_blocked_ips_and_ranges->check_ip(ip);
    }

    // If no lists are provided, allow all IPs
    return true;
}