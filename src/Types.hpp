#pragma once

#include <chrono>
#include <optional>
#include <string>

using IPAddress = std::string;

struct IPEdge {
    std::string src_ip;
    std::string dst_ip;
    std::optional<int> src_port;
    std::optional<int> dst_port;
    int protocol;
    std::chrono::milliseconds start_timestamp;
    std::chrono::milliseconds end_timestamp;
};