#pragma once

#include "json/JsonHelper.hpp"
#include <boost/json.hpp>
#include <chrono>
#include <cstdint>
#include <optional>

namespace utils::flow {

struct FlowTimestamps {
    std::chrono::milliseconds start;
    std::chrono::milliseconds end;
};

inline std::optional<FlowTimestamps>
extract_timestamp_pair(const boost::json::object &data, const char *start_key,
                       const char *end_key) {
    std::optional<int64_t> start =
        JsonHelper::extract_value<int64_t>(data, start_key);
    std::optional<int64_t> end = JsonHelper::extract_value<int64_t>(data, end_key);

    if (!start || !end || *end < *start) {
        return std::nullopt;
    }

    return FlowTimestamps{std::chrono::milliseconds(*start),
                          std::chrono::milliseconds(*end)};
}

inline std::optional<FlowTimestamps>
extract_timestamps(const boost::json::object &data, const char *start_key,
                   const char *end_key, const char *fallback_start_key,
                   const char *fallback_end_key) {
    std::optional<FlowTimestamps> preferred =
        extract_timestamp_pair(data, start_key, end_key);
    if (preferred) {
        return preferred;
    }

    return extract_timestamp_pair(data, fallback_start_key, fallback_end_key);
}

inline std::optional<FlowTimestamps>
extract_forward_timestamps(const boost::json::object &data) {
    return extract_timestamps(data, "flowStartMilliseconds", "flowEndMilliseconds",
                              "biFlowStartMilliseconds", "biFlowEndMilliseconds");
}

inline std::optional<FlowTimestamps>
extract_reverse_timestamps(const boost::json::object &data) {
    return extract_timestamps(data, "flowStartMilliseconds_Rev",
                              "flowEndMilliseconds_Rev",
                              "biFlowStartMilliseconds_Rev",
                              "biFlowEndMilliseconds_Rev");
}

} // namespace utils::flow
