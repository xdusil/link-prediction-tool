#pragma once

#include "graph/IGraphManager.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

/**
 * @brief Utility for collecting aggregated flow data between vertex pairs.
 *
 * Collects flow (edge) data between two vertices with configurable fields.
 * Uses std::optional for fields that may not be collected.
 *
 * "Flow" here refers to individual network flows (represented as edges
 * in the graph). A vertex pair typically has multiple flows in each direction.
 */
class FlowDataCollector {
public:
    using Timestamps = std::vector<std::chrono::milliseconds>;

    /**
     * @brief Bitmask specifying which fields to collect.
     */
    enum class Fields : uint8_t {
        None = 0,
        ForwardStartTimes = 1 << 0,
        ForwardEndTimes = 1 << 1,
        ReverseStartTimes = 1 << 2,
        ReverseEndTimes = 1 << 3,
        Duration = 1 << 4,
        // Count fields avoid timestamp-vector allocation when only edge counts are needed.
        ForwardCount = 1 << 5,
        ReverseCount = 1 << 6,

        // Common combinations
        CountsOnly = ForwardCount | ReverseCount,
        StartTimesOnly = ForwardStartTimes | ReverseStartTimes,
        ResponseTiming = ForwardEndTimes | ReverseStartTimes,
        StartTimesAndResponseTiming = StartTimesOnly | ResponseTiming,
        StartTimesAndDuration = StartTimesOnly | Duration,
        ResponseTimingAndDuration = ResponseTiming | Duration,
        AllWithoutReverseEnd = StartTimesOnly | ResponseTiming | Duration,
        All = ForwardStartTimes | ForwardEndTimes | ReverseStartTimes | ReverseEndTimes |
              Duration | CountsOnly
    };

    /**
     * @brief Aggregated flow data with optional fields.
     *
     * Fields are std::optional - accessing via value() throws std::bad_optional_access
     * if the field wasn't collected.
     */
    class FlowData {
    public:
        // Accessors (throw std::optional_access if not collected)

        const Timestamps& forward_start_times() const {
            return m_forward_start_times.value();
        }
        const Timestamps& forward_end_times() const {
            return m_forward_end_times.value();
        }
        const Timestamps& reverse_start_times() const {
            return m_reverse_start_times.value();
        }
        const Timestamps& reverse_end_times() const {
            return m_reverse_end_times.value();
        }
        double total_duration_ms() const { return m_total_duration_ms.value(); }

        // Check if fields were collected

        bool has_forward_start_times() const noexcept {
            return m_forward_start_times.has_value();
        }
        bool has_forward_end_times() const noexcept {
            return m_forward_end_times.has_value();
        }
        bool has_reverse_start_times() const noexcept {
            return m_reverse_start_times.has_value();
        }
        bool has_reverse_end_times() const noexcept {
            return m_reverse_end_times.has_value();
        }
        bool has_duration() const noexcept { return m_total_duration_ms.has_value(); }
        bool has_forward_count() const noexcept { return m_forward_count.has_value(); }
        bool has_reverse_count() const noexcept { return m_reverse_count.has_value(); }

        // Counts (stored directly when requested, otherwise derived from collected vectors)

        /**
         * @brief Number of forward flows from src to dst.
         *
         * Uses the explicit forward count when collected. Otherwise derives the
         * count from a collected forward timestamp vector.
         *
         * @throws std::bad_optional_access if no forward count or timestamp data
         * was collected.
         */
        std::size_t forward_count() const {
            if (m_forward_count)
                return *m_forward_count;
            if (m_forward_start_times)
                return m_forward_start_times->size();
            if (m_forward_end_times)
                return m_forward_end_times->size();
            throw std::bad_optional_access();
        }

        /**
         * @brief Number of reverse flows from dst to src.
         *
         * Uses the explicit reverse count when collected. Otherwise derives the
         * count from a collected reverse timestamp vector.
         *
         * @throws std::bad_optional_access if no reverse count or timestamp data
         * was collected.
         */
        std::size_t reverse_count() const {
            if (m_reverse_count)
                return *m_reverse_count;
            if (m_reverse_start_times)
                return m_reverse_start_times->size();
            if (m_reverse_end_times)
                return m_reverse_end_times->size();
            throw std::bad_optional_access();
        }

        /**
         * @brief Total flow count. Requires both directions to have been collected.
         */
        std::size_t total_count() const { return forward_count() + reverse_count(); }

        /**
         * @brief Check if any flows exist. Requires both directions collected.
         */
        bool has_flows() const { return total_count() > 0; }

        /**
         * @brief Check if communication is bidirectional. Requires both directions
         * collected.
         */
        bool is_bidirectional() const {
            return forward_count() > 0 && reverse_count() > 0;
        }

    private:
        friend class FlowDataCollector;

        std::optional<Timestamps> m_forward_start_times;
        std::optional<Timestamps> m_forward_end_times;
        std::optional<Timestamps> m_reverse_start_times;
        std::optional<Timestamps> m_reverse_end_times;
        std::optional<double> m_total_duration_ms;
        std::optional<std::size_t> m_forward_count;
        std::optional<std::size_t> m_reverse_count;
    };

    /**
     * @brief Collect flow data with fields known at compile time.
     *
     * @tparam F Bitmask of Fields to collect
     * @tparam GraphTraits The graph traits type
     * @param graph The graph manager
     * @param src Source vertex
     * @param dst Destination vertex
     * @return Collected FlowData
     */
    template <Fields F, typename GraphTraits>
    static FlowData collect(const IGraphManager<GraphTraits>& graph,
                            const typename GraphTraits::Vertex& src,
                            const typename GraphTraits::Vertex& dst);

    /**
     * @brief Collect flow data with specified fields (runtime dispatch).
     *
     * For common field combinations, dispatches to optimized compile-time versions.
     * Otherwise, collects all fields.
     *
     * @tparam GraphTraits The graph traits type
     * @param graph The graph manager
     * @param src Source vertex
     * @param dst Destination vertex
     * @param fields Bitmask of Fields to collect
     * @return Collected FlowData
     */
    template <typename GraphTraits>
    static FlowData collect(const IGraphManager<GraphTraits>& graph,
                            const typename GraphTraits::Vertex& src,
                            const typename GraphTraits::Vertex& dst, Fields fields);

    /**
     * @brief Get sorted combined start timestamps (forward + reverse).
     *
     * @param data The FlowData instance
     * @return Sorted start timestamps
     */
    static std::vector<std::chrono::milliseconds>
    get_sorted_start_times(const FlowData& data);
};

constexpr FlowDataCollector::Fields operator|(FlowDataCollector::Fields a,
                                              FlowDataCollector::Fields b) noexcept {
    return static_cast<FlowDataCollector::Fields>(static_cast<uint8_t>(a) |
                                                  static_cast<uint8_t>(b));
}

constexpr FlowDataCollector::Fields& operator|=(FlowDataCollector::Fields& a,
                                                FlowDataCollector::Fields b) noexcept {
    return a = a | b;
}

constexpr bool has_field(FlowDataCollector::Fields fields,
                         FlowDataCollector::Fields check) noexcept {
    return (static_cast<uint8_t>(fields) & static_cast<uint8_t>(check)) != 0;
}

template <FlowDataCollector::Fields F, typename GraphTraits>
FlowDataCollector::FlowData
FlowDataCollector::collect(const IGraphManager<GraphTraits>& graph,
                           const typename GraphTraits::Vertex& src,
                           const typename GraphTraits::Vertex& dst) {
    FlowData data;

    if constexpr (has_field(F, Fields::ForwardStartTimes))
        data.m_forward_start_times.emplace();
    if constexpr (has_field(F, Fields::ForwardEndTimes))
        data.m_forward_end_times.emplace();
    if constexpr (has_field(F, Fields::ReverseStartTimes))
        data.m_reverse_start_times.emplace();
    if constexpr (has_field(F, Fields::ReverseEndTimes))
        data.m_reverse_end_times.emplace();
    if constexpr (has_field(F, Fields::Duration))
        data.m_total_duration_ms = 0.0;
    if constexpr (has_field(F, Fields::ForwardCount))
        data.m_forward_count = 0;
    if constexpr (has_field(F, Fields::ReverseCount))
        data.m_reverse_count = 0;

    constexpr bool needs_forward_properties =
        has_field(F, Fields::ForwardStartTimes) ||
        has_field(F, Fields::ForwardEndTimes) || has_field(F, Fields::Duration);
    constexpr bool needs_reverse_properties =
        has_field(F, Fields::ReverseStartTimes) ||
        has_field(F, Fields::ReverseEndTimes) || has_field(F, Fields::Duration);
    constexpr bool traverse_forward =
        needs_forward_properties || has_field(F, Fields::ForwardCount);
    constexpr bool traverse_reverse =
        needs_reverse_properties || has_field(F, Fields::ReverseCount);

    // Forward flows (src -> dst)
    if constexpr (traverse_forward) {
        auto [it, end] = graph.get_out_edges(src);
        for (; it != end; ++it) {
            if (graph.get_target_vertex(*it) != dst)
                continue;

            if constexpr (has_field(F, Fields::ForwardCount))
                ++(*data.m_forward_count);
            if constexpr (needs_forward_properties) {
                const auto& props = graph.get_edge_properties(*it);

                if constexpr (has_field(F, Fields::ForwardStartTimes))
                    data.m_forward_start_times->push_back(props.start_timestamp);
                if constexpr (has_field(F, Fields::ForwardEndTimes))
                    data.m_forward_end_times->push_back(props.end_timestamp);
                if constexpr (has_field(F, Fields::Duration)) {
                    *data.m_total_duration_ms += static_cast<double>(
                        (props.end_timestamp - props.start_timestamp).count());
                }
            }
        }
    }

    // Reverse flows (dst -> src)
    if constexpr (traverse_reverse) {
        auto [it, end] = graph.get_out_edges(dst);
        for (; it != end; ++it) {
            if (graph.get_target_vertex(*it) != src)
                continue;

            if constexpr (has_field(F, Fields::ReverseCount))
                ++(*data.m_reverse_count);
            if constexpr (needs_reverse_properties) {
                const auto& props = graph.get_edge_properties(*it);

                if constexpr (has_field(F, Fields::ReverseStartTimes))
                    data.m_reverse_start_times->push_back(props.start_timestamp);
                if constexpr (has_field(F, Fields::ReverseEndTimes))
                    data.m_reverse_end_times->push_back(props.end_timestamp);
                if constexpr (has_field(F, Fields::Duration)) {
                    *data.m_total_duration_ms += static_cast<double>(
                        (props.end_timestamp - props.start_timestamp).count());
                }
            }
        }
    }

    return data;
}

template <typename GraphTraits>
FlowDataCollector::FlowData
FlowDataCollector::collect(const IGraphManager<GraphTraits>& graph,
                           const typename GraphTraits::Vertex& src,
                           const typename GraphTraits::Vertex& dst, Fields fields) {
    // Dispatch common field combinations to optimized compile-time versions
    switch (fields) {
    case Fields::None:
        return collect<Fields::None>(graph, src, dst);
    case Fields::CountsOnly:
        return collect<Fields::CountsOnly>(graph, src, dst);
    case Fields::StartTimesOnly:
        return collect<Fields::StartTimesOnly>(graph, src, dst);
    case Fields::ResponseTiming:
        return collect<Fields::ResponseTiming>(graph, src, dst);
    case Fields::Duration:
        return collect<Fields::Duration>(graph, src, dst);
    case Fields::StartTimesAndResponseTiming:
        return collect<Fields::StartTimesAndResponseTiming>(graph, src, dst);
    case Fields::StartTimesAndDuration:
        return collect<Fields::StartTimesAndDuration>(graph, src, dst);
    case Fields::ResponseTimingAndDuration:
        return collect<Fields::ResponseTimingAndDuration>(graph, src, dst);
    case Fields::AllWithoutReverseEnd:
        return collect<Fields::AllWithoutReverseEnd>(graph, src, dst);
    default:
        // For uncommon field combinations
        return collect<Fields::All>(graph, src, dst);
    }
}

inline std::vector<std::chrono::milliseconds>
FlowDataCollector::get_sorted_start_times(const FlowData& data) {
    std::vector<std::chrono::milliseconds> times;
    times.reserve(data.total_count());

    const auto& fwd = data.forward_start_times();
    const auto& rev = data.reverse_start_times();
    times.insert(times.end(), fwd.begin(), fwd.end());
    times.insert(times.end(), rev.begin(), rev.end());

    std::sort(times.begin(), times.end());
    return times;
}
