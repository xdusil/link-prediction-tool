#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

namespace ground_truth {

enum class DependencyType : std::uint8_t {
    DD = 0,
    TD2,
    RR2,
    TD3,
    RR3,
    Unknown
};

constexpr std::size_t DEPENDENCY_TYPE_COUNT = 6;

using DependencyTypeCounts = std::array<std::size_t, DEPENDENCY_TYPE_COUNT>;

struct ProjectionStats {
    std::size_t total_dependencies = 0;
    std::size_t retained_dependencies = 0;
    DependencyTypeCounts total_by_type{};
    DependencyTypeCounts retained_by_type{};
};

inline constexpr std::array<DependencyType, DEPENDENCY_TYPE_COUNT>
all_dependency_types() {
    return {DependencyType::DD,      DependencyType::TD2, DependencyType::RR2,
            DependencyType::TD3,     DependencyType::RR3,
            DependencyType::Unknown};
}

inline std::size_t to_index(DependencyType type) {
    return static_cast<std::size_t>(type);
}

inline const char *to_string(DependencyType type) {
    switch (type) {
    case DependencyType::DD:
        return "DD";
    case DependencyType::TD2:
        return "TD2";
    case DependencyType::RR2:
        return "RR2";
    case DependencyType::TD3:
        return "TD3";
    case DependencyType::RR3:
        return "RR3";
    case DependencyType::Unknown:
        return "Unknown";
    }

    return "Unknown";
}

/**
 * @brief Interface for dependency analyzis.
 *
 * This interface defines the methods for calculating and loading dependencies between
 * IP addresses.
 *
 * @tparam DependencySet The type of the set of dependencies.
 */
template <typename DependencySet>
class IDependencyAnalyzer {
public:
    virtual ~IDependencyAnalyzer() = default;

    /**
     * @brief Calculate all dependencies between IP addresses.
     *
     * @param filename The name of the file to parse.
     * @param output_filename The name of the file to write the dependencies to.
     * @return A set of all dependencies.
     */
    virtual const DependencySet &
    calculate_dependencies(const std::string &filename,
                           std::optional<std::string> output_filename = std::nullopt) = 0;

    /**
     * @brief Load dependencies from a file.
     *
     * @param filename The name of the file to load dependencies from.
     * @return A set of all dependencies.
     */
    virtual const DependencySet &load_dependencies(const std::string &filename) = 0;

    /**
     * @brief Get all dependencies.
     *
     * @return A set of all dependencies.
     */
    virtual const DependencySet &get_dependencies() const = 0;

    /**
     * @brief Calculate how much of the reference truth is represented in a
     * retained vertex universe.
     *
     * A dependency is retained when both its source and destination IPs are
     * present in the retained graph/candidate universe.
     *
     * @param retained_ips Set of retained IP addresses.
     * @return Projection statistics.
     */
    virtual ProjectionStats calculate_projection_stats(
        const std::unordered_set<std::string> &retained_ips) const = 0;
};
} // namespace ground_truth
