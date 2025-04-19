#pragma once

#include <optional>
#include <string>

namespace ground_truth {

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
};
} // namespace ground_truth