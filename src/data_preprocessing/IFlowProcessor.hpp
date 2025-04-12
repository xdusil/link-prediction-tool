#pragma once

#include "Types.hpp"
#include <string>

/**
 * @brief Interface for processing flow data from files.
 *
 * This interface provides methods for reading flow data from files, processing flows,
 * and retrieving statistics about the processed data.
 */
class IFlowProcessor {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~IFlowProcessor() = default;

    /**
     * @brief Processes initial flow data from a file.
     *
     * This method reads flow data from a file and updates the internal data structures
     * for IP address tracking.
     *
     * @param filename The name of the file to read flow data from.
     */
    virtual void process_flow_file(const std::string &filename) = 0;

    /**
     * @brief Processes filtered flow data from a file.
     *
     * This method reads flow data from a file, applies filtering,
     * and updates the underlying data structures accordingly.
     *
     * @param filename The name of the file to read flow data from.
     */
    virtual void process_filtered_flows(const std::string &filename) = 0;

    /**
     * @brief Gets the count of internal IP addresses.
     *
     * @return The count of internal IP addresses.
     */
    virtual std::size_t get_internal_addresses_count() const = 0;

    /**
     * @brief Gets the count of external IP addresses.
     *
     * @return The count of external IP addresses.
     */
    virtual std::size_t get_external_addresses_count() const = 0;

    /**
     * @brief Gets the count of total edges.
     *
     * @return The count of total edges.
     */
    virtual std::size_t get_total_edges_count() const = 0;

    /**
     * @brief Gets the total number of flows processed.
     *
     * @return The total number of flows processed.
     */
    virtual std::size_t get_total_flows_count() const = 0;
};