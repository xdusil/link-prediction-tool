#pragma once

#include "io/FileReader.hpp"
#include "json/JsonHelper.hpp"
#include <unordered_set>
#include <string>
#include "Types.hpp"
#include "../constrained_collections/counters/IEvictingCounter.hpp"
#include "../constrained_collections/reservoirs/ICapacityLimitedReservoir.hpp"
#include "utils/ip/IIPHandler.hpp"

/**
 * @brief Class for processing flow data from files.
 *
 * This class provides methods for reading flow data from files, updating
 * internal and external IP address counters, and adding flow edges to a
 * reservoir.
 */
class FlowProcessor {
public:

    /**
     * @brief Constructor for FlowProcessor.
     *
     * @param internal_addresses A set of internal IP addresses.
     * @param internal_counter An evicting counter for internal IP addresses.
     * @param external_counter An evicting counter for external IP addresses.
     * @param reservoir A capacity-limited reservoir for IP edges.
     * @param ip_handler An IP handler for checking IP address ranges.
     */
    FlowProcessor(const std::unordered_set<std::string>& internal_addresses,
                IEvictingCounter<IPAddress>& internal_counter,
                IEvictingCounter<IPAddress>& external_counter,
                ICapacityLimitedReservoir<IPAddress, IPEdge>& reservoir,
                IIPHandler& ip_handler);

    /**
    * @brief Processes initial flow data from a file.
    *
    * This method reads flow data from a file and updates the internal and
    * external IP address counters.
    *
    * @param filename The name of the file to read flow data from.
    */
    void process_flow_file(const std::string& filename);


    /**
     * @brief Processes filtered flow data from a file.
     *
     * This method reads flow data from a file, filters the data based on
     * the internal and external IP address counters, and adds the flow edges
     * to the reservoir.
     *
     * @param filename The name of the file to read flow data from.
     */
    void process_filtered_flows(const std::string& filename);

private:
    const std::unordered_set<std::string>& m_internal_addresses;  // Internal IP addresses
    IEvictingCounter<IPAddress>& m_internal_counter;              // Evicting counter for internal IP addresses
    IEvictingCounter<IPAddress>& m_external_counter;              // Evicting counter for external IP addresses
    ICapacityLimitedReservoir<IPAddress, IPEdge>& m_reservoir;    // Capacity-limited reservoir for IP edges
    IIPHandler& m_ip_handler;                                     // IP handler for checking IP addresses

    /**
     * @brief Updates the internal or external IP address counters.
     *
     * This method increments the count for the IP address in the appropriate
     * counter, or evicts items if the counter is at its limit.
     *
     * @param ip The IP address to update the counter for.
     */
    void update_counters(const std::string& ip);

    /**
     * @brief Parses flow data from a JSON object.
     *
     * @param data The JSON object containing flow data.
     * @return The parsed flow data as an IPEdge object.
     */
    IPEdge parse_flow_from_json(const boost::json::object& data) const;

    /**
     * @brief Adds an edge to the reservoir.
     *
     * This method adds an edge to the reservoir for the source and destination
     * IP addresses.
     *
     * @param src_ip The source IP address.
     * @param dst_ip The destination IP address.
     * @param edge The edge to add to the reservoir.
     */
    void add_edge_to_reservoir(const std::string& src_ip, const std::string& dst_ip, IPEdge& edge);

    /**
     * @brief Logs a message for missing keys in a JSON object.
     *
     * @param line The JSON object as a string.
     */
    void log_missing_keys(const std::string& line) const;
};
