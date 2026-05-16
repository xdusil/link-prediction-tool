#pragma once

#include "IFlowProcessor.hpp"
#include "Types.hpp"
#include "constrained_collections/counters/IEvictingCounter.hpp"
#include "constrained_collections/reservoirs/ICapacityLimitedReservoir.hpp"
#include "io/FileReader.hpp"
#include "json/JsonHelper.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <chrono>
#include <cstddef>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vector>

/**
 * @brief Class for processing flow data from files.
 *
 * This class provides methods for reading flow data from files, selecting
 * stable endpoint samples, and adding structured flow edges to a reservoir.
 */
class FlowProcessor : public IFlowProcessor {
public:
    /**
     * @brief Constructor for FlowProcessor.
     *
     * @param internal_counter An evicting counter for internal IP addresses.
     * @param external_counter An evicting counter for external IP addresses.
     * @param reservoir A capacity-limited reservoir for IP edges.
     * @param allowed_ips_checker An IP checker for allowed IP addresses.
     * @param internal_ips_checker An IP checker for internal IP addresses.
     * @param temporal_bucket_count Number of temporal buckets used for edge sampling.
     */
    FlowProcessor(IEvictingCounter<IPAddress> &internal_counter,
                  IEvictingCounter<IPAddress> &external_counter,
                  ICapacityLimitedReservoir<IPAddress, IPEdge> &reservoir,
                  const IIPChecker &allowed_ips_checker,
                  const IIPChecker &internal_ips_checker,
                  std::size_t temporal_bucket_count);

    /**
     * @brief Processes initial flow data from a file.
     *
     * This method reads flow data from a file and computes exact endpoint
     * statistics used for deterministic endpoint retention.
     *
     * @param filename The name of the file to read flow data from.
     */
    void process_flow_file(const std::string &filename) override;

    /**
     * @brief Processes filtered flow data from a file.
     *
     * This method reads flow data from a file, filters the data based on the
     * retained endpoint sets, and adds temporally bucketed flow edges to the
     * reservoir.
     *
     * @param filename The name of the file to read flow data from.
     */
    void process_filtered_flows(const std::string &filename) override;

    /**
     * @brief Gets the count of internal IP addresses.
     *
     * @return The count of internal IP addresses.
     */
    std::size_t get_internal_addresses_count() const override;

    /**
     * @brief Gets the count of external IP addresses.
     *
     * @return The count of external IP addresses.
     */
    std::size_t get_external_addresses_count() const override;

    /**
     * @brief Gets the count of total edges in the reservoir.
     *
     * @return The count of total edges in the reservoir.
     */
    std::size_t get_total_edges_count() const override;

    /**
     * @brief Gets the total number of flows processed.
     *
     * @return The total number of flows processed.
     */
    std::size_t get_total_flows_count() const override;

private:
    struct EndpointStats {
        std::size_t flow_count = 0;
        std::unordered_set<IPAddress> peers;
        std::unordered_set<int> protocols;
        std::chrono::milliseconds first_seen = std::chrono::milliseconds::max();
        std::chrono::milliseconds last_seen = std::chrono::milliseconds::zero();
    };

    IEvictingCounter<IPAddress>
        &m_internal_counter; // Evicting counter for internal IP addresses
    IEvictingCounter<IPAddress>
        &m_external_counter; // Evicting counter for external IP addresses
    ICapacityLimitedReservoir<IPAddress, IPEdge>
        &m_reservoir;                         // Capacity-limited reservoir for IP edges
    const IIPChecker &m_allowed_ips_checker;  // Allowed IP checker
    const IIPChecker &m_internal_ips_checker; // Internal IP checker
    std::size_t m_total_flows = 0;            // Total number of flows processed
    std::size_t m_temporal_bucket_count;
    std::unordered_map<IPAddress, EndpointStats> m_internal_endpoint_stats;
    std::unordered_map<IPAddress, EndpointStats> m_external_endpoint_stats;
    std::unordered_set<IPAddress> m_selected_internal_ips;
    std::unordered_set<IPAddress> m_selected_external_ips;
    std::chrono::milliseconds m_observation_start = std::chrono::milliseconds::max();
    std::chrono::milliseconds m_observation_end = std::chrono::milliseconds::zero();

    /**
     * @brief Updates endpoint statistics used for deterministic endpoint selection.
     *
     * @param ip The endpoint whose stats are updated.
     * @param peer The peer communicating with the endpoint.
     * @param protocol The flow protocol.
     * @param start_timestamp The flow start timestamp.
     * @param end_timestamp The flow end timestamp.
     */
    void update_endpoint_stats(const std::string &ip, const std::string &peer,
                               int protocol,
                               std::chrono::milliseconds start_timestamp,
                               std::chrono::milliseconds end_timestamp);

    /**
     * @brief Selects retained internal and external endpoints after the first pass.
     */
    void finalize_endpoint_selection();

    /**
     * @brief Selects the strongest endpoints from the given stats map.
     *
     * @param endpoint_stats Exact stats collected during the first pass.
     * @param limit Number of endpoints to retain.
     * @return Set of retained endpoint IPs.
     */
    static std::unordered_set<IPAddress>
    select_top_endpoints(const std::unordered_map<IPAddress, EndpointStats> &endpoint_stats,
                         std::size_t limit);

    /**
     * @brief Checks whether the IP survived endpoint sampling.
     *
     * @param ip Endpoint IP.
     * @return true if the endpoint is retained, false otherwise.
     */
    bool is_retained_endpoint(const std::string &ip) const;

    /**
     * @brief Builds a stable reservoir key for a pair and temporal bucket.
     *
     * @param src_ip Source IP address.
     * @param dst_ip Destination IP address.
     * @param edge Edge metadata used for temporal bucketing.
     * @return Stable pair-bucket key used by the reservoir.
     */
    std::string make_edge_bucket_key(const std::string &src_ip,
                                     const std::string &dst_ip,
                                     const IPEdge &edge) const;

    /**
     * @brief Returns the temporal bucket for an edge start timestamp.
     *
     * @param timestamp Edge start timestamp.
     * @return Zero-based temporal bucket index.
     */
    std::size_t get_temporal_bucket(std::chrono::milliseconds timestamp) const;

    /**
     * @brief Parses flow data from a JSON object.
     *
     * @param data The JSON object containing flow data.
     * @return The parsed flow data as an IPEdge object.
     */
    static IPEdge parse_flow_from_json(const boost::json::object &data);

    /**
     * @brief Parses reverse flow data from a JSON object.
     *
     * @param data The JSON object containing reverse flow data.
     * @return The parsed reverse flow data as an IPEdge object.
     */
    static IPEdge parse_rev_flow_from_json(const boost::json::object &data);

    /**
     * @brief Extracts the start and end timestamps from a flow JSON object.
     *
     * @param data JSON flow object.
     * @return Start and end timestamps.
     */
    static std::pair<std::chrono::milliseconds, std::chrono::milliseconds>
    extract_time_window(const boost::json::object &data);

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
    void add_edge_to_reservoir(const std::string &src_ip, const std::string &dst_ip,
                               const IPEdge &edge);

    /**
     * @brief Checks if the JSON data contains reverse flow information.
     *
     * This method checks if the JSON object contains fields needed for reverse flow.
     *
     * @param data The JSON object to check.
     * @return true if reverse flow data exists, false otherwise.
     */
    static bool has_reverse_flow_data(const boost::json::object &data);

    /**
     * @brief Logs a message for missing keys in a JSON object.
     *
     * @param line The JSON object as a string.
     * @param line_no The line number in the file (optional).
     */
    static void log_missing_keys(const std::string &line, std::size_t line_no = 0);
};
