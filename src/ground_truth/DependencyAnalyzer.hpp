#pragma once

#include "ground_truth/IDependencyAnalyzer.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <chrono>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Namespace for ground truth dependency analyzis.
 *
 * This namespace contains classes and functions for analyzing dependencies in network
 * flows.
 */
namespace ground_truth {

// Base types
using Timestamp = std::chrono::milliseconds;
using IPAddress = std::string;

// Edge related types
struct EdgeProperties {
    Timestamp start_forward;
    Timestamp end_forward;
    Timestamp start_reverse;
    Timestamp end_reverse;
    int protocol;
};

// Container types
using IPDict =
    std::unordered_map<IPAddress,
                       std::unordered_map<IPAddress, std::vector<EdgeProperties>>>;
using DependencyList = std::vector<std::pair<IPAddress, IPAddress>>;

// Structures for extended dependencies
struct TD2Dependency {
    IPAddress destination;
    IPAddress middle;
};

struct RR2Dependency {
    IPAddress depends_on;
    IPAddress originator;
};

struct TD3Dependency {
    IPAddress destination;
    IPAddress middle_fst;
    IPAddress middle_snd;
};

struct RR3Dependency {
    IPAddress depends_on;
    IPAddress originator;
    IPAddress middle;
};

// Hash function for pairs
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const noexcept {
        // Asymmetric hash: (A, B) and (B, A)
        auto h1 = std::hash<T1>()(pair.first);
        auto h2 = std::hash<T2>()(pair.second);
        return h1 ^ (h2 + (h2 << 1)); // h1 XOR (3 * h2)
    }
};

// Alias for complex dependency maps
using TD2DependencyMap = std::unordered_map<IPAddress, std::vector<TD2Dependency>>;
using RR2DependencyMap = std::unordered_map<IPAddress, std::vector<RR2Dependency>>;
using TD3DependencyMap = std::unordered_map<IPAddress, std::vector<TD3Dependency>>;
using RR3DependencyMap = std::unordered_map<IPAddress, std::vector<RR3Dependency>>;
using DependencySet = std::unordered_set<std::pair<IPAddress, IPAddress>, pair_hash>;
using DependencyTypeSet = std::set<DependencyType>;

// Main class for analyzing dependencies in network flows
class DependencyAnalyzer : public IDependencyAnalyzer<DependencySet> {
public:
    /**
     * @brief Constructor to initialize the dependency zer.
     *
     * @param n_occurrences Minimum number of occurrences required to confirm a
     * dependency.
     * @param epsilon Maximum allowable time difference between flows for RR dependencies.
     * @param allowed_ips_checker The IP checker to check if an IP is allowed.
     */
    DependencyAnalyzer(int n_occurrences, int epsilon, IIPChecker &allowed_ips_checker);

    /**
     * @brief Calculate all dependencies between IP addresses.
     *
     * @param filename The name of the file to parse.
     * @param output_filename The name of the file to write the dependencies to.
     * @return A set of all dependencies.
     */
    const DependencySet &calculate_dependencies(
        const std::string &filename,
        std::optional<std::string> output_filename = std::nullopt) override;

    /**
     * @brief Load dependencies from a file.
     *
     * @param filename The name of the file to load dependencies from.
     * @return A set of all dependencies.
     */
    const DependencySet &load_dependencies(const std::string &filename) override;

    /**
     * @brief Get all dependencies.
     *
     * @return A set of all dependencies.
     */
    const DependencySet &get_dependencies() const override;

    /**
     * @brief Calculate projection statistics for a given set of retained IPs.
     *
     * @param retained_ips The set of IP addresses that are retained in the graph.
     * @return Projection statistics.
     */
    ProjectionStats calculate_projection_stats(
        const std::unordered_set<IPAddress> &retained_ips) const override;

private:
    /**
     * @brief Determine direct dependencies between IP addresses.
     *
     * A direct dependency is a dependency between two IP addresses that have a
     * minimum number of occurrences.
     * @return A list of direct dependencies.
     */
    DependencyList determine_direct_dependencies();

    /**
     * @brief Determine TD2 dependencies between IP addresses.
     *
     * @param direct_dependencies A list of direct dependencies.
     * @return A map of TD2 dependencies.
     */
    TD2DependencyMap
    determine_TD2_dependencies(const DependencyList &direct_dependencies);

    /**
     * @brief Determine RR2 dependencies between IP addresses.
     *
     * @param direct_dependencies A list of direct dependencies.
     * @return A map of RR2 dependencies.
     */
    RR2DependencyMap
    determine_RR2_dependencies(const DependencyList &direct_dependencies);
    TD3DependencyMap determine_TD3_dependencies(const DependencyList &direct_dependencies,
                                                const TD2DependencyMap &td2_dependencies);

    /**
     * @brief Determine RR3 dependencies between IP addresses.
     *
     * @param direct_dependencies A list of direct dependencies.
     * @param rr2_dependencies A map of RR2 dependencies.
     * @return A map of RR3 dependencies.
     */
    RR3DependencyMap determine_RR3_dependencies(const DependencyList &direct_dependencies,
                                                const RR2DependencyMap &rr2_dependencies);

    /**
     * @brief Parse flow data from a file.
     *
     * @param filename The name of the file to parse.
     */
    void parse_flow_data(const std::string &filename);

    /**
     * @brief Count the number of appearances of a LR dependency.
     *
     * Let A, B, C be IP addresses.
     * @param start_forward Start timestamp of the forward flow from A to B.
     * @param end_forward End timestamp of the forward flow from A to B.
     * @param start_reverse Start timestamp of the reverse flow from B to A.
     * @param end_reverse End timestamp of the reverse flow from B to A.
     * @param edge_properties Properties of all possible bidirectional edges from B to C.
     * @return The number of appearances of the LR dependency.
     */
    int count_appearances_of_LR_dependency(
        Timestamp start_forward, Timestamp end_forward, Timestamp start_reverse,
        Timestamp end_reverse, const std::vector<EdgeProperties> &edge_properties) const;

    /**
     * @brief Count the number of appearances of a RR dependency.
     *
     * Let A, B, C be IP addresses. There are 2 flows from A to B and A to C.
     * @param start_forward Start timestamp of the forward flow from A to B.
     * @param start_reverse Start timestamp of the reverse flow from B to A.
     * @param end_reverse End timestamp of the reverse flow from B to A.
     * @param edge_properties Properties of all possible bidirectional edges from A to C.
     * @return The number of appearances of the RR dependency.
     */
    int count_appearances_of_RR_dependency(
        Timestamp start_forward, Timestamp start_reverse, Timestamp end_reverse,
        const std::vector<EdgeProperties> &edge_properties) const;

    /**
     * @brief Reset the dependency analyzer.
     */
    void reset();

    /**
     * @brief Print dependency statistics.
     *
     * @param direct_dependencies A list of direct dependencies.
     * @param td2_dependencies A map of TD2 dependencies.
     * @param rr2_dependencies A map of RR2 dependencies.
     * @param td3_dependencies A map of TD3 dependencies.
     * @param rr3_dependencies A map of RR3 dependencies.
     */
    void print_dependency_stats(const DependencyList &direct_dependencies,
                                const TD2DependencyMap &td2_dependencies,
                                const RR2DependencyMap &rr2_dependencies,
                                const TD3DependencyMap &td3_dependencies,
                                const RR3DependencyMap &rr3_dependencies) const;

    /**
     * @brief Parse a CSV line and process the dependency.
     *
     * @param line The line to process.
     * @return True if the line was processed successfully, false otherwise.
     */
    bool process_dependency_line(const std::string &line);

    const int
        m_n_occurrences; // Minimum number of occurrences required to confirm a dependency
    const int
        m_epsilon; // Maximum allowable time difference between flows for dependencies
    const IIPChecker &m_allowed_ips_checker; // IP checker to check if an IP is allowed
    std::ostringstream m_oss;                // Output stream for logging
    std::unordered_set<std::pair<IPAddress, IPAddress>, pair_hash>
        m_all_dependencies; // Set of all dependencies
    IPDict m_ip_dict;       // Dictionary of IP addresses and their corresponding flows
};

} // namespace GroundTruth
