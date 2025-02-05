
#include "DependencyAnalyzer.hpp"
#include "../json/JsonHelper.hpp"
#include "boost/asio/ip/address.hpp"
#include "io/FileReader.hpp"
#include <chrono>
#include <fstream>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define UDP_PROTOCOL 17
#define TCP_PROTOCOL 6

namespace ground_truth {

DependencyAnalyzer::DependencyAnalyzer(int n_occurrences, int epsilon,
                                       IIPHandler &ip_handler)
    : m_n_occurrences(n_occurrences), m_epsilon(epsilon), m_ip_handler(ip_handler) {}

IPDict DependencyAnalyzer::parse_flow_data(const std::string &filename) const {
    FileReader reader(filename);
    IPDict m_ip_dict;

    std::string line;
    while (reader.get_next_line(line)) {
        auto data = JsonHelper::parse_json_line(line);

        auto src_ip = JsonHelper::extract_value<std::string>(data, "sourceIPv4Address");
        auto dst_ip =
            JsonHelper::extract_value<std::string>(data, "destinationIPv4Address");
        auto protocol = JsonHelper::extract_value<int64_t>(data, "protocolIdentifier");
        if (!protocol || !src_ip || !dst_ip ||
            (*protocol != UDP_PROTOCOL && *protocol != TCP_PROTOCOL) ||
            !m_ip_handler.is_ip_allowed(*src_ip) ||
            !m_ip_handler.is_ip_allowed(*dst_ip)) {
            continue;
        }

        auto start_forward =
            data.contains("flowStartMilliseconds")
                ? JsonHelper::extract_value<int64_t>(data, "flowStartMilliseconds")
                : JsonHelper::extract_value<int64_t>(data, "biFlowStartMilliseconds");
        auto end_forward =
            data.contains("flowEndMilliseconds")
                ? JsonHelper::extract_value<int64_t>(data, "flowEndMilliseconds")
                : JsonHelper::extract_value<int64_t>(data, "biFlowEndMilliseconds");
        auto start_reverse =
            data.contains("flowStartMilliseconds_Rev")
                ? JsonHelper::extract_value<int64_t>(data, "flowStartMilliseconds_Rev")
                : JsonHelper::extract_value<int64_t>(data, "biFlowStartMilliseconds_Rev");
        auto end_reverse =
            data.contains("flowEndMilliseconds_Rev")
                ? JsonHelper::extract_value<int64_t>(data, "flowEndMilliseconds_Rev")
                : JsonHelper::extract_value<int64_t>(data, "biFlowEndMilliseconds_Rev");

        if (!src_ip || !dst_ip || !start_forward || !end_forward || !start_reverse ||
            !end_reverse) {
            continue;
        }

        m_ip_dict[*src_ip][*dst_ip].push_back({std::chrono::milliseconds(*start_forward),
                                             std::chrono::milliseconds(*end_forward),
                                             std::chrono::milliseconds(*start_reverse),
                                             std::chrono::milliseconds(*end_reverse),
                                             static_cast<int>(*protocol)});
    }

    return m_ip_dict;
}

std::unordered_set<std::pair<IPAddress, IPAddress>, pair_hash>
DependencyAnalyzer::calculate_all_dependencies(const std::string &filename) {
    // Parse flow data
    m_ip_dict = parse_flow_data(filename);

    std::unordered_set<std::pair<IPAddress, IPAddress>, pair_hash> all_dependencies;

    auto direct_dependencies = determine_direct_dependencies();
    auto td2_dependencies = determine_TD2_dependencies(direct_dependencies);
    auto rr2_dependencies = determine_RR2_dependencies(direct_dependencies);
    auto td3_dependencies =
        determine_TD3_dependencies(direct_dependencies, td2_dependencies);
    auto rr3_dependencies =
        determine_RR3_dependencies(direct_dependencies, rr2_dependencies);

    std::fstream file("dependencies.txt", std::ios::out);
    file << oss.str();

    std::cout << "DD: " << direct_dependencies.size() << std::endl;
    std::cout << "TD2: " << td2_dependencies.size() << std::endl;
    std::cout << "RR2: " << rr2_dependencies.size() << std::endl;
    std::cout << "TD3: " << td3_dependencies.size() << std::endl;
    std::cout << "RR3: " << rr3_dependencies.size() << std::endl;

    return all_dependencies;
}

int DependencyAnalyzer::count_appearances_of_LR_dependency(
    Timestamp start_forward, Timestamp end_forward, Timestamp start_reverse,
    Timestamp end_reverse, const std::vector<EdgeProperties> &edge_properties) const {
    int count = 0;
    for (const auto &edge : edge_properties) {
        if (start_forward <= edge.start_forward && edge.end_forward <= end_forward &&
            start_reverse <= edge.start_reverse && edge.end_reverse <= end_reverse) {
            count++;
        }
    }
    return count;
}

int DependencyAnalyzer::count_appearances_of_RR_dependency(
    Timestamp start_forward, Timestamp start_reverse, Timestamp end_reverse,
    const std::vector<EdgeProperties> &edge_properties) const {
    int count = 0;
    for (const auto &edge : edge_properties) {
        if (start_forward <= start_reverse && end_reverse <= edge.start_forward &&
            edge.start_forward <= edge.start_reverse &&
            edge.start_forward - end_reverse <= std::chrono::milliseconds(m_epsilon)) {
            count++;
        }
    }
    return count;
}

DependencyList DependencyAnalyzer::determine_direct_dependencies() {
    DependencyList dependencies;

    for (const auto &[src_ip, target_map] : m_ip_dict) {
        for (const auto &[target_ip, edge_properties] : target_map) {
            if (edge_properties.size() > m_n_occurrences) {
                dependencies.emplace_back(src_ip, target_ip);
                all_dependencies.insert({src_ip, target_ip});
                oss << "DD: (" << src_ip << ", " << target_ip << ")" << std::endl;
            }
        }
    }
    return dependencies;
}

TD2DependencyMap DependencyAnalyzer::determine_TD2_dependencies(
    const DependencyList &direct_dependencies) {
    TD2DependencyMap dependencies;

    // src -> mid
    for (const auto &[src_ip, mid_ip] : direct_dependencies) {
        auto it = m_ip_dict.find(mid_ip);
        if (it == m_ip_dict.end()) {
            continue;
        }

        const auto &mid_targets = it->second;

        // mid -> dst
        for (const auto &[dst_ip, edge_properties] : mid_targets) {
            int count_appereances = 0;

            // Skip if the destination IP doesn't meet the occurrence threshold or is the
            // same as the source IP
            if (edge_properties.size() <= m_n_occurrences || dst_ip == src_ip) {
                continue;
            };

            // src -> mid (edges)
            for (const auto &edge : m_ip_dict.at(src_ip).at(mid_ip)) {
                count_appereances += count_appearances_of_LR_dependency(
                    edge.start_forward, edge.end_forward, edge.start_reverse,
                    edge.end_reverse, edge_properties);

                if (count_appereances > m_n_occurrences) {
                    dependencies[src_ip].push_back({dst_ip, mid_ip});
                    all_dependencies.insert({src_ip, dst_ip});
                    oss << "TD2: (" << src_ip << ", " << dst_ip << ")" << std::endl;
                    break;
                }
            }
        }
    }
    return dependencies;
}

RR2DependencyMap DependencyAnalyzer::determine_RR2_dependencies(
    const DependencyList &direct_dependencies) {
    RR2DependencyMap dependencies;

    for (const auto &[src_ip, mid_ip] : direct_dependencies) {
        auto src_it = m_ip_dict.find(src_ip);
        if (src_it == m_ip_dict.end()) {
            continue;
        }

        const auto &src_targets = src_it->second;

        for (const auto &[dst_ip, edge_properties] : src_targets) {
            int count_appearances = 0;

            // Skip if the destination IP doesn't meet the occurrence threshold or is the
            // same as the middle IP
            if (edge_properties.size() <= m_n_occurrences || dst_ip == mid_ip) {
                continue;
            }

            // Check properties of edges between src_ip and mid_ip
            auto mid_it = src_targets.find(mid_ip);
            if (mid_it == src_targets.end()) {
                continue;
            }

            for (const auto &edge : mid_it->second) {
                count_appearances += count_appearances_of_RR_dependency(
                    edge.start_forward, edge.start_reverse, edge.end_reverse,
                    edge_properties);

                if (count_appearances > m_n_occurrences) {
                    dependencies[dst_ip].push_back({mid_ip, src_ip});
                    all_dependencies.insert({dst_ip, mid_ip});
                    oss << "RR2: (" << dst_ip << mid_ip << ")" << std::endl;
                    break;
                }
            }
        }
    }

    return dependencies;
}

TD3DependencyMap
DependencyAnalyzer::determine_TD3_dependencies(const DependencyList &direct_dependencies,
                                               const TD2DependencyMap &td2_dependencies) {
    TD3DependencyMap dependencies;

    for (const auto &[src_ip, td2_list] : td2_dependencies) {
        for (const auto &[mid_ip, mid_ip2] : td2_list) {

            for (const auto &[potential_src, potential_dst] : direct_dependencies) {
                if (potential_src != mid_ip2 || potential_dst == src_ip ||
                    potential_dst == mid_ip) {
                    continue;
                }

                auto mid_it = m_ip_dict.find(mid_ip);
                if (mid_it == m_ip_dict.end()) {
                    continue;
                }

                // Check properties of edges between mid_ip and mid_ip2
                auto mid_to_mid2_it = mid_it->second.find(mid_ip2);
                if (mid_to_mid2_it == mid_it->second.end()) {
                    continue;
                }

                const std::string &dst_ip = potential_dst;
                int count_appearances_inner = 0;

                auto mid2_it = m_ip_dict.find(mid_ip2);
                if (mid2_it == m_ip_dict.end()) {
                    continue;
                }

                auto mid2_to_dst_it = mid2_it->second.find(dst_ip);
                if (mid2_to_dst_it == mid2_it->second.end()) {
                    continue;
                }

                for (const auto &edge : mid_to_mid2_it->second) {

                    count_appearances_inner += count_appearances_of_LR_dependency(
                        edge.start_forward, edge.end_forward, edge.start_reverse,
                        edge.end_reverse, mid2_to_dst_it->second);

                    if (count_appearances_inner > m_n_occurrences) {
                        dependencies[src_ip].push_back({dst_ip, mid_ip, mid_ip2});
                        all_dependencies.insert({src_ip, dst_ip});
                        oss << "TD3: (" << src_ip << ", " << dst_ip << ")" << std::endl;
                        break;
                    }
                }
            }
        }
    }

    return dependencies;
}

RR3DependencyMap
DependencyAnalyzer::determine_RR3_dependencies(const DependencyList &direct_dependencies,
                                               const RR2DependencyMap &rr2_dependencies) {
    RR3DependencyMap dependencies;

    for (const auto &[mid_ip2, rr2_list] : rr2_dependencies) {
        for (const auto &[mid_ip, src_ip] : rr2_list) {

            for (const auto &[potential_src, potential_dst] : direct_dependencies) {
                if (potential_src != src_ip || potential_dst == mid_ip2 ||
                    potential_dst == mid_ip) {
                    continue;
                }

                auto src_it = m_ip_dict.find(src_ip);
                if (src_it == m_ip_dict.end()) {
                    continue;
                }

                auto src_to_mid2_it = src_it->second.find(mid_ip2);
                if (src_to_mid2_it == src_it->second.end()) {
                    continue;
                }

                const std::string &dst_ip = potential_dst;
                auto src_to_dst_it = src_it->second.find(dst_ip);
                if (src_to_dst_it == src_it->second.end()) {
                    continue;
                }

                int count_appearances = 0;

                for (const auto &edge : src_to_mid2_it->second) {
                    count_appearances += count_appearances_of_RR_dependency(
                        edge.start_forward, edge.start_reverse, edge.end_reverse,
                        src_to_dst_it->second);

                    if (count_appearances > m_n_occurrences) {
                        dependencies[dst_ip].push_back({mid_ip, src_ip, mid_ip2});
                        all_dependencies.insert({dst_ip, mid_ip});
                        oss << "RR3: (" << dst_ip << ", " << mid_ip << ")" << std::endl;
                        break;
                    }
                }
            }
        }
    }

    return dependencies;
}
} // namespace GroundTruth
