
#include "DependencyAnalyzer.hpp"
#include "../json/JsonHelper.hpp"
#include "io/FileReader.hpp"
#include "io/FileWriter.hpp"
#include "utils/ip/IIPChecker.hpp"
#include <chrono>
#include <fstream>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define UDP_PROTOCOL 17
#define TCP_PROTOCOL 6

namespace ground_truth {

DependencyAnalyzer::DependencyAnalyzer(int n_occurrences, int epsilon,
                                       IIPChecker &allowed_ips_checker)
    : m_n_occurrences(n_occurrences), m_epsilon(epsilon),
      m_allowed_ips_checker(allowed_ips_checker) {}

void DependencyAnalyzer::parse_flow_data(const std::string &filename) {
    FileReader reader(filename);

    std::string line;
    while (reader.get_next_line(line)) {
        auto data = JsonHelper::parse_json(line);

        auto src_ip = JsonHelper::extract_value<std::string>(data, "sourceIPv4Address");
        auto dst_ip =
            JsonHelper::extract_value<std::string>(data, "destinationIPv4Address");
        auto protocol = JsonHelper::extract_value<int64_t>(data, "protocolIdentifier");
        if (!protocol || !src_ip || !dst_ip ||
            (*protocol != UDP_PROTOCOL && *protocol != TCP_PROTOCOL) ||
            !m_allowed_ips_checker.check_ip(*src_ip) ||
            !m_allowed_ips_checker.check_ip(*dst_ip)) {
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
                : (data.contains("biFlowStartMilliseconds_Rev")
                    ? JsonHelper::extract_value<int64_t>(data, "biFlowStartMilliseconds_Rev")
                    : start_forward);  // Fallback to forward timestamp

        auto end_reverse = 
            data.contains("flowEndMilliseconds_Rev")
                ? JsonHelper::extract_value<int64_t>(data, "flowEndMilliseconds_Rev")
                : (data.contains("biFlowEndMilliseconds_Rev")
                    ? JsonHelper::extract_value<int64_t>(data, "biFlowEndMilliseconds_Rev")
                    : end_forward);  // Fallback to forward timestamp

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
}

const DependencySet &DependencyAnalyzer::get_dependencies() const {
    return m_all_dependencies;
}

const DependencySet &DependencyAnalyzer::calculate_dependencies(
    const std::string &filename,
    const std::optional<std::string> output_filename /*= std::nullopt*/) {
    reset();

    // Parse flow data - build IP dictionary
    parse_flow_data(filename);

    auto direct_dependencies = determine_direct_dependencies();
    auto td2_dependencies = determine_TD2_dependencies(direct_dependencies);
    auto rr2_dependencies = determine_RR2_dependencies(direct_dependencies);
    auto td3_dependencies =
        determine_TD3_dependencies(direct_dependencies, td2_dependencies);
    auto rr3_dependencies =
        determine_RR3_dependencies(direct_dependencies, rr2_dependencies);

    if (output_filename.has_value()) {
        FileWriter writer(output_filename.value());
        writer.write(m_oss.str());
        std::cout << "Ground truth dependencies written to " << output_filename.value()
                  << std::endl;
    }

    print_dependency_stats(direct_dependencies, td2_dependencies, rr2_dependencies,
                           td3_dependencies, rr3_dependencies);

    return m_all_dependencies;
}

const DependencySet &DependencyAnalyzer::load_dependencies(const std::string &filename) {
    reset();
    FileReader reader(filename);

    std::string line;
    while (reader.get_next_line(line)) {
        // Skip empty lines
        if (line.empty())
            continue;

        // Find the opening and closing parentheses
        size_t open_paren = line.find('(');
        size_t close_paren = line.find(')');

        if (open_paren == std::string::npos || close_paren == std::string::npos) {
            continue; // Skip malformed lines
        }

        // Extract the content between parentheses
        std::string content = line.substr(open_paren + 1, close_paren - open_paren - 1);

        // Find the comma separating IPs
        size_t comma_pos = content.find(',');
        if (comma_pos == std::string::npos) {
            continue; // Skip malformed lines
        }

        // Extract source and destination IPs
        std::string src_ip = content.substr(0, comma_pos);
        std::string dst_ip = content.substr(comma_pos + 1);

        // Trim whitespace
        src_ip.erase(0, src_ip.find_first_not_of(" \t"));
        src_ip.erase(src_ip.find_last_not_of(" \t") + 1);
        dst_ip.erase(0, dst_ip.find_first_not_of(" \t"));
        dst_ip.erase(dst_ip.find_last_not_of(" \t") + 1);

        m_all_dependencies.insert({src_ip, dst_ip});
    }

    std::cout << "Loaded " << m_all_dependencies.size() << " dependencies from "
              << filename << std::endl;
    return m_all_dependencies;
}

void DependencyAnalyzer::reset() {
    m_ip_dict.clear();
    m_oss.str("");
    m_all_dependencies.clear();
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
                m_all_dependencies.insert({src_ip, target_ip});
                m_oss << "DD: (" << src_ip << ", " << target_ip << ")"
                      << "\n";
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
                    m_all_dependencies.insert({src_ip, dst_ip});
                    m_oss << "TD2: (" << src_ip << ", " << dst_ip << ")"
                          << "\n";
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
                    m_all_dependencies.insert({dst_ip, mid_ip});
                    m_oss << "RR2: (" << dst_ip << ", " << mid_ip << ")"
                          << "\n";
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
                        m_all_dependencies.insert({src_ip, dst_ip});
                        m_oss << "TD3: (" << src_ip << ", " << dst_ip << ")"
                              << "\n";
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
                        m_all_dependencies.insert({dst_ip, mid_ip});
                        m_oss << "RR3: (" << dst_ip << ", " << mid_ip << ")"
                              << "\n";
                        break;
                    }
                }
            }
        }
    }

    return dependencies;
}

void DependencyAnalyzer::print_dependency_stats(
    const DependencyList& direct_dependencies,
    const TD2DependencyMap& td2_dependencies,
    const RR2DependencyMap& rr2_dependencies,
    const TD3DependencyMap& td3_dependencies,
    const RR3DependencyMap& rr3_dependencies) const {
    
    size_t dd_count = direct_dependencies.size();
    
    size_t td2_count = 0;
    size_t rr2_count = 0;
    size_t td3_count = 0;
    size_t rr3_count = 0;
    
    for (const auto& [_, dsts] : td2_dependencies) {
        td2_count += dsts.size();
    }
    
    for (const auto& [_, dsts] : rr2_dependencies) {
        rr2_count += dsts.size();
    }
    
    for (const auto& [_, dsts] : td3_dependencies) {
        td3_count += dsts.size();
    }
    
    for (const auto& [_, dsts] : rr3_dependencies) {
        rr3_count += dsts.size();
    }
    
    std::cout << "Dependency statistics:\n"
              << "---------------------------\n"
              << "Direct dependencies (DD): " << dd_count << "\n"
              << "TD2 dependencies: " << td2_count << "\n"
              << "RR2 dependencies: " << rr2_count << "\n"
              << "TD3 dependencies: " << td3_count << "\n"
              << "RR3 dependencies: " << rr3_count << "\n"
              << "---------------------------\n"
              << "Total dependencies: " << (dd_count + td2_count + rr2_count + td3_count + rr3_count) << "\n"
              << "Unique dependencies: " << m_all_dependencies.size() << std::endl;
}
} // namespace GroundTruth
