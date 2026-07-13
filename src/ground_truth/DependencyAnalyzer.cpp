
#include "DependencyAnalyzer.hpp"
#include "../json/JsonHelper.hpp"
#include "io/FileReader.hpp"
#include "io/FileWriter.hpp"
#include "utils/flow/FlowTimestampExtractor.hpp"
#include "utils/ip/AllowedIPChecker.hpp"
#include "utils/ip/IIPChecker.hpp"
#include "utils/string/StringUtils.hpp"
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

namespace {

EdgeProperties make_edge_properties(const utils::flow::FlowTimestamps &forward_timestamps,
                                    std::optional<utils::flow::FlowTimestamps> reverse_timestamps,
                                    int protocol) {
    const std::optional<Timestamp> reverse_start =
        reverse_timestamps ? std::make_optional(reverse_timestamps->start)
                           : std::nullopt;
    const std::optional<Timestamp> reverse_end =
        reverse_timestamps ? std::make_optional(reverse_timestamps->end)
                           : std::nullopt;

    return {
        forward_timestamps.start,
        forward_timestamps.end,
        reverse_start,
        reverse_end,
        protocol};
}

DependencyType parse_dependency_type(const std::string &value) {
    if (value == "DD") {
        return DependencyType::DD;
    }
    if (value == "TD2") {
        return DependencyType::TD2;
    }
    if (value == "RR2") {
        return DependencyType::RR2;
    }
    if (value == "TD3") {
        return DependencyType::TD3;
    }
    if (value == "RR3") {
        return DependencyType::RR3;
    }

    return DependencyType::Unknown;
}

} // namespace

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

        const std::optional<utils::flow::FlowTimestamps> forward_timestamps =
            utils::flow::extract_forward_timestamps(data);
        const std::optional<utils::flow::FlowTimestamps> reverse_timestamps =
            utils::flow::extract_reverse_timestamps(data);

        if (!src_ip || !dst_ip || !forward_timestamps) {
            continue;
        }

        m_ip_dict[*src_ip][*dst_ip].push_back(make_edge_properties(
            *forward_timestamps, reverse_timestamps, static_cast<int>(*protocol)));
    }
}

const DependencySet &DependencyAnalyzer::get_dependencies() const {
    return m_all_dependencies;
}

ProjectionStats DependencyAnalyzer::calculate_projection_stats(
    const std::unordered_set<IPAddress> &retained_ips) const {
    ProjectionStats stats;

    for (const auto &[dependency, types] : m_dependency_types) {
        ++stats.total_dependencies;
        for (const auto type : types) {
            ++stats.total_by_type[to_index(type)];
        }

        if (!retained_ips.contains(dependency.first) ||
            !retained_ips.contains(dependency.second)) {
            continue;
        }

        ++stats.retained_dependencies;
        for (const auto type : types) {
            ++stats.retained_by_type[to_index(type)];
        }
    }

    return stats;
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

        // Add CSV header
        writer.write_line("src_ip,dst_ip,dependency_type");

        // Write dependencies to file
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
    std::cout << "Loading dependencies from " << filename << std::endl;

    std::string line;
    size_t line_number = 0;
    size_t invalid_lines = 0;

    // Skip header line
    if (reader.get_next_line(line)) {
        line_number++;
        // Check if it looks like a header
        if (line.find("source_ip") != std::string::npos ||
            line.find("src_ip") != std::string::npos) {
            // It's a header, continue to next line
        } else {
            // Process this line as it's not a header
            if (!process_dependency_line(line)) {
                std::cerr << "Warning: Line " << line_number
                          << " has invalid format: " << line << std::endl;
                invalid_lines++;
            }
        }
    }

    // Process remaining lines
    while (reader.get_next_line(line)) {
        line_number++;
        // Skip empty lines
        if (line.empty())
            continue;

        // Try to process in CSV format
        if (!process_dependency_line(line)) {
            std::cerr << "Warning: Line " << line_number
                      << " has invalid format: " << line << std::endl;
            invalid_lines++;
        }
    }

    if (invalid_lines > 0) {
        std::cerr << "Warning: " << invalid_lines
                  << " lines had invalid format and were skipped." << std::endl;
    }

    std::cout << "Loaded " << m_all_dependencies.size() << " dependencies from "
              << filename << std::endl;
    return m_all_dependencies;
}

bool DependencyAnalyzer::process_dependency_line(const std::string &line) {
    std::string src_ip;
    std::string dst_ip;
    std::string dependency_type_token;
    auto dependency_type = DependencyType::Unknown;

    // Find the first comma
    size_t first_comma = line.find(',');
    if (first_comma == std::string::npos) {
        return false; // Invalid line
    }

    // Find the second comma
    size_t second_comma = line.find(',', first_comma + 1);

    src_ip = line.substr(0, first_comma);
    if (second_comma == std::string::npos) {
        dst_ip = line.substr(first_comma + 1);
    } else {
        dst_ip = line.substr(first_comma + 1, second_comma - first_comma - 1);
        dependency_type_token = line.substr(second_comma + 1);
    }

    // Trim whitespace
    utils::trim_in_place(src_ip);
    utils::trim_in_place(dst_ip);
    utils::trim_in_place(dependency_type_token);

    // Validate IP addresses
    if (!AllowedIPChecker::is_valid_ipv4(src_ip) ||
        !AllowedIPChecker::is_valid_ipv4(dst_ip)) {
        return false; // Invalid IP address
    }

    if (!dependency_type_token.empty()) {
        dependency_type = parse_dependency_type(dependency_type_token);
    }

    register_dependency(src_ip, dst_ip, dependency_type);
    return true;
}

void DependencyAnalyzer::reset() {
    m_ip_dict.clear();
    m_oss.str("");
    m_all_dependencies.clear();
    m_dependency_types.clear();
}

void DependencyAnalyzer::register_dependency(const IPAddress &src_ip,
                                             const IPAddress &dst_ip,
                                             DependencyType type) {
    const auto dependency = std::make_pair(src_ip, dst_ip);
    m_all_dependencies.insert(dependency);
    m_dependency_types[dependency].insert(type);
}

int DependencyAnalyzer::count_appearances_of_LR_dependency(
    Timestamp start_forward, Timestamp end_forward,
    std::optional<Timestamp> start_reverse,
    std::optional<Timestamp> end_reverse,
    const std::vector<EdgeProperties> &edge_properties) const {
    if (!start_reverse || !end_reverse) {
        return 0;
    }

    int count = 0;
    for (const auto &edge : edge_properties) {
        if (!edge.start_reverse || !edge.end_reverse) {
            continue;
        }

        if (start_forward <= edge.start_forward && edge.end_forward <= end_forward &&
            *start_reverse <= *edge.start_reverse && *edge.end_reverse <= *end_reverse) {
            count++;
        }
    }
    return count;
}

int DependencyAnalyzer::count_appearances_of_RR_dependency(
    Timestamp start_forward, std::optional<Timestamp> start_reverse,
    std::optional<Timestamp> end_reverse,
    const std::vector<EdgeProperties> &edge_properties) const {
    if (!start_reverse || !end_reverse) {
        return 0;
    }

    int count = 0;
    for (const auto &edge : edge_properties) {
        if (!edge.start_reverse) {
            continue;
        }

        if (start_forward <= *start_reverse && *end_reverse <= edge.start_forward &&
            edge.start_forward <= *edge.start_reverse &&
            edge.start_forward - *end_reverse <= std::chrono::milliseconds(m_epsilon)) {
            count++;
        }
    }
    return count;
}

DependencyList DependencyAnalyzer::determine_direct_dependencies() {
    DependencyList dependencies;

    for (const auto &[src_ip, target_map] : m_ip_dict) {
        for (const auto &[target_ip, edge_properties] : target_map) {
            if (meets_occurrence_threshold(edge_properties.size())) {
                dependencies.emplace_back(src_ip, target_ip);
                register_dependency(src_ip, target_ip, DependencyType::DD);
                m_oss << src_ip << "," << target_ip << ",DD\n";
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
            if (!meets_occurrence_threshold(edge_properties.size()) || dst_ip == src_ip) {
                continue;
            };

            // src -> mid (edges)
            for (const auto &edge : m_ip_dict.at(src_ip).at(mid_ip)) {
                count_appereances += count_appearances_of_LR_dependency(
                    edge.start_forward, edge.end_forward, edge.start_reverse,
                    edge.end_reverse, edge_properties);

                if (meets_occurrence_threshold(count_appereances)) {
                    dependencies[src_ip].push_back({dst_ip, mid_ip});
                    register_dependency(src_ip, dst_ip, DependencyType::TD2);
                    m_oss << src_ip << "," << dst_ip << ",TD2\n";
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
            if (!meets_occurrence_threshold(edge_properties.size()) || dst_ip == mid_ip) {
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

                if (meets_occurrence_threshold(count_appearances)) {
                    dependencies[dst_ip].push_back({mid_ip, src_ip});
                    register_dependency(dst_ip, mid_ip, DependencyType::RR2);
                    m_oss << dst_ip << "," << mid_ip << ",RR2\n";
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

                    if (meets_occurrence_threshold(count_appearances_inner)) {
                        dependencies[src_ip].push_back({dst_ip, mid_ip, mid_ip2});
                        register_dependency(src_ip, dst_ip, DependencyType::TD3);
                        m_oss << src_ip << "," << dst_ip << ",TD3\n";
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

                    if (meets_occurrence_threshold(count_appearances)) {
                        dependencies[dst_ip].push_back({mid_ip, src_ip, mid_ip2});
                        register_dependency(dst_ip, mid_ip, DependencyType::RR3);
                        m_oss << dst_ip << "," << mid_ip << ",RR3\n";
                        break;
                    }
                }
            }
        }
    }

    return dependencies;
}

void DependencyAnalyzer::print_dependency_stats(
    const DependencyList &direct_dependencies, const TD2DependencyMap &td2_dependencies,
    const RR2DependencyMap &rr2_dependencies, const TD3DependencyMap &td3_dependencies,
    const RR3DependencyMap &rr3_dependencies) const {

    size_t dd_count = direct_dependencies.size();

    size_t td2_count = 0;
    size_t rr2_count = 0;
    size_t td3_count = 0;
    size_t rr3_count = 0;

    for (const auto &[_, dsts] : td2_dependencies) {
        td2_count += dsts.size();
    }

    for (const auto &[_, dsts] : rr2_dependencies) {
        rr2_count += dsts.size();
    }

    for (const auto &[_, dsts] : td3_dependencies) {
        td3_count += dsts.size();
    }

    for (const auto &[_, dsts] : rr3_dependencies) {
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
              << "Total dependencies: "
              << (dd_count + td2_count + rr2_count + td3_count + rr3_count) << "\n"
              << "Unique dependencies: " << m_all_dependencies.size() << std::endl;
}
} // namespace GroundTruth
