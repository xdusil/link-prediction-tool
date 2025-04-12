#include "NetworkGraphManager.hpp"

// Add a vertex with associated properties.
Vertex NetworkGraphManager::add_vertex(const VertexProperties &properties) {
    auto [vertex, inserted] = ip_to_vertex.try_emplace(properties.ip_addr, BoostGraphManager::add_vertex(properties));
    return vertex->second;
}

// Add an edge between two vertices with associated properties.
bool NetworkGraphManager::add_edge(const Vertex &src, const Vertex &dst, const EdgeProperties &properties) {
    return BoostGraphManager::add_edge(src, dst, properties);
}

// Add an edge between two vertices with associated properties.
bool NetworkGraphManager::add_edge(const std::string &src_ip, const std::string &dst_ip, const EdgeProperties &properties) {
    auto src_it = ip_to_vertex.find(src_ip);
    auto dst_it = ip_to_vertex.find(dst_ip);

    if (src_it == ip_to_vertex.end() || dst_it == ip_to_vertex.end()) {
        return false;
    }

    return add_edge(src_it->second, dst_it->second, properties);
}

// Add an edge between two vertices with associated properties.
bool NetworkGraphManager::add_edge_and_vertex_if_not_exists(const VertexProperties &src_properties, const VertexProperties &dst_properties, const EdgeProperties &properties) {
    auto src_it = ip_to_vertex.find(src_properties.ip_addr);
    auto dst_it = ip_to_vertex.find(dst_properties.ip_addr);

    if (src_it == ip_to_vertex.end()) {
        src_it = ip_to_vertex.try_emplace(src_properties.ip_addr, BoostGraphManager::add_vertex(src_properties)).first;
    }

    if (dst_it == ip_to_vertex.end()) {
        dst_it = ip_to_vertex.try_emplace(dst_properties.ip_addr, BoostGraphManager::add_vertex(dst_properties)).first;
    }

    return add_edge(src_it->second, dst_it->second, properties);
}

// Get map of IP addresses to vertex descriptors.
const std::unordered_map<std::string, Vertex> &NetworkGraphManager::get_ip_to_vertex() const {
    return ip_to_vertex;
}