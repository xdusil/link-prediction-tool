#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <chrono>
#include <string>

/**
 * @brief Properties for vertices in the network graph.
 */
struct VertexProperties {
    std::string ip_addr; // IP address for the node

    VertexProperties() = default;
    VertexProperties(const std::string &ip_addr) : ip_addr(ip_addr) {}
};

/**
 * @brief Properties for edges in the network graph.
 */
struct EdgeProperties {
    std::chrono::milliseconds start_timestamp;
    std::chrono::milliseconds end_timestamp;
    int protocol;
    int src_port;
    int dst_port;

    EdgeProperties()
        : start_timestamp(0), end_timestamp(0), src_port(0), dst_port(0), protocol(0) {}

    EdgeProperties(const std::chrono::milliseconds &date_first,
                   const std::chrono::milliseconds &date_last, int protocol, int src_port,
                   int dst_port)
        : start_timestamp(date_first), end_timestamp(date_last), protocol(protocol),
          src_port(src_port), dst_port(dst_port) {}
};

/**
 * @brief Graph type for the network graph.
 *
 * This graph is a directed graph with multiple edges between the same nodes.
 * Each vertex has an IP address, and each edge has a start and end timestamp,
 * protocol, and source and destination port numbers.
 */
using Graph = boost::adjacency_list<boost::multisetS, // Allow multiple edges between the
                                                      // same nodes
                                    boost::vecS,      // Use a vector to store vertices
                                    boost::directedS, // Directed graph
                                    VertexProperties, // Properties for each vertex
                                    EdgeProperties,   // Properties for each edge
                                    boost::allow_parallel_edge_tag // Allow parallel edges
                                    >;

/**
 * @brief Vertex descriptor type for the network graph.
 */
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;

/**
 * @brief Edge descriptor type for the network graph.
 */
using Edge = boost::graph_traits<Graph>::edge_descriptor;