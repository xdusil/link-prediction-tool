#pragma once

#include "graph/boost/manager/BoostGraphManager.hpp"
#include "NetworkGraphDefinition.hpp"

/**
 * @brief Graph manager implementation for network graphs.
 *
 * This class provides methods for adding vertices and edges to a network graph.
 * The graph is based on the definitions in `NetworkGraphDefinition.hpp`.
 */
class NetworkGraphManager : public BoostGraphManager<Graph> {
public:
    // Constructor
    NetworkGraphManager() = default;

    // Destructor
    ~NetworkGraphManager() override = default;

    /**
     * @brief Add a vertex with associated properties.
     *
     * @param properties Properties of the vertex to add.
     * @return The added vertex descriptor.
     */
    Vertex add_vertex(const VertexProperties &properties) override;

    /**
     * @brief Add an edge between two vertices with associated properties.
     *
     * @param src The source vertex.
     * @param dst The destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    bool add_edge(const Vertex &src, const Vertex &dst,
                  const EdgeProperties &properties) override;

    /**
     * @brief Add an edge between two vertices with associated properties.
     *
     * @param src_ip The source IP address.
     * @param dst_ip The destination IP address.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    bool add_edge(const std::string &src_ip, const std::string &dst_ip,
                  const EdgeProperties &properties);

    /**
     * @brief Add an edge between two vertices with associated properties.
     * If the vertices do not exist, they are added to the graph.
     *
     * @param src_properties Properties of the source vertex.
     * @param dst_properties Properties of the destination vertex.
     * @param properties Properties of the edge to add.
     * @return True if the edge was successfully added, false otherwise.
     */
    bool add_edge_and_vertex_if_not_exists(const VertexProperties &src_properties,
                                           const VertexProperties &dst_properties,
                                           const EdgeProperties &properties);

    /**
     * @brief Get map of IP addresses to vertex descriptors.
     *
     * @return The map of IP addresses to vertex descriptors.
     */
    const std::unordered_map<std::string, Vertex> &get_ip_to_vertex() const {
        return ip_to_vertex;
    }

private:
    std::unordered_map<std::string, Vertex>
        ip_to_vertex; // Map IP addresses to vertex descriptors
};
