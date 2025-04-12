#pragma once

#include "graph/IGraphManager.hpp"
#include "NetworkGraphDefinition.hpp"
#include <string>
#include <unordered_map>

/**
 * @brief Interface for managing network graphs.
 *
 * This interface extends the generic graph manager interface with 
 * network-specific functionality.
 *
 * @tparam GraphTraits The graph traits type defining the graph element types.
 */
template <typename GraphTraits>
class INetworkGraphManager : public virtual IGraphManager<GraphTraits> {
public:
    // Define the graph element types
    using Vertex = typename GraphTraits::Vertex;

    virtual ~INetworkGraphManager() = default;

    /**
     * @brief Get map of IP addresses to vertex descriptors.
     *
     * @return The map of IP addresses to vertex descriptors.
     */
    virtual const std::unordered_map<std::string, Vertex> &get_ip_to_vertex() const = 0;
};