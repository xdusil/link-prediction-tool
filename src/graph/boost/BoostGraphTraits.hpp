#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

/**
 * @brief Traits for a graph managed by the Boost Graph Library.
 *
 * This struct defines the types used to represent the elements of a graph managed by the
 * Boost Graph Library.
 *
 * @tparam Graph The type of the graph to manage.
 */
template <typename Graph>
struct BoostGraphTraits {
    // Define the graph element types
    using GraphType = Graph;
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = typename boost::graph_traits<Graph>::edge_descriptor;
    using VertexPropertyType = boost::property_map<Graph, boost::vertex_bundle_t>::type;
    using VertexProperties = boost::property_traits<VertexPropertyType>::value_type;
    using EdgePropertyType = boost::property_map<Graph, boost::edge_bundle_t>::type;
    using EdgeProperties = boost::property_traits<EdgePropertyType>::value_type;
    using vertex_iterator = typename boost::graph_traits<Graph>::vertex_iterator;
    using edge_iterator = typename boost::graph_traits<Graph>::edge_iterator;
    using out_edge_iterator = typename boost::graph_traits<Graph>::out_edge_iterator;
};