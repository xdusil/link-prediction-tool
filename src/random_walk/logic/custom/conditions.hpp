#pragma once

#include "graph/network/NetworkGraphDefinition.hpp"
#include "utils/utils.hpp"

/**
 * @brief Custom conditions for the random walk logic.
 */
namespace custom_conditions {

/**
 * @brief Condition 1: expresses the opening of LR dependency
 *
 * @param current The properties of the current edge.
 * @param previous The properties of the previous edge.
 * @param was_two_steps_ago Whether the target vertex was two steps ago.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_LR_opening_conditions(const EdgeProperties &current,
                                            const EdgeProperties &previous,
                                            bool was_two_steps_ago) {
    return !was_two_steps_ago && (previous.start_timestamp <= current.start_timestamp &&
                                  current.start_timestamp <= current.end_timestamp &&
                                  current.end_timestamp <= previous.end_timestamp);
}

/**
 * @brief Condition 2: expresses the return from LR dependency
 * 
 * @param current The properties of the current edge.
 * @param previous The properties of the previous edge.
 * @param walk_sequence The current walk sequence.
 * @param current_vertex The current vertex.
 * @param was_two_steps_ago Whether the target vertex was two steps ago.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_return_from_LR_conditions(const EdgeProperties &current,
                                                const EdgeProperties &previous,
                                                const std::vector<Vertex> &walk_sequence,
                                                const Vertex &current_vertex,
                                                bool was_two_steps_ago) {
    return !was_two_steps_ago &&
           utils::is_vertex_pair_in_sequence_opposite(
               walk_sequence, walk_sequence[walk_sequence.size() - 1], current_vertex) &&
           (current.start_timestamp <= previous.start_timestamp &&
            previous.start_timestamp <= previous.end_timestamp &&
            previous.end_timestamp <= current.end_timestamp);
}

/**
 * @brief Condition 3: expresses the opening of RR dependency that should happen
 * within the specified time ε.
 *
 * @param current The properties of the current edge.
 * @param previous The properties of the previous edge.
 * @param epsilon The time window for the RR dependency.
 * @param was_one_step_ago Whether the target vertex was one step ago.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_RR_opening_conditions(const EdgeProperties &current,
                                            const EdgeProperties &previous, int epsilon,
                                            bool was_one_step_ago) {
    return previous.end_timestamp <= current.start_timestamp && !was_one_step_ago &&
           std::chrono::duration_cast<std::chrono::milliseconds>(current.start_timestamp -
                                                                 previous.end_timestamp)
                   .count() <= epsilon;
}

/**
 * @brief Condition 4: formalizes return over reverse edge representing reverse flow.
 *
 * @param current The properties of the current edge.
 * @param previous The properties of the previous edge.
 * @param walk_sequence The current walk sequence.
 * @param current_vertex The current vertex.
 * @param was_two_steps_ago Whether the target vertex was two steps ago.
 * @param epsilon_rev The time window for the reverse flow.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_return_over_reverse_edge_conditions(
    const EdgeProperties &current, const EdgeProperties &previous,
    const std::vector<Vertex> &walk_sequence, const Vertex &current_vertex,
    bool was_two_steps_ago, int epsilon_rev) {
    return was_two_steps_ago &&
           utils::is_vertex_pair_in_sequence_opposite(
               walk_sequence, walk_sequence[walk_sequence.size() - 1], current_vertex) &&
           (previous.src_port == current.dst_port &&
            previous.dst_port == current.src_port &&
            previous.start_timestamp <= current.start_timestamp &&
            std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
                         previous.end_timestamp - current.end_timestamp)
                         .count()) <= epsilon_rev);
}

} // namespace custom_conditions