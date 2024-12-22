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
 * @param current_edge The properties of the current edge.
 * @param previous_edge The properties of the previous edge.
 * @param was_two_steps_ago Whether the target vertex was two steps ago.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_LR_opening_conditions(const auto &current_edge,
                                            const auto &previous_edge,
                                            bool was_two_steps_ago) {
    return !was_two_steps_ago &&
           (previous_edge.start_timestamp <= current_edge.start_timestamp &&
            current_edge.start_timestamp <= current_edge.end_timestamp &&
            current_edge.end_timestamp <= previous_edge.end_timestamp);
}

/**
 * @brief Condition 2: expresses the return from LR dependency
 *
 * @param current_edge The properties of the current edge.
 * @param previous_edge The properties of the previous edge.
 * @param walk_sequence The current walk sequence.
 * @param current_vertex The current vertex.
 * @param was_two_steps_ago Whether the target vertex was two steps ago.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_return_from_LR_conditions(const auto &current_edge,
                                                const auto &previous_edge,
                                                const auto &walk_sequence,
                                                const auto &current_vertex,
                                                bool was_two_steps_ago) {
    return !was_two_steps_ago &&
           utils::is_vertex_pair_in_sequence_opposite(
               walk_sequence, walk_sequence[walk_sequence.size() - 1], current_vertex) &&
           (current_edge.start_timestamp <= previous_edge.start_timestamp &&
            previous_edge.start_timestamp <= previous_edge.end_timestamp &&
            previous_edge.end_timestamp <= current_edge.end_timestamp);
}

/**
 * @brief Condition 3: expresses the opening of RR dependency that should happen
 * within the specified time ε.
 *
 * @param current_edge The properties of the current edge.
 * @param previous_edge The properties of the previous edge.
 * @param epsilon The time window for the RR dependency.
 * @param was_one_step_ago Whether the target vertex was one step ago.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_RR_opening_conditions(const auto &current_edge,
                                            const auto &previous_edge, int epsilon,
                                            bool was_one_step_ago) {
    return previous_edge.end_timestamp <= current_edge.start_timestamp &&
           !was_one_step_ago &&
           std::chrono::duration_cast<std::chrono::milliseconds>(
               current_edge.start_timestamp - previous_edge.end_timestamp)
                   .count() <= epsilon;
}

/**
 * @brief Condition 4: formalizes return over reverse edge representing reverse flow.
 *
 * @param current_edge The properties of the current edge.
 * @param previous_edge The properties of the previous edge.
 * @param walk_sequence The current walk sequence.
 * @param current_vertex The current vertex.
 * @param was_two_steps_ago Whether the target vertex was two steps ago.
 * @param epsilon_rev The time window for the reverse flow.
 * @return True if the conditions are satisfied, false otherwise.
 */
inline bool satisfies_return_over_reverse_edge_conditions(
    const auto &current_edge, const auto &previous_edge, const auto &walk_sequence,
    const auto &current_vertex, bool was_two_steps_ago, int epsilon_rev) {
    return was_two_steps_ago &&
           utils::is_vertex_pair_in_sequence_opposite(
               walk_sequence, walk_sequence[walk_sequence.size() - 1], current_vertex) &&
           (previous_edge.src_port == current_edge.dst_port &&
            previous_edge.dst_port == current_edge.src_port &&
            previous_edge.start_timestamp <= current_edge.start_timestamp &&
            std::abs(std::chrono::duration_cast<std::chrono::milliseconds>(
                         previous_edge.end_timestamp - current_edge.end_timestamp)
                         .count()) <= epsilon_rev);
}

} // namespace custom_conditions