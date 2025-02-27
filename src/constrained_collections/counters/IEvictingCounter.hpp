#pragma once

#include <unordered_map>

/**
 * @brief Interface for an evicting counter that tracks items and their counts
 * with a size constraint.
 *
 * This interface defines methods for adding items or incrementing their counts,
 * evicting items when the size limit is reached, and retrieving the items and
 * their counts.
 *
 * @tparam Key       The type of the items to count.
 * @tparam Container The type of the container used to store the items and their
 * counts.
 */
template <typename Key, typename Container = std::unordered_map<Key, int>>
class IEvictingCounter {
public:
    virtual ~IEvictingCounter() = default;

    /**
     * @brief Add an item or increment its count. If the size limit is reached,
     * decrements and evicts items as needed.
     *
     * @param key The item to add or increment.
     */
    virtual void add_or_decrement(const Key &key) = 0;

    /**
     * @brief Check if the counter contains a specific key.
     *
     * @param key The item to check.
     * @return True if the item exists, false otherwise.
     */
    virtual bool contains(const Key &key) const = 0;

    /**
     * @brief Retrieve all items and their counts.
     *
     * @return A container of items and their counts.
     */
    virtual const Container &get_items() const = 0;

    /**
     * @brief Retrieve all items and their counts, and clear the counter.
     *
     * It is recommended to use move semantics to avoid copying the container.
     *
     * @return A container of items and their counts.
     */
    virtual Container get_items_and_clear() = 0;

    /**
     * @brief Get the limit of the counter.
     *
     * @return The maximum size of the counter.
     */
    virtual std::size_t get_limit() const;

    /**
     * @brief Get the size of the counter.
     *
     * @return The number of items in the counter.
     */
    virtual std::size_t get_size() const;

    /**
     * @brief Clear the counter.
     */
    virtual void clear() = 0;
};