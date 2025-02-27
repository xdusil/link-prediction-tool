#pragma once
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief Interface for a reservoir that is limited in capacity per key.
 *
 * The reservoir is limited in capacity per key, and if the capacity is
 * exceeded, probabilistic replacement of existing items is performed.
 *
 * @tparam Key The key type.
 * @tparam Value The value type.
 */
template <typename Key, typename Value>
class ICapacityLimitedReservoir {
public:
    virtual ~ICapacityLimitedReservoir() = default;

    /**
     * @brief Adds a value associated with a key to the reservoir.
     *
     * If the reservoir exceeds its capacity, probabilistically replaces
     * existing items.
     *
     * @param key The key to associate the value with.
     * @param value The value to add to the reservoir.
     */
    virtual void add(const Key &key, const Value &value) = 0;

    /**
     * @brief Retrieves all values associated with a specific key.
     *
     * @param key The key to retrieve values for.
     * @return A vector of values associated with the key.
     */
    virtual const std::vector<Value> &get(const Key &key) const = 0;

    /**
     * @brief Retrieves all keys in the reservoir.
     *
     * @return A vector of all keys in the reservoir.
     */
    virtual std::vector<Key> get_keys() const = 0;

    /**
     * @brief Checks if a key exists in the reservoir.
     *
     * @param key The key to check.
     * @return True if the key exists, false otherwise.
     */
    virtual bool contains(const Key &key) const = 0;

    /**
     * @brief Gets the current count of items stored for a specific key.
     *
     * @param key The key to count items for.
     * @return The number of items stored for the key.
     */
    virtual std::size_t get_size(const Key &key) const = 0;

    /**
     * @brief Get total number of keys in the reservoir.
     *
     * @return The total number of keys in the reservoir.
     */
    virtual std::size_t get_key_count() const = 0;

    /**
     * @brief Get the total number of items in the reservoir.
     *
     * This is the sum of the number of items stored for each key.
     * @return The total number of items in the reservoir.
     */
    virtual std::size_t get_total_size() const = 0;

    /**
     * @brief Get the maximum capacity for the reservoir.
     *
     * @return The maximum number of values allowed per key.
     */
    virtual std::size_t get_capacity() const = 0;

    /**
     * @brief Sets the seed for the reservoir's random number generator.
     *
     * @param seed The seed to set.
     */
    virtual void set_seed(unsigned int seed) = 0;

    /**
     * @brief Begin iterator for the reservoir.
     *
     * @return An iterator to the beginning of the reservoir.
     */
    virtual auto begin() const -> typename std::unordered_map<
        Key, std::pair<std::vector<Value>, std::size_t>>::const_iterator = 0;

    /**
     * @brief End iterator for the reservoir.
     *
     * @return An iterator to the end of the reservoir.
     */
    virtual auto end() const -> typename std::unordered_map<
        Key, std::pair<std::vector<Value>, std::size_t>>::const_iterator = 0;
};
